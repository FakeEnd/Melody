# from run_bce.py
import argparse
import os
import sys
from functools import partial
import random
from typing import Dict, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import loguru
import numpy as np
import time

from echo_logger import dumps_json
from loguru import logger
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
import threading
import atexit

import resource

from torch.utils.data import DataLoader
from tqdm import tqdm

from _config import pdir
from dataset_util import CellRegionDataset
from global_constants import track_39_bigwig_file_names
from melodyG2.check_utils import extract_tissue_name_from_track_name


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))



sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# from _config import pdir



start_time_str = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--auto-resume', action='store_true', help='Automatically resume latest checkpoint. If no checkpoint is available, do not set it.')
parser.add_argument('--project', type=str, default='Melody', help='Base directory for data and checkpoints')
parser.add_argument('--lab_name', type=str, default=f'scgpt_stage_2_chunk_415208_lr_0.001_{start_time_str}', help='Name of the lab for logging')
parser.add_argument('--from_ckpt', type=str,
                    default=pdir + "/data/checkpoints/Melody-G1.pth"
                    , help='Path to the checkpoint file to load')
parser.add_argument('--appointed_csv_dataset', type=str, default=None, help='Path to the CSV file for the dataset')
parser.add_argument('--use_cg_loss', default=True, action='store_true', help='Use CG loss in training')
parser.add_argument('--use_avg_loss', default=True, action='store_true', help='Use average loss in training')
parser.add_argument('--wcg', type=float, default=0.001, help='Weight for CG loss')
parser.add_argument('--wavg', type=float, default=1, help='Weight for average loss')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default: 64')
parser.add_argument('--window_size', type=int, default=10000, help='Window size, default: 10000')
parser.add_argument('--seed', type=int, default=5, help='Random seed, default: 4')
parser.add_argument('--genome_path', type=str,
                    default=pdir + '/data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
                    help='Genome .fa file path')
parser.add_argument('--bigwigs_dir', type=str, default=pdir + '/data/bigwigs/', help='BigWigs directory')
# parser.add_argument('--bigwigs_files', type=str, default=['GSM5652317_Blood-B-Z000000UB.hg38.bigwig'], nargs='+', help='BigWigs file')
parser.add_argument('--cpg_cls_bin_width', type=int, default=5,
                    help='Bin width for CpG count classification, default: 5')
parser.add_argument('--cpg_cls_n', type=int, default=7, help='Max number of classes for CpG counts, default: 7')
parser.add_argument('--use_advanced_focal_loss', action='store_true', help='Use Advanced focal loss in training')
parser.add_argument('--use_fixed_focal_cpg_loss',  default=True, action='store_true', help='Use Fixed focal loss in training')
parser.add_argument('--low_methy_cpg_focal_weight', type=float, default=32.0, help='Focal loss low_methy_cpg_focal_weight weight, default: 32.0')
parser.add_argument('--any_cpg_focal_weight', type=float, default=8.0, help='Focal loss any_cpg_focal_weight weight, default: 8.0')
parser.add_argument('--side_head_structure', type=str, default='max_pool_no_softplus', help='Side head structure')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate, default: 0.001')
parser.add_argument('--meth_pre_process_func', type=str, default='None',
                    help='Method to preprocess, default: None')
parser.add_argument('--train_hour', type=int, default=720, help='Max train hour, default: 72')

# one_stage_ckpt
parser.add_argument('--one_stage_ckpt', type=str, default=None,
                    help='Path to the one stage model checkpoint, default: None. If provided, will use this checkpoint for the 2 stage of training.')
parser.add_argument('--use_39_track', action='store_true', help='Use 39 track, default: False. If true, will use 39 track for training.')
parser.add_argument('--model_cls', type=str, default='MelodyG',)
parser.add_argument('--no-compile', default=False, action='store_false', dest='compile', help='Do not compile the model, default: Compile. If --no-compile, will not compile the model.')
parser.add_argument('--test' , action='store_true', help='Test mode, default: False. If true, will run the test mode.')
parser.add_argument('--cell_embedding_path', type=str, default=pdir + '/embeddings/scgpt',)
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from stateless import sigmoid_first, get_bigwig_filepaths, check_args, config_model_loss_fn, \
    update_eval_logs, get_score_metrics, get_best_step_and_metrics, methy2bin, load_ckpt, print_model_info, save_model, \
    get_pre_process_func, get_embedding, extract_tissue_name_from_path

from selene_sdk.sequences import Genome
from selene_util import GenomicSignalFeatures, RandomPositions, RandomPositionsSampler, SamplerDataLoader
from cell_embedding import MelodyG
from basic_blocks import make_cpg_and_valid_and_low_methy_mask_multi_track, get_alpha_tensor, \
    get_real_pred_and_cg_avg_loss_from_raw_model_output, collect_track_losses, mask_cell_region
from wandb_worker import WandbWorker
from check_utils import check_runs_return_multiple_pic_dict_batched_calc_total


import basic_blocks

args = parser.parse_args()

args.bigwigs_files = track_39_bigwig_file_names
args = check_args(args)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
flag_use_focal = args.use_fixed_focal_cpg_loss or args.use_advanced_focal_loss
func_ = sigmoid_first


BATCH_SIZE, WINDOW_SIZE, SEED, DEVICE = args.batch_size, args.window_size, args.seed, 'cuda'
bigwig_files, track_names = get_bigwig_filepaths(dir_=args.bigwigs_dir,
                                                 filenames=args.bigwigs_files,
                                                 fallback_dirs=[pdir + '/data/bigwigs/',])
track_names = [extract_tissue_name_from_track_name(track_name) for track_name in track_names]
pre_process_func: Callable = get_pre_process_func(args.meth_pre_process_func)

# bigwig_files = bigwig_files[:3]
# track_names = track_names[:3]
genome = Genome(input_path=args.genome_path, blacklist_regions='hg38')
genome.get = genome.get_encoding_from_coords
methy_data = {
    extract_tissue_name_from_path(m): GenomicSignalFeatures([m], features=[m], shape=(10000,)) for m in bigwig_files
}

cell_embeddings = get_embedding(bigwig_files, args.cell_embedding_path)


if args.model_cls == 'MelodyG':
    model = MelodyG(args, n_track=1)
else:
    raise ValueError(f"Unknown model class: {args.model_cls}")

model = model.to(DEVICE)
config_model_loss_fn(model, args=args)
if args.from_ckpt: load_ckpt(model, args.from_ckpt)
if args.compile:
    model.compile()
    basic_blocks.calc_cg_and_avg_loss_variable_len = torch.compile(basic_blocks.calc_cg_and_avg_loss_variable_len)
    make_cpg_and_valid_and_low_methy_mask_multi_track = torch.compile(make_cpg_and_valid_and_low_methy_mask_multi_track)

print_model_info(model)

test_track = ['Pancreas-Delta', 'Blood-Granulocytes', 'Blood-Monocytes', 'Aorta-Endothel', 'Cortex-Neuron']


def train():
    LOGS_TOTAL, SCORES_TOTAL = {}, {}
    train_dataset = CellRegionDataset(pdir + "cell_zones/train_data.json",
                                   genome=genome, methy_dataset_dict=methy_data, embedding_dict=cell_embeddings,
                                      test_track=test_track)
    train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)

    wdb = WandbWorker(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # i, error_num, err_max = 0, 0, 300
    past_losses = []
    start_time = datetime.now()
    train_hour, save_minutes = args.train_hour, 180
    next_checkpoint_time = start_time + timedelta(minutes=180)  # Checkpoints every x minutes
    end_time = start_time + timedelta(hours=train_hour)  # Total run time is x hours
    for i in range(999999):
        for batch in tqdm(train_dataloader, desc=f"step {i}"):
            # i += 1
            # if i > 20:
            #     break
            optimizer.zero_grad()  # Reset gradients for next accumulation
            current_time = datetime.now()
            # Check if total running time has exceeded 8 hours
            if current_time >= end_time:
                logger.info(f"Exiting due to time limit.")
                save_model(model, i, args.lab_name, optimizer, past_losses, False, True,
                           start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time, LOGS_TOTAL=LOGS_TOTAL, SCORES_TOTAL=SCORES_TOTAL)
                logger.info(f"Saved final checkpoint!")
                exit(0)

            seq = batch[0]
            methy = batch[1]
            cell_embedding = batch[2]
            cell_region_mask = batch[3]
            # sample_idx = batch[3]

            seq_tensor = seq.to(device=DEVICE, dtype=torch.float32).transpose(1, 2).contiguous()  # [bsz, 4, seq_len]
            methy_tensor = methy.to(device=DEVICE, dtype=torch.float32)  # shape [bsz, n_track, seq_len]
            bin_methy = pre_process_func(methy_tensor)
            model.train()
            cpg_mask, valid_mask, low_methy_mask, combined_mask = make_cpg_and_valid_and_low_methy_mask_multi_track(seq_tensor,
                                                                                                                    methy_tensor)

            cell_region_mask = mask_cell_region(cell_region_mask).cuda()
            valid_cpg_mask = cpg_mask & valid_mask
            batch_idx_valid, track_idx_valid, pos_idx_valid = torch.where(valid_mask)
            if not valid_mask.any():
                msg = f"No valid data in batch {i}, skipping this batch."
                logger.error(msg)
                continue
            alpha = get_alpha_tensor(methy_tensor, args, valid_cpg_mask, combined_mask, cell_region_mask, batch_idx_valid, track_idx_valid, pos_idx_valid)
            model.train()
            pred_raw = model(seq_tensor,
                             cell_embedding=cell_embedding.cuda(),)
            pred, loss_cg, loss_avg = get_real_pred_and_cg_avg_loss_from_raw_model_output(pred_raw, seq_tensor, methy_tensor, args)
            loss_tracks, loss_di = basic_blocks.collect_cell_losses(track_names, batch_idx_valid, track_idx_valid, pos_idx_valid, model, pred, bin_methy, i, alpha, flag_use_focal)
            loss_total = loss_tracks + loss_cg + loss_avg
            loss_total.backward()
            optimizer.step()
            step_logs = {
                "Loss/Step": loss_tracks,
                "Loss_CG/Step": loss_cg,  # Log scaled loss
                "Loss_Avg/Step": loss_avg,  # Log scaled loss
                "Loss_Total/Step": loss_total,
                "Loss_Tracks/Step": loss_tracks,
            }
            wdb.log(step_logs, i)
            past_losses.append(loss_total.detach().cpu().numpy())


        train_loss = np.mean(past_losses[-40:])
        logger.info(f" LR: {optimizer.param_groups[0]['lr']:.6f}, Step {i}: train loss: {train_loss:.6f}")

        check_loss_fn = nn.BCELoss(reduction='none')


        results = check_runs_return_multiple_pic_dict_batched_calc_total(
            model=model,
            genome=genome,
            methy_data_dict=methy_data,
            cell_embedding_dict=cell_embeddings,
            see_length=args.window_size,
            track_names=test_track,
            device=DEVICE,
            batch_size=64,
            loss_fn=check_loss_fn,
            methy_data_process=pre_process_func if not (
                    args.meth_pre_process_func is None or args.meth_pre_process_func == 'None') else methy2bin,
            calc_corr_using_original_myth=True,
            model_output_post_process=func_,
            tiny=False
        )
        real_eval_logs = {}
        all_tracks_logs_accumulator = {}
        for track_name in track_names:
            eval_logs = {}
            for mode, mode_results in results.items():
                update_eval_logs(track_name, mode_results, eval_logs, mode)
            for key, value in eval_logs.items():
                generic_key = key.split('/', 1)[1]
                all_tracks_logs_accumulator.setdefault(generic_key, []).append(value)
            if eval_logs:
                wdb.log(eval_logs, i)
        for generic_key, values_list in all_tracks_logs_accumulator.items():
            average_value = np.mean(values_list)
            real_eval_logs[generic_key] = average_value
        di_scores = get_score_metrics(real_eval_logs)

        SCORES_TOTAL[i] = di_scores
        log_best_valid_step = get_best_step_and_metrics(SCORES_TOTAL)
        flag = (i > -1 or args.auto_resume)
        if flag and (log_best_valid_step['best_step/best_step'] == i):
            save_model(model, "best", args.lab_name, None, past_losses, use_jit=False, save_optimizer=False,
                       start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time,
                       # LOGS_TOTAL=LOGS_TOTAL, SCORES_TOTAL=SCORES_TOTAL
                       )
        wdb.log(real_eval_logs, i)
        wdb.log(log_best_valid_step, i)
        wdb.log(di_scores, i)
        if (current_time >= next_checkpoint_time) or (i % 10 == 0 and i != 0):
            save_model(model, i, args.lab_name, optimizer, past_losses, use_jit=False, save_optimizer=True,
                       start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time,
                       # LOGS_TOTAL=LOGS_TOTAL, SCORES_TOTAL=SCORES_TOTAL
                       )
            next_checkpoint_time += timedelta(minutes=save_minutes)

if __name__ == '__main__':
    train()

