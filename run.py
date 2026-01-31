import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)

import sys
from typing import Callable
import numpy as np
import time

from loguru import logger
from datetime import datetime, timedelta
from pathlib import Path
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from _config import pdir

start_time_str = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

parser.add_argument('--auto-resume', action='store_true', help='Automatically resume latest checkpoint. If no checkpoint is available, do not set it.')
parser.add_argument('--project', type=str, default='Melody', help='Base directory for data and checkpoints')
parser.add_argument('--lab_name', type=str, default=f'GridSearch_{start_time_str}', help='Name of the lab for logging')
parser.add_argument('--from_ckpt', type=str, default=None, help='Path to the checkpoint file to load')
parser.add_argument('--use_cg_loss', action='store_true', help='Use CG loss in training')
parser.add_argument('--use_avg_loss', action='store_true', help='Use average loss in training')
parser.add_argument('--wcg', type=float, default=0.01, help='Weight for CG loss')
parser.add_argument('--wavg', type=float, default=1, help='Weight for average loss')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default: 64')
parser.add_argument('--window_size', type=int, default=10000, help='Window size, default: 10000') # model input length
parser.add_argument('--split_length', type=int, default=10000, help='split_length size, default: 10000') # check region length. Must: model_length >= check_size or check_size % model_length == 0
parser.add_argument('--seed', type=int, default=5, help='Random seed, default: 4')
parser.add_argument('--genome_path', type=str,
                    default=pdir + '/data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
                    help='Genome .fa file path')
parser.add_argument('--bigwigs_dir', type=str, default='/data/wangding/nature_meth/', help='BigWigs directory')
parser.add_argument('--bigwigs_files', type=str, default=['GSM5652317_Blood-B-Z000000UB.hg38.bigwig'], nargs='+', help='BigWigs file')
parser.add_argument('--cpg_cls_bin_width', type=int, default=5,
                    help='Bin width for CpG count classification, default: 5')
parser.add_argument('--cpg_cls_n', type=int, default=7, help='Max number of classes for CpG counts, default: 7')
parser.add_argument('--use_advanced_focal_loss', action='store_true', help='Use Advanced focal loss in training')
parser.add_argument('--use_fixed_focal_cpg_loss',  action='store_true', help='Use Fixed focal loss in training')
parser.add_argument('--low_methy_cpg_focal_weight', type=float, default=32.0, help='Focal loss low_methy_cpg_focal_weight weight, default: 32.0')
parser.add_argument('--any_cpg_focal_weight', type=float, default=8.0, help='Focal loss any_cpg_focal_weight weight, default: 8.0')
parser.add_argument('--side_head_structure', type=str, default='max_pool_no_softplus', help='Side head structure')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate, default: 0.001')
parser.add_argument('--meth_pre_process_func', type=str, default='None',
                    help='Method to preprocess, default: None')
parser.add_argument('--train_hour', type=int, default=99, help='Max train hour, default: 99')
parser.add_argument('--use_swanlab', action='store_true', help='Use SWAN Lab, default: False. If true, will use swanlab for logging instead of wandb.')
# one_stage_ckpt
parser.add_argument('--one_stage_ckpt', type=str, default=None,
                    help='Path to the one stage model checkpoint, default: None. If provided, will use this checkpoint for the 2 stage of training.')
parser.add_argument('--use_39_track', action='store_true', help='Use 39 track, default: False. If true, will use 39 track for training.')
parser.add_argument('--model_cls', type=str, default='Melody',)
parser.add_argument('--no-compile', action='store_false', dest='compile', help='Do not compile the model, default: Compile. If --no-compile, will not compile the model.')
parser.add_argument('--test' , action='store_true', help='Test mode, default: False. If true, will run the test mode.')
# stop_at_step default 50_0000
parser.add_argument('--stop_at_step', type=int, default=50_0000, help='Stop at step, default: 50_0000. If set, will stop training at this step.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
args.bigwigs_dir = '/data/wangding/meth_merged_final'
args.bigwigs_files = ['Adipocytes.merged.hg38.bigwig', 'Endothelium_Pancreas_Islet.merged.hg38.bigwig', 'Macrophages_Liver.merged.hg38.bigwig', 'Blood_B_General.merged.hg38.bigwig', 'Endothelium_Pancreas.merged.hg38.bigwig', 'Macrophages_Lung_Alveolar.merged.hg38.bigwig', 'Blood_B_Memory.merged.hg38.bigwig', 'Endothelium_Saphenous_Vein.merged.hg38.bigwig', 'Macrophages_Lung_Interstitial.merged.hg38.bigwig', 'Blood_Granulocytes.merged.hg38.bigwig', 'Epithelial_Endometrium.merged.hg38.bigwig', 'Muscle_Cardiomyocytes.merged.hg38.bigwig', 'Blood_Monocytes.merged.hg38.bigwig', 'Epithelial_Esophagus.merged.hg38.bigwig', 'Muscle_Skeletal.merged.hg38.bigwig', 'Blood_NK.merged.hg38.bigwig', 'Epithelial_Fallopian.merged.hg38.bigwig', 'Muscle_Smooth_Aorta.merged.hg38.bigwig', 'Blood_T_CD3_General.merged.hg38.bigwig', 'Epithelial_Larynx.merged.hg38.bigwig', 'Muscle_Smooth_Bladder.merged.hg38.bigwig', 'Blood_T_CD4_Central_Memory.merged.hg38.bigwig', 'Epithelial_Ovary.merged.hg38.bigwig', 'Muscle_Smooth_Bronchus.merged.hg38.bigwig', 'Blood_T_CD4_Effector_Memory.merged.hg38.bigwig', 'Epithelial_Pharynx.merged.hg38.bigwig', 'Muscle_Smooth_Coronary.merged.hg38.bigwig', 'Blood_T_CD4_General.merged.hg38.bigwig', 'Epithelial_Skin_Keratinocytes.merged.hg38.bigwig', 'Muscle_Smooth_Prostate.merged.hg38.bigwig', 'Blood_T_CD4_Naive.merged.hg38.bigwig', 'Epithelial_Tongue_Base.merged.hg38.bigwig', 'Neuron_Cerebellum.merged.hg38.bigwig', 'Blood_T_CD8_Effector_Memory.merged.hg38.bigwig', 'Epithelial_Tongue.merged.hg38.bigwig', 'Neuron_Cortex.merged.hg38.bigwig', 'Blood_T_CD8_Effector.merged.hg38.bigwig', 'Epithelial_Tonsil_Palatine.merged.hg38.bigwig', 'Neuron_General.merged.hg38.bigwig', 'Blood_T_CD8_General.merged.hg38.bigwig', 'Epithelial_Tonsil_Pharyngeal.merged.hg38.bigwig', 'Oligodendrocytes.merged.hg38.bigwig', 'Blood_T_CD8_Naive.merged.hg38.bigwig', 'Fibroblasts_Colon.merged.hg38.bigwig', 'Pancreas_Acinar.merged.hg38.bigwig', 'BoneMarrow_Erythrocyte_Progenitor.merged.hg38.bigwig', 'Fibroblasts_Dermal.merged.hg38.bigwig', 'Pancreas_Alpha.merged.hg38.bigwig', 'Bone_Osteoblasts.merged.hg38.bigwig', 'Fibroblasts_Heart.merged.hg38.bigwig', 'Pancreas_Beta.merged.hg38.bigwig', 'Endothelium_Aorta.merged.hg38.bigwig', 'Kidney_Epithelial_Glomerular.merged.hg38.bigwig', 'Pancreas_Delta.merged.hg38.bigwig', 'Endothelium_Kidney_Glomerular.merged.hg38.bigwig', 'Kidney_Epithelial_Tubular.merged.hg38.bigwig', 'Pancreas_Duct.merged.hg38.bigwig', 'Endothelium_Kidney_Tubular.merged.hg38.bigwig', 'Kidney_Podocytes.merged.hg38.bigwig', 'Thyroid_Epithelial.merged.hg38.bigwig', 'Endothelium_Liver.merged.hg38.bigwig', 'Liver_Hepatocytes.merged.hg38.bigwig', 'Endothelium_Lung_Alveolar.merged.hg38.bigwig', 'Macrophages_Colon.merged.hg38.bigwig']

import torch
from stateless import sigmoid_first, get_bigwig_filepaths, check_args, config_model_loss_fn, load_ckpt, print_model_info, save_model, get_pre_process_func
args = check_args(args)

from models import Melody
from selene_sdk.sequences import Genome
from selene_util import GenomicSignalFeatures, RandomPositions, RandomPositionsSampler, SamplerDataLoader

from basic_blocks import make_cpg_and_valid_and_low_methy_mask_multi_track, get_alpha_tensor, \
    get_real_pred_and_cg_avg_loss_from_raw_model_output, collect_track_losses
from wandb_worker import WandbWorker
import basic_blocks

flag_use_focal = args.use_fixed_focal_cpg_loss or args.use_advanced_focal_loss
func_ = sigmoid_first

BATCH_SIZE, WINDOW_SIZE, SEED, DEVICE = args.batch_size, args.window_size, args.seed, 'cuda'
bigwig_files, track_names = get_bigwig_filepaths(dir_=args.bigwigs_dir, filenames=args.bigwigs_files)
pre_process_func: Callable = get_pre_process_func(args.meth_pre_process_func)
genome = Genome(input_path=args.genome_path, blacklist_regions='hg38')
genome.get = genome.get_encoding_from_coords
methy_data = GenomicSignalFeatures(bigwig_files, features=bigwig_files, shape=(10000,))


if args.model_cls == 'Melody':
    model = Melody(args, n_track=len(bigwig_files))
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


def train():
    train_random_pos = RandomPositions(genome, sample_length=WINDOW_SIZE, mode='train')
    train_sampler = RandomPositionsSampler([genome, methy_data], train_random_pos)
    train_dataloader = SamplerDataLoader(train_sampler, num_workers=4, batch_size=BATCH_SIZE, seed=SEED)
    wdb = WandbWorker(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    i, error_num, err_max = 0, 0, 300
    past_losses = []
    start_time = datetime.now()
    train_hour, save_minutes = args.train_hour, 180
    next_checkpoint_time = start_time + timedelta(minutes=180)  # Checkpoints every x minutes
    end_time = start_time + timedelta(hours=train_hour)  # Total run time is x hours
    for seq, methy in train_dataloader:
        i += 1
        optimizer.zero_grad()  # Reset gradients for next accumulation
        current_time = datetime.now()
        # Check if total running time has exceeded 8 hours
        if current_time >= end_time:
            logger.info(f"Exiting due to time limit.")
            save_model(model, i, args.lab_name, optimizer, past_losses, False, False,
                       start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time)
            logger.info(f"Saved final checkpoint!")
            exit(0)
        # stop at 100W steps
        if i >= 100_0000:
            logger.info(f"Exiting due to reaching 100W steps limit.")
            save_model(model, i, args.lab_name, optimizer, past_losses, False, False,
                       start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time)
            logger.info(f"Saved final checkpoint!")
            exit(0)
        seq_tensor = seq.to(device=DEVICE, dtype=torch.float32).transpose(1, 2).contiguous()  # [bsz, 4, seq_len]
        methy_tensor = methy.to(device=DEVICE, dtype=torch.float32)  # shape [bsz, n_track, seq_len]
        bin_methy = pre_process_func(methy_tensor)
        model.train()
        cpg_mask, valid_mask, low_methy_mask, combined_mask = make_cpg_and_valid_and_low_methy_mask_multi_track(seq_tensor,
                                                                                                                methy_tensor)
        valid_cpg_mask = cpg_mask & valid_mask
        batch_idx_valid, track_idx_valid, pos_idx_valid = torch.where(valid_mask)
        if not valid_mask.any():
            msg = f"No valid data in batch {i}, skipping this batch."
            logger.error(msg)
            continue
        alpha = get_alpha_tensor(methy_tensor, args, valid_cpg_mask, combined_mask, batch_idx_valid, track_idx_valid, pos_idx_valid)
        model.train()
        pred_raw = model(seq_tensor)
        pred, loss_cg, loss_avg = get_real_pred_and_cg_avg_loss_from_raw_model_output(pred_raw, seq_tensor, methy_tensor, args)
        loss_tracks, loss_di = collect_track_losses(track_names, batch_idx_valid, track_idx_valid, pos_idx_valid, model, pred, bin_methy, i, alpha, flag_use_focal)
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
        if i == 20 or i % 2000 == 0:
            train_loss = np.mean(past_losses[-40:])
            logger.info(f" LR: {optimizer.param_groups[0]['lr']:.6f}, Step {i}: train loss: {train_loss:.6f}")
        # save at 50000 steps
        if i % 50000 == 0 and i >= 100:
            save_model(model, i, args.lab_name, optimizer, past_losses, use_jit=False, save_optimizer=True,
                       start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time
                       )
            logger.info(f"Saved model at step {i}.")

        if current_time >= next_checkpoint_time:
            save_model(model, i, args.lab_name, optimizer, past_losses, use_jit=False, save_optimizer=False,
                       start_time=start_time, end_time=end_time, next_checkpoint_time=next_checkpoint_time
                       )
            next_checkpoint_time += timedelta(minutes=save_minutes)

        if len(past_losses) > 100:
            past_losses = past_losses[-100:]



if __name__ == '__main__':
    train()
