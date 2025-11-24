# from run_bce.py
import argparse
import json
import os
import sys
from collections import defaultdict

import random


import math
import pandas as pd
import torch

import numpy as np

from datetime import datetime
from pathlib import Path
import seaborn as sns

import resource

from matplotlib import pyplot as plt
from tqdm import tqdm

from _config import pdir
from global_constants import track_39_bigwig_file_names

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))




start_time_str = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--from_snapshot', type=str, help='Path to the checkpoint directory to resume training from')
parser.add_argument('--auto-resume', action='store_true', help='Automatically resume latest checkpoint. If no checkpoint is available, do not set it.')
parser.add_argument('--project', type=str, default='Melody', help='Base directory for data and checkpoints')
parser.add_argument('--lab_name', type=str, default=f'multi_cell_{start_time_str}', help='Name of the lab for logging')
parser.add_argument('--from_ckpt', type=str,
                    default=pdir + "/data/checkpoints/Melody-G1.pth",
                    help='Path to the checkpoint file to load')
parser.add_argument('--appointed_csv_dataset', type=str, default=None, help='Path to the CSV file for the dataset')
parser.add_argument('--use_cg_loss', action='store_true', help='Use CG loss in training')
parser.add_argument('--use_avg_loss', action='store_true', help='Use average loss in training')
parser.add_argument('--wcg', type=float, default=0.001, help='Weight for CG loss')
parser.add_argument('--wavg', type=float, default=1, help='Weight for average loss')
parser.add_argument('--use_mt_loss', action='store_true', help='Use MT loss in training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default: 64')
parser.add_argument('--window_size', type=int, default=10000, help='Window size, default: 10000')
parser.add_argument('--seed', type=int, default=5, help='Random seed, default: 4')
parser.add_argument('--genome_path', type=str,
                    default=pdir + '/data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
                    help='Genome .fa file path')
parser.add_argument('--bigwigs_dir', type=str, default=pdir + '/data/bigwigs/', help='BigWigs directory')
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
parser.add_argument('--train_hour', type=int, default=72, help='Max train hour, default: 72')
# exclude non cpg regions
parser.add_argument('--exclude_non_cpg', action='store_true', help='Exclude non CpG regions, default: False')
# use euphonium
parser.add_argument("--use_euphonium", action='store_true', help='Use euphonium 2 stage model, default: False')
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
    load_ckpt, print_model_info, get_pre_process_func, get_embedding \


from selene_sdk.sequences import Genome, GenomicDataset
from cell_embedding import MelodyG
from basic_blocks import make_cpg_and_valid_and_low_methy_mask_multi_track, get_alpha_tensor, \
    get_real_pred_and_cg_avg_loss_from_raw_model_output
from check_utils import clean_seq_ndarray, get_aucs_and_accs_aggregated, get_corrs_aggregated


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
pre_process_func = get_pre_process_func(args.meth_pre_process_func)


genome = Genome(input_path=args.genome_path, cuda=True if DEVICE == 'cuda' else False)
methy_data = [GenomicDataset([m], genome, storage="BigWig") for m in bigwig_files]

cell_embeddings = get_embedding(bigwig_files, args.cell_embedding_path)



class RandomMethyWithEmbeddingDataset:

    def __init__(self, methy_dataset_list, embedding_list):
        assert len(methy_dataset_list) == len(embedding_list)
        self.methy_dataset_list = methy_dataset_list
        self.embedding_list = embedding_list

    def get(self, chrom, start, end):
        idx = random.randint(0, len(self.methy_dataset_list) - 1)
        methy_data = self.methy_dataset_list[idx]
        embedding = self.embedding_list[idx]
        methy = methy_data.get(chrom, start, end)

        return methy, np.mean(embedding, axis=0), np.array([[idx]])


    def uninitialize(self):
        return None

def extract_tissue_name_from_path(file_path):
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    first_underscore = name.find('_')
    last_dash = name.rfind('-')

    if first_underscore != -1 and last_dash != -1 and first_underscore < last_dash:
        return name[first_underscore + 1: last_dash]
    else:
        return None

def plot_pred_vs_methy(ax, pred_test, methy_data_test, chr, start, end, aucs_and_accs, corrs, track_name, batch_idx=0):

    pred = pred_test
    methy = -methy_data_test

    ax.plot(pred, color='#E41A1C', label='Pred')
    ax.plot(methy, color='#1F77B4', label='Methy')
    ax.set_title(f"{track_name} methy: {aucs_and_accs['methy_CpG']:.4f}, pred: {aucs_and_accs['pred_CpG']:.4f}\n"
                 f"acc_C/G: {aucs_and_accs['acc_C/G']:.4f}, "
                 f"acc_CpG: {aucs_and_accs['acc_CpG']:.4f}, acc_all: {aucs_and_accs['acc_all']:.4f}, "
                 f"auc_C/G: {aucs_and_accs['auc_C/G']:.4f}, auc_CpG: {aucs_and_accs['auc_CpG']:.4f}, "
                 f"auc_all: {aucs_and_accs['auc_all']:.4f} \n"
                 f"corr_C/G: {corrs['C/G']:.4f}, corr_CpG: {corrs['CpG']:.4f}, corr_All: {corrs['All']:.4f}, ", fontsize=12, fontweight="bold")
    ax.set_xlabel("Position in Sequence")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, len(pred) - 1])
    ax.set_ylim([-1, 1])

def find_matching_indices(query_list, target_list):
    indices = []
    for query in query_list:
        for i, val in enumerate(target_list):
            if val == query:
                indices.append(i)
    return indices

def filter_by_indices(lst, index_list):
    return [lst[i] for i in index_list]

def sigmoid_first(el):
    if isinstance(el, (tuple, list)):
        el = el[0]
    if isinstance(el , torch.Tensor):
        return torch.sigmoid(el)
    elif isinstance(el, np.ndarray):
        return 1.0 / (1.0 + np.exp(-el))
    else:
        raise TypeError(f"Unsupported type for sigmoid_first: {type(el)}")

def _plot_split_subplots(data_dict, individual_keys, figure_title, filename):

    all_keys = set(data_dict.keys())

    keys_specified = sorted([key for key in individual_keys if key in all_keys])
    keys_other = sorted(list(all_keys - set(keys_specified)))

    data_specified = {k: data_dict[k] for k in keys_specified}
    data_other = {k: data_dict[k] for k in keys_other}

    if not data_specified or not data_other:
        print(
            f"Skipping subplot generation for '{figure_title}' because one group is empty. Drawing a single heatmap instead.")
        df_full = pd.DataFrame.from_dict(data_dict, orient='index')
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_full, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(figure_title)
        plt.xlabel("Metrics")
        plt.ylabel("Type")
        # plt.savefig(filename, dpi=300)
        plt.show()
        return

    df_specified = pd.DataFrame.from_dict(data_specified, orient='index')
    df_other = pd.DataFrame.from_dict(data_other, orient='index')

    ratio1 = max(1, len(df_specified.index))
    ratio2 = max(1, len(df_other.index))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [ratio1, ratio2]}
    )

    sns.heatmap(df_specified, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Unseen")
    axes[0].set_ylabel("Type")

    tick_positions_spec = np.arange(len(df_specified.index)) + 0.5
    axes[0].set_yticks(tick_positions_spec)

    axes[0].set_yticklabels(df_specified.index, rotation=0, va='center')


    sns.heatmap(df_other, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Other Keys")
    axes[1].set_ylabel("Type")
    axes[1].set_xlabel("Metrics")

    tick_positions_other = np.arange(len(df_other.index)) + 0.5
    axes[1].set_yticks(tick_positions_other)
    axes[1].set_yticklabels(df_other.index, rotation=0, va='center')

    fig.suptitle(figure_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(filename, dpi=300)
    plt.show()


def draw_heatmap(aucs_accs_avg, corrs_avg, individual_keys):

    print("--- Plotting AUCs and ACCs ---")
    _plot_split_subplots(
        data_dict=aucs_accs_avg,
        individual_keys=individual_keys,
        figure_title="scgpt_stage_1_415208 Split AUCs and ACCs",
        filename="scgpt_stage_1_415208_AUCs_and_ACCs_subplots.png"
    )

    print("\n--- Plotting Correlations ---")
    _plot_split_subplots(
        data_dict=corrs_avg,
        individual_keys=individual_keys,
        figure_title="scgpt_stage_1_415208 Split Correlations",
        filename="scgpt_stage_1_415208_corrs_subplots.png"
    )


def convert(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

if __name__ == '__main__':
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    model = MelodyG(args, n_track=1)

    model.eval()
    model = model.to(DEVICE)
    config_model_loss_fn(model, args=args)
    if args.from_ckpt: load_ckpt(model, args.from_ckpt)
    if args.compile:
        model.compile()
        basic_blocks.calc_cg_and_avg_loss_variable_len = torch.compile(
            basic_blocks.calc_cg_and_avg_loss_variable_len)
        make_cpg_and_valid_and_low_methy_mask_multi_track = torch.compile(
            make_cpg_and_valid_and_low_methy_mask_multi_track)

    print_model_info(model)

    with open(pdir + "/cell_zones/test_data_dict.json", "r", encoding="utf-8") as f:
        type_dict = json.load(f)

    types = list(type_dict.keys())

    methy_data_files = [os.path.join(args.bigwigs_dir, f) for f in os.listdir(args.bigwigs_dir) if
                        f.endswith('.hg38.bigwig')]

    tissue_list = [extract_tissue_name_from_path(f) for f in methy_data_files]

    find_index = find_matching_indices(types, tissue_list)

    aucs_accs_sum = defaultdict(lambda: defaultdict(float))
    aucs_accs_count = defaultdict(lambda: defaultdict(int))
    corrs_sum = defaultdict(lambda: defaultdict(float))
    corrs_count = defaultdict(lambda: defaultdict(int))

    aucs_and_accs_dict = {}
    corrs_dict = {}

    for type_id, type in enumerate(tqdm(types)):
        if type in ['Pancreas-Delta', 'Blood-Granulocytes', 'Blood-Monocytes', 'Aorta-Endothel', 'Cortex-Neuron']:
            track_idx = find_index[type_id]
            region_list = type_dict[type]
            bigwig_files = [methy_data_files[track_idx]]
            methy_data = GenomicDataset(bigwig_files, genome=genome, storage='BigWig')

            preds = []
            labels = []
            seqs = []

            for region in region_list:
                chr, start, end = region
                length = end - start

                center_point = (start + end) // 2
                input_start = center_point - args.window_size // 2
                input_end = center_point + args.window_size // 2

                methy = methy_data.get(chr, input_start, input_end)

                seq = genome.get(chr, input_start, input_end)

                seq, methy = torch.tensor(seq).unsqueeze(0), torch.tensor(methy).unsqueeze(0)

                seq_tensor = seq.to(device=DEVICE, dtype=torch.float32).transpose(1,
                                                                                  2).contiguous()  # [bsz, 4, seq_len]
                methy_tensor = methy.to(device=DEVICE, dtype=torch.float32)  # shape [bsz, n_track, seq_len]

                bin_methy = pre_process_func(methy_tensor)
                cpg_mask, valid_mask, low_methy_mask, combined_mask = make_cpg_and_valid_and_low_methy_mask_multi_track(
                    seq_tensor,
                    methy_tensor)
                valid_cpg_mask = cpg_mask & valid_mask
                batch_idx_valid, track_idx_valid, pos_idx_valid = torch.where(valid_mask)
                alpha = get_alpha_tensor(methy_tensor, args, valid_cpg_mask, combined_mask, batch_idx_valid,
                                         track_idx_valid, pos_idx_valid)
                cell_embedding = torch.tensor(cell_embeddings[type]).unsqueeze(0)
                cell_embedding = torch.mean(cell_embedding, dim=1)
                pred_raw = model(seq_tensor, cell_embedding=cell_embedding.cuda())
                pred, loss_cg, loss_avg = get_real_pred_and_cg_avg_loss_from_raw_model_output(pred_raw, seq_tensor,
                                                                                              methy_tensor, args)
                pred = sigmoid_first(pred)

                loss_tracks, loss_di = basic_blocks.collect_cell_losses(track_names, batch_idx_valid, track_idx_valid,
                                                                        pos_idx_valid, model, pred, bin_methy, 0, alpha,
                                                                        flag_use_focal)

                target_pred = pred[:, :, args.window_size // 2 - length // 2: args.window_size // 2 + length // 2]
                target_methy = methy[:, :, args.window_size // 2 - length // 2: args.window_size // 2 + length // 2]
                target_seq = seq[:, args.window_size // 2 - length // 2: args.window_size // 2 + length // 2, :]

                target_pred = clean_seq_ndarray(target_pred)
                target_methy = clean_seq_ndarray(target_methy)
                target_seq_np = target_seq.squeeze(0).cpu().numpy()

                preds.append(target_pred)
                labels.append(target_methy)
                seqs.append(target_seq_np)

            aucs_and_accs = get_aucs_and_accs_aggregated(np.concatenate(preds), np.concatenate(labels),
                                                         np.concatenate(seqs))
            corrs = get_corrs_aggregated(np.concatenate(preds), np.concatenate(labels), np.concatenate(seqs))

            aucs_and_accs_dict[type] = aucs_and_accs
            corrs_dict[type] = corrs


    test_track = ['Pancreas-Delta', 'Blood-Granulocytes', 'Blood-Monocytes', 'Aorta-Endothel', 'Cortex-Neuron']
    draw_heatmap(aucs_and_accs_dict, corrs_dict, test_track)




