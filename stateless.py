import builtins
import time
from argparse import Namespace
from datetime import datetime, timedelta
import os
from functools import cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from echo_logger import dumps_json
import loguru
import torch.nn as nn
import gc

from matplotlib import pyplot as plt
from torch import Tensor

from global_constants import track_39_bigwig_file_names


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def methy2bin(methy_data):
    # input: # shape [bsz, 1, seq_len], float32 tensor or ndarray
    # output: shape [bsz, 1, seq_len], float32 tensor or ndarray
    # mapping:
    # [0.0, 0.5) -> 0.0
    # [0.5, 1.0] -> 1.0

    if isinstance(methy_data, torch.Tensor):
        binned_data = torch.zeros_like(methy_data, dtype=torch.float32)
        device = methy_data.device  # Get device to ensure constants are on the same device

        binned_data = torch.where(methy_data < 0.5, torch.tensor(0.0, dtype=torch.float32, device=device), binned_data)
        binned_data = torch.where(methy_data >= 0.5, torch.tensor(1.0, dtype=torch.float32, device=device), binned_data)

    elif isinstance(methy_data, np.ndarray):
        binned_data = np.zeros_like(methy_data, dtype=np.float32)

        binned_data[methy_data < 0.5] = 0.0
        binned_data[methy_data >= 0.5] = 1.0
    else:
        raise TypeError("Input methy_data must be either torch.Tensor or numpy.ndarray")

    return binned_data


def model_is_dp(param_keys):
    return all([key.startswith('module') for key in param_keys])


def load_ckpt(
        model: torch.nn.Module,
        load_pth,
        mute=False
):
    loguru.logger.warning(f'load model from {load_pth}')
    checkpoint = torch.load(load_pth, weights_only=False)['model_state_dict']
    new_model_dict = model.state_dict()
    new_model_keys = set(list(new_model_dict.keys()))

    pretrained_dict = {'.'.join(k.split('.')): v for k, v in checkpoint.items()}
    pretrained_keys = set(list('.'.join(k.split('.')) for k in checkpoint.keys()))
    is_dp = model_is_dp(pretrained_keys)
    if is_dp:
        # rmv 'module.' prefix
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        pretrained_keys = set(list(k[7:] for k in pretrained_keys))
        # only update the same keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_keys}
    diff_, same_ = new_model_keys - pretrained_keys, new_model_keys & pretrained_keys
    not_used_keys = pretrained_keys - new_model_keys
    if not mute:
        loguru.logger.info("Same Keys (Loaded Keys), Though maybe not updated by optimizer: ")
        loguru.logger.info(dumps_json(sorted(list(same_))))
        loguru.logger.info("Not Loaded Keys: (Keys in model but not in pretrained model)")
        loguru.logger.error(dumps_json(sorted(list(diff_))))
        loguru.logger.info("Not Used Keys (Keys in pretrained model but not in model): ")
        loguru.logger.error(dumps_json(sorted(list(not_used_keys))))
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)

@loguru.logger.catch
def save_model(model, step, save_dir, optimizer, past_loss_li, use_jit=False, save_optimizer=True, **kwargs):
    current_time = datetime.now().isoformat()
    if use_jit:
        scripted_model = torch.jit.script(model)
        jit_save_abs_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_step_{step}_model.jit.pt"))
        scripted_model.save(jit_save_abs_path)
        checkpoint = {
            'step': step,
            'model.jit.pt.path': jit_save_abs_path,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'loss_history': past_loss_li,
            'current_time': current_time,
            **kwargs
        }
    else:
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'loss_history': past_loss_li,
            'current_time': current_time,
            **kwargs
        }
    if save_optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_step_{step}.pth"))
    loguru.logger.info(f"Saved checkpoint: {os.path.join(save_dir, f'checkpoint_step_{step}.pth')}")

def make_cpg_and_valid_and_low_methy_mask(seq_tensor, methy_tensor):
    """
    seq_tensor: (bsz, 4, seq_len) (one-hot). [1, 0, 0, 0] (representing 'A'), [0, 1, 0, 0] ('C'),[0, 0, 1, 0] ('G'), and [0, 0, 0, 1] ('T').
    methy_tensor: (bsz, 1, seq_len)
    """
    # Create CpG flag (C followed by G)
    bsz, _, seq_len = seq_tensor.shape
    c_sites = seq_tensor[:, 1:2, :]  # Get C positions
    g_sites = seq_tensor[:, 2:3, :]  # Get G positions

    # Initialize CpG flag with zeros
    cpg_flag = torch.zeros((bsz, 1, seq_len), dtype=torch.float32, device=seq_tensor.device)  # shape [bsz, 1, seq_len]

    # Mark C positions in CpG
    cpg_flag[:, :, :-1] = (c_sites[:, :, :-1] * g_sites[:, :, 1:]) > 0

    # Mark G positions in CpG (shift the mask to mark G positions that follow C)
    g_in_cpg = torch.zeros_like(cpg_flag)
    g_in_cpg[:, :, 1:] = cpg_flag[:, :, :-1]

    # Combine both C and G positions in CpG
    combined_cpg = torch.max(cpg_flag, g_in_cpg)

    # Only set methylation values for CpG sites (where label_tensor[:, 0, :] is 0)
    # For non-CpG sites (where label_tensor[:, 0, :] is 1), keep methylation values as 0
    cpg_mask = combined_cpg.to(dtype=torch.bool)  # 1 for CpG sites (both C and G), 0 for non-CpG sites
    valid_mask = (methy_tensor[:, 0, :] >= 0)  # shape: [bsz, seq_len]
    low_methy_mask = (methy_tensor[:, 0, :] < 0.5)  # shape: [bsz, seq_len]
    # Combine masks
    combined_mask = cpg_mask[:, 0, :] * valid_mask * low_methy_mask  # shape: [bsz, seq_len]

    return cpg_mask[:, 0, :], valid_mask, low_methy_mask, combined_mask



def print_model_info(model):
    # Get the number of parameters in the model
    param_count = sum(p.numel() for p in model.parameters())

    # Calculate the disk space cost (in MB)
    disk_space = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"Model has {param_count:,} parameters and occupies {disk_space:.2f} MB on disk.")


def parse_cg_class_n_cuda(cg_counts, cpg_cls_bin_width: int, cpg_cls_n: int):
    """
    Optimized version to classify CpG counts into n classes using vectorized operations.

    Class 0 is always assigned to segments with 0 CpG counts.
    For segments with counts > 0:
    - Class 1: 1 to cpg_cls_bin_width counts
    - Class 2: cpg_cls_bin_width + 1 to 2 * cpg_cls_bin_width counts
    - ...
    - Class (cpg_cls_n - 2): (cpg_cls_n - 3) * cpg_cls_bin_width + 1 to (cpg_cls_n - 2) * cpg_cls_bin_width counts
    - Class (cpg_cls_n - 1): > (cpg_cls_n - 2) * cpg_cls_bin_width counts

    Parameters
    ----------
    cg_counts : torch.Tensor
        Tensor containing CpG counts for each segment. Expected integer type.
    cpg_cls_bin_width : int
        The width of each bin for positive counts (exclusive of class 0 and the max class).
    cpg_cls_n : int
        The total number of classes (must be >= 2).

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as cg_counts, containing class indices (dtype=torch.long).
    """
    if cpg_cls_bin_width <= 0:
        raise ValueError("cpg_cls_bin_width must be a positive integer.")
    if cpg_cls_n < 2:
        raise ValueError("cpg_cls_n must be at least 2 (for class 0 and at least one positive class).")

    zero_based_bin_index = (cg_counts - 1) // cpg_cls_bin_width
    calculated_class = zero_based_bin_index + 1
    max_class_index = cpg_cls_n - 1
    final_class = torch.clamp(calculated_class, min=0, max=max_class_index)
    return final_class.long()

def my_bce_with_logits_loss(inputs, targets, reduction='mean', alpha=None):
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

def sigmoid_first(el):
    if isinstance(el, (tuple, list)):
        el = el[0]
    if isinstance(el , torch.Tensor):
        return torch.sigmoid(el)
    elif isinstance(el, np.ndarray):
        return 1.0 / (1.0 + np.exp(-el))
    else:
        raise TypeError(f"Unsupported type for sigmoid_first: {builtins.type(el)}")

def mask2value(mask_tensor, rule: Dict[bool, float]):
    value_tensor = torch.zeros_like(mask_tensor, dtype=torch.float32, device=mask_tensor.device)
    for key, value in rule.items():
        value_tensor[mask_tensor == key] = value
    return value_tensor


def get_best_step_and_metrics(scores_total):
    best_i = -1
    for i in scores_total.keys():
        if best_i == -1 or scores_total[i]['AUC & Corr/valid'] > scores_total[best_i]['AUC & Corr/valid']:
            best_i = i
    best_step_dict = {'best_step/best_step': best_i}
    for key, value in scores_total[best_i].items():
        best_step_dict[f"best_step/{key}"] = value
    return best_step_dict

def get_pre_process_func(pre_process_func_name):
    pre_process_func = pre_process_func_name
    if pre_process_func == 'methy2bin':
        pre_process_func = methy2bin
    elif pre_process_func is None or pre_process_func == 'None':
        loguru.logger.warning(f"! You are using λ x -> x as pre_process_func, which is equivalent to no processing,")
        loguru.logger.warning(f"! However, when testing ROC/AUC/ACC, `methy2bin` is still used, since non-01 data ")
        loguru.logger.warning(
            f"! can cause ERROR on sklearn metrics (ValueError: continuous format is not supported) will be raised if do nothing at test time.")
        pre_process_func = lambda x: x
    else:
        loguru.logger.warning(
            f"!!! Unknown pre_process_func: {pre_process_func}, using default lambda x: x (identity function).")
        loguru.logger.warning(f"! You are using λ x -> x as pre_process_func, which is equivalent to no processing,")
        loguru.logger.warning(f"! However, when testing ROC/AUC/ACC, `methy2bin` is still used, since non-01 data ")
        loguru.logger.warning(
            f"! can cause ERROR on sklearn metrics (ValueError: continuous format is not supported) will be raised if do nothing at test time.")
        pre_process_func = lambda x: x
    return pre_process_func

def update_eval_logs(track_name, mode_results, eval_logs, mode):
    if track_name not in mode_results:
        return

    track_result = mode_results[track_name]

    if track_result.get("loss") is not None:
        eval_logs[f"{track_name}/Loss_{mode}/Step"] = track_result["loss"]
    if track_result.get("correlations"):
        for key, value in track_result["correlations"].items():
            eval_logs[f"{track_name}/Corr_{mode}/{key}/Step"] = value

    if track_result.get("aucs_and_accs"):
        for key, value in track_result["aucs_and_accs"].items():
            eval_logs[f"{track_name}/Metric_{mode}/{key}/Step"] = value


def check_loss_nan(loss, track_name, i, error_num, err_max=30):
    if torch.isnan(loss) or torch.isinf(loss) or loss > 1e7:
        msg = f"NaN loss detected for track {track_name} at step {i}. is Nan: {torch.isnan(loss)}, is Inf: {torch.isinf(loss)}. Value: {loss}"
        loguru.logger.error(msg)
        error_num += 1
        if error_num > err_max:
            loguru.logger.error(f"Error number exceeded {err_max}, exiting.")
            raise RuntimeError("Exiting due to too many errors.")


def get_track_names(track_filenames: List[str]) -> List[str]:
    track_names = [item_.replace(".hg38.bigwig", "") for item_ in track_filenames]
    return track_names

def get_track_index(track_name, track_names: List[str]) -> int:
    # firstly make sure both track_name and track_names are str and not end with '.hg38.bigwig'
    if isinstance(track_name, str):
        track_name = track_name.replace(".hg38.bigwig", "")
    if isinstance(track_names, list):
        track_names = [item_.replace(".hg38.bigwig", "") for item_ in track_names]
    if track_name not in track_names:
        raise ValueError(f"Track name {track_name} not found in track names {track_names}.")
    return track_names.index(track_name)

def get_track_output(model_output, track_name, track_names: List[str]):
    if isinstance(model_output, list) or isinstance(model_output, tuple):
        model_output, pred_cg, pred_avg = model_output
    return model_output[:, get_track_index(track_name, track_names), :]

def check_args(args: Namespace):
    if isinstance(args.bigwigs_files, str):
        args.bigwigs_files = [args.bigwigs_files]
    # if not end with .hg38.bigwig, add it
    # args.bigwigs_files = [item_ if item_.endswith('.hg38.bigwig') else item_ + '.hg38.bigwig' for item_ in args.bigwigs_files]
    if args.use_39_track:
        loguru.logger.warning(f"You are using --use_39_track, We will ignore your --bigwigs_files: {args.bigwigs_files} and use the default 39 cell type hg38 tracks.")
        args.bigwigs_files = track_39_bigwig_file_names
    return args


def get_bigwig_filepaths(
        dir_: str,
        filenames: List[str],
        fallback_dirs: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    search_dirs = [dir_] + (fallback_dirs or [])

    for current_dir in search_dirs:
        try:
            search_path_obj = Path(current_dir)
            if not search_path_obj.is_dir():
                loguru.logger.debug(f"Skipping not existed {current_dir}")
                continue

            resolved_paths_candidate = []
            base_names_candidate = []
            suffix_to_remove = '.hg38.bigwig'

            for fname_pattern in filenames:
                exact_path = search_path_obj / fname_pattern
                if exact_path.is_file():
                    resolved_paths_candidate.append(str(exact_path))
                    base_name = exact_path.name.removesuffix(suffix_to_remove)
                    base_names_candidate.append(base_name)
                    continue

                found_files = [p for p in search_path_obj.iterdir()
                               if fname_pattern in p.name and p.is_file()]

                if len(found_files) == 0:
                    raise FileNotFoundError(f"In '{current_dir}' No '{fname_pattern}' file")

                if len(found_files) > 1:
                    found_names = [p.name for p in found_files]
                    raise ValueError(f"In '{current_dir}''{fname_pattern}' have multiple options: {found_names}")

                found_path = found_files[0]
                resolved_paths_candidate.append(str(found_path))
                base_name = found_path.name.removesuffix(suffix_to_remove)
                base_names_candidate.append(base_name)

            loguru.logger.info(f"Found'{current_dir}',  {len(resolved_paths_candidate)} files in total")
            for path, name in zip(resolved_paths_candidate, base_names_candidate):
                loguru.logger.info(f"  - path: {path} -> name: {name}")

            return resolved_paths_candidate, base_names_candidate

        except (FileNotFoundError, ValueError) as e:

            loguru.logger.debug(f"Failed to find in '{current_dir}' : {e}. Try another dir.")
            continue

    raise FileNotFoundError(
        f"No file found in ('{dir_}') and {fallback_dirs} "
    )


def extract_tissue_name_from_path(file_path):
    filename = os.path.basename(file_path)

    name = os.path.splitext(filename)[0]

    first_underscore = name.find('_')
    last_dash = name.rfind('-')

    if first_underscore != -1 and last_dash != -1 and first_underscore < last_dash:
        return name[first_underscore + 1: last_dash]
    else:
        return None

def get_embedding_list(bigwig_files, cell_type_path):
    tissue_list = [extract_tissue_name_from_path(f) for f in bigwig_files]
    cell_embedding_path = [os.path.join(cell_type_path, f'{f}.npy') for f in tissue_list]
    cell_embeddings = []
    for c in cell_embedding_path:
        cell_embeddings.append(np.load(c))
        loguru.logger.info(f"Successfully loaded cell embedding '{c}'")

    return cell_embeddings

def get_embedding(bigwig_files, cell_type_path):
    tissue_list = [extract_tissue_name_from_path(f) for f in bigwig_files]
    cell_embedding_path = [os.path.join(cell_type_path, f'{f}.npy') for f in tissue_list]
    cell_embeddings = {}
    for tissue, path in zip(tissue_list, cell_embedding_path):
        cell_embeddings[tissue] = np.load(path)
        loguru.logger.info(f"Successfully loaded cell embedding '{path}'")

    return cell_embeddings


def focal_loss(inputs, targets, gamma=0, alpha=None):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    pt = torch.exp(-bce_loss)
    modulator = (1 - pt) ** gamma
    if alpha is not None:
        loss = alpha * modulator * bce_loss
    else:
        loss = modulator * bce_loss
    return loss.mean()



def config_model_loss_fn(model, args: Namespace):
    flag_use_focal = args.use_fixed_focal_cpg_loss or args.use_advanced_focal_loss
    if flag_use_focal:
        model.loss_fn = focal_loss

    else:
        model.loss_fn = my_bce_with_logits_loss
        loguru.logger.warning("! Using BCEWithLogitsLoss")



def get_score_metrics(eval_logs: Dict[str, float]):
    # eval_logs[f"eQTL/{df_name}/Margin_{margin}/Dist_{dist_str}"] = corr
    for_check_corr_valid = eval_logs.get("Corr_valid/CpG/Step", 0) * 0.5 + eval_logs.get("Corr_mega_valid/CpG/Step", 0) * 0.5
    for_check_corr_test = eval_logs.get("Corr_test/CpG/Step", 0) * 0.5 + eval_logs.get("Corr_mega_test/CpG/Step", 0) * 0.5
    for_check_corr_train = eval_logs.get("Corr_train/CpG/Step", 0) * 0.5 + eval_logs.get("Corr_mega_train/CpG/Step", 0) * 0.5
    for_check_auc_valid = eval_logs.get("Metric_valid/auc_CpG/Step", 0) * 0.5 + eval_logs.get("Metric_mega_valid/auc_CpG/Step", 0) * 0.5
    for_check_auc_test = eval_logs.get("Metric_test/auc_CpG/Step", 0) * 0.5 + eval_logs.get("Metric_mega_test/auc_CpG/Step", 0) * 0.5
    for_check_auc_train = eval_logs.get("Metric_train/auc_CpG/Step", 0) * 0.5 + eval_logs.get("Metric_mega_train/auc_CpG/Step", 0) * 0.5
    # for_check_eqtl = eval_logs.get("eQTL/GTEX_WholeBlood/Margin_1/Dist_all", 0) + eval_logs.get("eQTL/MDSs/Margin_50/Dist_all", 0)
    for_check__valid_score = for_check_corr_valid + for_check_auc_valid
    # for_check__valid_score_with_eqtl = for_check__valid_score + for_check_eqtl
    for_check__test_score = for_check_corr_test + for_check_auc_test
    # for_check__test_score_with_eqtl = for_check__test_score + for_check_eqtl
    for_check__train_score = for_check_corr_train + for_check_auc_train
    # for_check__train_score_with_eqtl = for_check__train_score + for_check_eqtl
    di = {
        "Score/corr_valid": for_check_corr_valid,
        "Score/corr_test": for_check_corr_test,
        "Score/corr_train": for_check_corr_train,
        "Score/auc_valid": for_check_auc_valid,
        "Score/auc_test": for_check_auc_test,
        "Score/auc_train": for_check_auc_train,
        # "Score/eqtl": for_check_eqtl,
        "AUC & Corr/valid": for_check__valid_score,
        "AUC & Corr/test": for_check__test_score,
        "AUC & Corr/train": for_check__train_score,
        "AUC & Corr/valid & test": for_check__valid_score + for_check__test_score,
        # "AUC & Corr & eQTL/valid": for_check__valid_score_with_eqtl,
        # "AUC & Corr & eQTL/test": for_check__test_score_with_eqtl,
        # "AUC & Corr & eQTL/train": for_check__train_score_with_eqtl,
        # "eQTL_Brief/MDS_margin50_dist_all": eval_logs.get("eQTL/MDSs/Margin_50/Dist_all", 0),

        #
        # "eQTL_Brief/GTEX_WholeBlood_margin1_dist_all": eval_logs.get("eQTL/GTEX_WholeBlood/Margin_1/Dist_all", 0),
        # "Hotpot/Golden": for_check__valid_score + for_check__test_score + for_check_eqtl * 4 + for_check__train_score * 0.5,
        # "Hotpot/without_train": for_check__valid_score + for_check__test_score + for_check_eqtl * 4,
    }
    # Dist_0-1000 or Dist_0-500 or Dist_0-2000
    for maybe_need_key in ['Dist_0-1000', 'Dist_0-500']:
        for key, value in eval_logs.items():
            if maybe_need_key in key:
                key_ = key.removeprefix('eQTL/')
                di[f"eQTL_Important/{key_}"] = value

    return di

