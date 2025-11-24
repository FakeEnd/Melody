import argparse
import json
from datetime import datetime, timedelta
import gc
import os

from _config import pdir
from cell_embedding import MelodyG


import random
import time
from functools import cache
from typing import List, Tuple, Callable, Union, Dict, Any

import loguru
from echo_logger import dumps_json
from loguru import logger
from torch import float32, nn
from numpy import ndarray
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from dataset_util import SequentialGenomeDatasetWithEmbedding
from global_constants import track_39_bigwig_file_names
from stateless import cleanup, sigmoid_first, load_ckpt, methy2bin, get_bigwig_filepaths, get_pre_process_func, \
    get_embedding_list, check_args, get_embedding
from variable_blocks_util import get_scaled_blocks_from_variable_blocks_by_split_and_window
from selene_sdk.sequences import Genome, GenomicDataset, RandomPositions

def clean_seq_ndarray(sequence: ndarray | List | torch.Tensor) -> ndarray:
    if isinstance(sequence, List):
        sequence = np.array(sequence)
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.squeeze()
        sequence = sequence.detach().cpu().to(dtype=float32).numpy()
    if sequence.ndim == 1:
        sequence = sequence[..., None]
    sequence = sequence.astype(np.float32)
    if sequence.ndim == 2 and sequence.shape[0] == 1:
        sequence = sequence.transpose()
    return sequence

import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from typing import Dict

# noinspection PyPep8Naming
def get_one_hot_embedding_mask(
        original_seq: Union[np.ndarray, torch.Tensor],  # must be one-hot encoded (shape: (seq_len, 4))
        mask_type: str = "CpG",
):
    assert original_seq.ndim == 2 and original_seq.shape[1] == 4
    if mask_type == "A":
        return original_seq[:, 0] == 1
    elif mask_type == "C":
        return original_seq[:, 1] == 1
    elif mask_type == "G":
        return original_seq[:, 2] == 1
    elif mask_type == "T":
        return original_seq[:, 3] == 1
    elif mask_type == "C/G":  # C or G
        return (original_seq[:, 1] == 1).astype(bool) | (original_seq[:, 2] == 1).astype(bool)
    elif mask_type == "CpG":
        if original_seq.shape[0] == 0:
            if isinstance(original_seq, np.ndarray):
                return np.array([], dtype=bool)
            elif isinstance(original_seq, torch.Tensor):
                return torch.tensor([], dtype=torch.bool, device=original_seq.device)
            else:
                raise TypeError("Input must be ndarray or torch.Tensor")
        mask_C = original_seq[:, 1] == 1
        mask_G = original_seq[:, 2] == 1
        mask_C_prefix = mask_C[:-1]
        mask_G_shifted = mask_G[1:]
        mask_CpG_pairs_prefix = mask_C_prefix & mask_G_shifted

        if isinstance(original_seq, np.ndarray):
            mask_CpG_both = np.zeros(original_seq.shape[0], dtype=bool)
            mask_CpG_both[:-1] = mask_CpG_both[:-1] | mask_CpG_pairs_prefix  # mark 'C'
            mask_CpG_both[1:] = mask_CpG_both[1:] | mask_CpG_pairs_prefix  # mark 'G'
        elif isinstance(original_seq, torch.Tensor):
            mask_CpG_both = torch.zeros(original_seq.shape[0], dtype=torch.bool, device=original_seq.device)
            mask_CpG_both[:-1] = mask_CpG_both[:-1] | mask_CpG_pairs_prefix  # mark 'C'
            mask_CpG_both[1:] = mask_CpG_both[1:] | mask_CpG_pairs_prefix  # mark 'G'
        else:
            raise TypeError("Input must be ndarray or torch.Tensor")
        return mask_CpG_both
    else:
        raise ValueError(f"Invalid mask_type: {mask_type}. Must be 'C', 'G', or 'CpG'.")



def get_corrs_aggregated(
        pred_all: np.ndarray,
        myth_all: np.ndarray,
        original_seq_all: np.ndarray,
) -> Dict[str, float]:

    pred_flat = pred_all.squeeze()
    myth_flat = myth_all.squeeze()
    corrs = {}


    corrs["All"] = np.corrcoef(pred_flat, myth_flat)[0, 1]


    c_or_g_mask = get_one_hot_embedding_mask(original_seq_all, mask_type="C/G")
    cpg_mask = get_one_hot_embedding_mask(original_seq_all, mask_type="CpG")


    if np.sum(c_or_g_mask) > 1:
        corrs["C/G"] = np.corrcoef(pred_flat[c_or_g_mask], myth_flat[c_or_g_mask])[0, 1]
    else:
        corrs["C/G"] = np.nan

    if np.sum(cpg_mask) > 1:
        corrs["CpG"] = np.corrcoef(pred_flat[cpg_mask], myth_flat[cpg_mask])[0, 1]
    else:
        corrs["CpG"] = np.nan

    return corrs


def get_aucs_and_accs_aggregated(
        pred_all: np.ndarray,
        myth_all: np.ndarray,
        original_seq_all: np.ndarray,
) -> Dict[str, float]:

    pred_flat = pred_all.squeeze()
    myth_binary = (myth_all.squeeze() > 0.5).astype(int)
    pred_binary = (pred_flat > 0.5).astype(int)

    metrics = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        if len(np.unique(myth_binary)) > 1:
            metrics["auc_all"] = roc_auc_score(myth_binary, pred_flat)
        else:
            metrics["auc_all"] = np.nan
        metrics["acc_all"] = np.mean(pred_binary == myth_binary)


        c_or_g_mask = get_one_hot_embedding_mask(original_seq_all, mask_type="C/G")
        cpg_mask = get_one_hot_embedding_mask(original_seq_all, mask_type="CpG")


        myth_binary_cg = myth_binary[c_or_g_mask]
        if len(np.unique(myth_binary_cg)) > 1:
            metrics["auc_C/G"] = roc_auc_score(myth_binary_cg, pred_flat[c_or_g_mask])
        else:
            metrics["auc_C/G"] = np.nan
        metrics["acc_C/G"] = np.mean(pred_binary[c_or_g_mask] == myth_binary_cg)
        metrics["methy_C/G"] = np.mean(myth_all[c_or_g_mask])
        metrics["pred_C/G"] = np.mean(pred_flat[c_or_g_mask])

        myth_binary_cpg = myth_binary[cpg_mask]
        if len(np.unique(myth_binary_cpg)) > 1:
            metrics["auc_CpG"] = roc_auc_score(myth_binary_cpg, pred_flat[cpg_mask])
        else:
            metrics["auc_CpG"] = np.nan
        metrics["acc_CpG"] = np.mean(pred_binary[cpg_mask] == myth_binary_cpg)
        metrics["methy_CpG"] = np.mean(myth_all[cpg_mask])
        metrics["pred_CpG"] = np.mean(pred_flat[cpg_mask])

    return metrics

global_random_positions: RandomPositions = None
global_pic_counter = 0
global_save_pic_dir = None

class GenomicRegionDataset(Dataset):
    def __init__(self,
                 runs: List[Tuple[str, int]],
                 genome: Genome,
                 methy_data: GenomicDataset,
                 see_length: int,
                 methy_data_process: Callable = None):
        self.runs = runs
        self.genome = genome
        self.methy_data = methy_data
        self.see_length = see_length
        self.methy_data_process = methy_data_process

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        chr_, chunk_start = self.runs[idx]
        end = chunk_start + self.see_length

        # Get sequence
        seq_for_model = self.genome.get(chr_, chunk_start, end)  # (see_length, 4)
        original_seq_for_plotting = seq_for_model.copy()  # Keep a copy for plotting

        # Get methylation data
        methy_raw = self.methy_data.get(chr_, chunk_start, end)  # (n_track, see_length)
        original_methy_for_corr = methy_raw.copy() if self.methy_data_process is not None else None

        if self.methy_data_process is not None:
            methy_processed = self.methy_data_process(methy_raw)
        else:
            methy_processed = methy_raw

        methy_for_loss = methy_processed # (n_track, processed_see_length)

        return {
            "seq_for_model": seq_for_model.astype(np.float32),  # Ensure float32 for PyTorch
            "methy_for_loss": methy_for_loss.astype(np.float32),
            "original_methy_for_corr": original_methy_for_corr.astype(
                np.float32) if original_methy_for_corr is not None else None,
            "original_seq_for_plotting": original_seq_for_plotting.astype(np.float32),
            "run_info": (chr_, chunk_start),
            "end_pos": end
        }

class GenomicCellDataset(Dataset):
    def __init__(self,
                 runs: List[Tuple[str, int]],
                 genome: Genome,
                 methy_data: GenomicDataset,
                 cell_embedding,
                 see_length: int,
                 methy_data_process: Callable = None):
        self.runs = runs
        self.genome = genome
        self.methy_data = methy_data
        self.cell_embedding = np.mean(cell_embedding, axis=0)
        self.see_length = see_length
        self.methy_data_process = methy_data_process

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        chr_, chunk_start = self.runs[idx]
        end = chunk_start + self.see_length

        # Get sequence
        seq_for_model = self.genome.get(chr_, chunk_start, end)  # (see_length, 4)
        original_seq_for_plotting = seq_for_model.copy()  # Keep a copy for plotting

        # Get methylation data
        methy_raw = self.methy_data.get(chr_, chunk_start, end)  # (n_track, see_length)
        original_methy_for_corr = methy_raw.copy() if self.methy_data_process is not None else None

        if self.methy_data_process is not None:
            methy_processed = self.methy_data_process(methy_raw)
        else:
            methy_processed = methy_raw

        methy_for_loss = methy_processed # (n_track, processed_see_length)

        return {
            "seq_for_model": seq_for_model.astype(np.float32),  # Ensure float32 for PyTorch
            "methy_for_loss": methy_for_loss.astype(np.float32),
            "cell_embedding": self.cell_embedding.astype(np.float32),
            "original_methy_for_corr": original_methy_for_corr.astype(
                np.float32) if original_methy_for_corr is not None else None,
            "original_seq_for_plotting": original_seq_for_plotting.astype(np.float32),
            "run_info": (chr_, chunk_start),
            "end_pos": end
        }

def genomic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Stack numerical data
    seq_for_model = torch.from_numpy(np.stack([item["seq_for_model"] for item in batch])).transpose(1, 2)
    methy_for_loss = torch.from_numpy(np.stack([item["methy_for_loss"] for item in batch]))
    cell_embedding = torch.from_numpy(np.stack([item["cell_embedding"] for item in batch]))

    # Collect other data into lists
    original_methy_for_corr_list = []
    for item in batch:
        # Handle None case for original_methy_for_corr
        if item["original_methy_for_corr"] is not None:
            original_methy_for_corr_list.append(torch.from_numpy(item["original_methy_for_corr"]))
        else:
            original_methy_for_corr_list.append(None)  # Keep as None if original was None

    original_seq_for_plotting = [torch.from_numpy(item["original_seq_for_plotting"]) for item in batch]
    run_info = [item["run_info"] for item in batch]
    end_pos = [item["end_pos"] for item in batch]

    return {
        "seq_for_model_batch": seq_for_model,
        "methy_for_loss_batch": methy_for_loss,
        "cell_embedding_batch": cell_embedding,
        "original_methy_for_corr_list": original_methy_for_corr_list,
        "original_seq_for_plotting_list": original_seq_for_plotting,
        "run_info_list": run_info,
        "end_pos_list": end_pos,
        "batch_size": len(batch)  # Actual batch size, useful for the last partial batch
    }


@logger.catch
def check_runs_return_multiple_pic_dict_batched_calc_total(
        model: nn.Module,
        genome: Genome,
        methy_data_list: List[GenomicDataset],
        cell_embedding_list:List,
        track_names: List[str],
        see_length: int = 10000,
        batch_size: int = 32,
        device: str = 'cuda',
        dtype=torch.float32,
        loss_fn: Callable = None,
        methy_data_process: Callable = None,
        model_output_post_process: Callable = None,
        calc_corr_using_original_myth: bool = False,
        tiny: bool = False,
        num_workers: int = 2,
        test_sample_length: int = None
):
    # ... (get_all_run_splits_var_blocks and runs_constant_all setup remains the same) ...
    train_runs, valid_runs, test_runs = get_all_run_splits_var_blocks(window_size=see_length if test_sample_length is None else test_sample_length)
    runs_constant_all = {
        'valid': valid_runs,
        'test': test_runs,
    }
    if tiny:
        # Make sure tiny mode has enough data for at least one full batch
        runs_constant_all = {k: v[:batch_size * 2] for k, v in runs_constant_all.items() if v}

    final_results = {}
    model.eval()
    # cleanup()
    model.to(device=device)
    track_names = [track_name.replace(".hg38.bigwig", "") for track_name in track_names]
    track_names = random.sample(track_names, 3)
    loguru.logger.info(f"track_names: {track_names}")

    for mode, runs in runs_constant_all.items():
        if not runs:
            logger.info(f"No runs specified for mode: {mode}. Skipping.")
            continue

        track_keys_for_agg = track_names
        aggregated_data = {
            key: {"preds": [], "methys": [], "orig_seqs": [], "orig_methys": [], "losses": []}
            for key in track_keys_for_agg
        }

        for track_idx, track_name in enumerate(tqdm(track_names, desc=f"Aggregating data for {mode}")):

            dataset = GenomicCellDataset(
                runs=runs, genome=genome, methy_data=methy_data_list[track_idx], cell_embedding=cell_embedding_list[track_idx],
                see_length=see_length, methy_data_process=methy_data_process
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                collate_fn=genomic_collate_fn, num_workers=num_workers
            )

            # --- STAGE 1: Data Aggregation ---
            # Initialize storage for aggregated data. This will live in RAM.


            with torch.no_grad():
                for batch_data in dataloader:
                    seq_batch = batch_data["seq_for_model_batch"].to(device=device, dtype=dtype)
                    cell_embedding_batch = batch_data["cell_embedding_batch"].to(device=device, dtype=dtype)
                    pred_batch = model(seq_batch, cell_embedding_batch)
                    if isinstance(pred_batch, tuple):
                        pred_batch = pred_batch[0]

                    # Move batch to CPU once for efficiency, then process
                    pred_batch_np = pred_batch.cpu().numpy()
                    methy_for_loss_batch_np = batch_data["methy_for_loss_batch"].cpu().numpy()
                    orig_seq_batch_np = torch.stack(batch_data["original_seq_for_plotting_list"]).cpu().numpy()

                    orig_methy_batch_np = None
                    can_use_orig_myth = calc_corr_using_original_myth
                    if can_use_orig_myth:
                        if any(item is None for item in batch_data["original_methy_for_corr_list"]):
                            can_use_orig_myth = False
                            logger.warning(
                                "`original_methy_for_corr` contains None. Disabling `calc_corr_using_original_myth` for this run.")
                        else:
                            orig_methy_batch_np = torch.stack(batch_data["original_methy_for_corr_list"]).cpu().numpy()

                    # Loop through items in the batch to append to lists
                    for i in range(batch_data["batch_size"]):
                        # for track_idx, track_name in enumerate(track_names):
                        store = aggregated_data[track_name]
                        pred_item = pred_batch_np[i, 0, :]
                        methy_item = methy_for_loss_batch_np[i, 0, :]

                        if model_output_post_process:
                            pred_item = model_output_post_process(pred_item)

                        store["preds"].append(clean_seq_ndarray(pred_item))
                        store["methys"].append(clean_seq_ndarray(methy_item))
                        store["orig_seqs"].append(orig_seq_batch_np[i])
                        if can_use_orig_myth:
                            store["orig_methys"].append(
                                clean_seq_ndarray(orig_methy_batch_np[i, 0, :]))

                        if loss_fn:
                            loss = loss_fn(torch.from_numpy(pred_item).to(device),
                                           torch.from_numpy(methy_item).to(device)).mean().item()
                            store["losses"].append(loss)

        # --- STAGE 2: Aggregated Metric Calculation ---
        logger.info(f"Calculating aggregated metrics for {mode}...")
        mode_results = {}

        for track_name in tqdm(track_keys_for_agg, desc=f"Calculating aggregating data for {mode}"):
            store = aggregated_data[track_name]
            if not store["preds"]:
                continue

            # Concatenate all data pieces into large arrays
            all_preds = np.concatenate(store["preds"], axis=0)
            all_methys = np.concatenate(store["methys"], axis=0)
            all_orig_seqs = np.concatenate(store["orig_seqs"], axis=0)

            corr_target_methys = all_methys
            if calc_corr_using_original_myth and store["orig_methys"]:
                corr_target_methys = np.concatenate(store["orig_methys"], axis=0)

            # Call the new, efficient metric functions
            aucs_and_accs = get_aucs_and_accs_aggregated(all_preds, all_methys, all_orig_seqs)
            corrs = get_corrs_aggregated(all_preds, corr_target_methys, all_orig_seqs)

            avg_loss = np.mean(store["losses"]) if store["losses"] else None

            track_result = {
                "loss": avg_loss,
                "correlations": corrs,
                "aucs_and_accs": aucs_and_accs,
                "num_regions_processed": len(store["preds"]),
                "total_positions": all_preds.shape[0]
            }

            mode_results[track_name] = track_result

        final_results[mode] = mode_results

    # cleanup()
    return final_results

def extract_tissue_name_from_path(file_path):
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    first_underscore = name.find('_')
    last_dash = name.rfind('-')

    if first_underscore != -1 and last_dash != -1 and first_underscore < last_dash:
        return name[first_underscore + 1: last_dash]
    else:
        return None

@logger.catch
def check_runs_return_multiple_pic_dict_batched_calc_total_all(
        model: nn.Module,
        genome: Genome,
        methy_data_list: List[GenomicDataset],
        # cell_embedding_dict,
        cell_embedding_list,
        track_names: List[str],
        test_track,
        see_length: int = 10000,
        batch_size: int = 32,
        device: str = 'cuda',
        dtype=torch.float32,
        loss_fn: Callable = None,
        methy_data_process: Callable = None,
        model_output_post_process: Callable = None,
        calc_corr_using_original_myth: bool = False,
        tiny: bool = False,
        num_workers: int = 2,
        test_sample_length: int = None,
        split_length=10000
):

    model_input_length = see_length
    test_region_length = split_length if test_sample_length is None else test_sample_length

    # ... (get_all_run_splits_var_blocks and runs_constant_all setup remains the same) ...
    train_runs, valid_runs, test_runs = get_all_run_splits_var_blocks(window_size=see_length if test_sample_length is None else test_sample_length)
    runs_constant_all = {
        'sample_valid': valid_runs,
        'sample_test': test_runs,
    }
    if tiny:
        # Make sure tiny mode has enough data for at least one full batch
        runs_constant_all = {k: v[:batch_size * 2] for k, v in runs_constant_all.items() if v}

    final_results = {}
    model.eval()
    # # cleanup()
    # model.to(device=device)
    track_names = [track_name.replace(".hg38.bigwig", "") for track_name in track_names]
    # track_names = random.sample(track_names, 3)
    loguru.logger.info(f"track_names: {track_names}")
    #
    for mode, runs in runs_constant_all.items():
        if not runs:
            logger.info(f"No runs specified for mode: {mode}. Skipping.")
            continue

        track_keys_for_agg = track_names
        aggregated_data = {
            key: {"preds": [], "methys": [], "orig_seqs": [], "orig_methys": [], "losses": []}
            for key in track_keys_for_agg
        }

        for track_idx, track_name in enumerate(tqdm(track_names, desc=f"Aggregating data for {mode}")):

            dataset = GenomicCellDataset(
                runs=runs, genome=genome, methy_data=methy_data_list[track_idx], cell_embedding=cell_embedding_list[track_idx],
                see_length=see_length, methy_data_process=methy_data_process
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                collate_fn=genomic_collate_fn, num_workers=num_workers
            )

            # --- STAGE 1: Data Aggregation ---
            # Initialize storage for aggregated data. This will live in RAM.


            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc=f"predicting {mode}", leave=False):
                    seq_batch = batch_data["seq_for_model_batch"].to(device=device, dtype=dtype)
                    cell_embedding_batch = batch_data["cell_embedding_batch"].to(device=device, dtype=dtype)
                    pred_batch = model(seq_batch, cell_embedding_batch)
                    if isinstance(pred_batch, tuple):
                        pred_batch = pred_batch[0]

                    # Move batch to CPU once for efficiency, then process
                    pred_batch_np = pred_batch.cpu().numpy()
                    methy_for_loss_batch_np = batch_data["methy_for_loss_batch"].cpu().numpy()
                    orig_seq_batch_np = torch.stack(batch_data["original_seq_for_plotting_list"]).cpu().numpy()

                    orig_methy_batch_np = None
                    can_use_orig_myth = calc_corr_using_original_myth
                    if can_use_orig_myth:
                        if any(item is None for item in batch_data["original_methy_for_corr_list"]):
                            can_use_orig_myth = False
                            logger.warning(
                                "`original_methy_for_corr` contains None. Disabling `calc_corr_using_original_myth` for this run.")
                        else:
                            orig_methy_batch_np = torch.stack(batch_data["original_methy_for_corr_list"]).cpu().numpy()

                    # Loop through items in the batch to append to lists
                    for i in range(batch_data["batch_size"]):
                        # for track_idx, track_name in enumerate(track_names):
                        store = aggregated_data[track_name]
                        pred_item = pred_batch_np[i, 0, :]
                        methy_item = methy_for_loss_batch_np[i, 0, :]

                        if model_output_post_process:
                            pred_item = model_output_post_process(pred_item)

                        store["preds"].append(clean_seq_ndarray(pred_item))
                        store["methys"].append(clean_seq_ndarray(methy_item))
                        store["orig_seqs"].append(orig_seq_batch_np[i])
                        if can_use_orig_myth:
                            store["orig_methys"].append(
                                clean_seq_ndarray(orig_methy_batch_np[i, 0, :]))

                        # if loss_fn:
                        #     loss = loss_fn(torch.from_numpy(pred_item).to(device),
                        #                    torch.from_numpy(methy_item).to(device)).mean().item()
                        #     store["losses"].append(loss)

        # --- STAGE 2: Aggregated Metric Calculation ---
        logger.info(f"Calculating aggregated metrics for {mode}...")
        mode_results = {}

        for track_name in tqdm(track_keys_for_agg, desc=f"Calculating aggregating data for {mode}"):
            store = aggregated_data[track_name]
            if not store["preds"]:
                continue

            # Concatenate all data pieces into large arrays
            all_preds = np.concatenate(store["preds"], axis=0)
            all_methys = np.concatenate(store["methys"], axis=0)
            all_orig_seqs = np.concatenate(store["orig_seqs"], axis=0)

            corr_target_methys = all_methys
            if calc_corr_using_original_myth and store["orig_methys"]:
                corr_target_methys = np.concatenate(store["orig_methys"], axis=0)

            # Call the new, efficient metric functions
            aucs_and_accs = get_aucs_and_accs_aggregated(all_preds, all_methys, all_orig_seqs)
            corrs = get_corrs_aggregated(all_preds, corr_target_methys, all_orig_seqs)

            avg_loss = np.mean(store["losses"]) if store["losses"] else None

            track_result = {
                "loss": avg_loss,
                "correlations": corrs,
                "aucs_and_accs": aucs_and_accs,
                "num_regions_processed": len(store["preds"]),
                "total_positions": all_preds.shape[0]
            }

            mode_results[track_name] = track_result

        final_results[mode] = mode_results

    # ===================== PART 2: valid / test =====================
    sequential_tasks = {
        'valid': ['chr10'],
        'test': ['chr8', 'chr9'],
    }

    dataset_fetch_length = max(model_input_length, test_region_length)
    stride = test_region_length

    for mode, chrom_list in sequential_tasks.items():
        aggregated_data = {
            key: {"preds": [], "methys": [], "orig_seqs": [], "orig_methys": [], "losses": []}
            for key in track_names
        }

        for track_idx, track_name in enumerate(tqdm(track_names, desc=f"Aggregating data for {mode}")):
            # cell_embedding = cell_embedding_dict[extract_tissue_name_from_path(track_name)]
            cell_embedding_ = cell_embedding_list[track_idx]
            methy_data_ = methy_data[track_idx]
            with torch.no_grad():
                for chrom in chrom_list:
                    logger.info(f"Sequential scanning {chrom} for mode={mode}")
                    dataset = SequentialGenomeDatasetWithEmbedding(
                        genome=genome,
                        methy_data=methy_data_,
                        cell_embedding=cell_embedding_,
                        chromosomes_to_scan=[chrom],
                        window_size=dataset_fetch_length,
                        stride=stride,
                    )
                    if len(dataset) == 0:
                        logger.warning(f"No data found for {chrom} in mode {mode}. Skipping chromosome.")
                        continue

                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0
                    )

                    for batch_idx, (dna_batch, methy_batch_raw, cell_embedding) in enumerate(
                            tqdm(dataloader, desc=f"Aggregating data for {mode} ({chrom})", leave=False)
                    ):
                        if tiny and batch_idx >= 2:
                            logger.info(f"[tiny mode] early stop after {batch_idx} batches for {mode}/{chrom}.")
                            break

                        # dna_batch: [B, L_fetch, 4]; methy_batch_raw: [B, n_track, L_fetch]
                        bsz, L_fetch, _ = dna_batch.shape

                        seq_batch = dna_batch.to(device=device, dtype=dtype).permute(0, 2, 1)  # [B, 4, L_fetch]
                        cell_embedding = cell_embedding.to(device=device, dtype=dtype)
                        pred_batch = model(seq_batch, cell_embedding)
                        if isinstance(pred_batch, tuple):
                            pred_batch = pred_batch[0]
                        final_pred_batch_np = pred_batch.cpu().numpy()


                        methy_batch_raw_np_full = methy_batch_raw.cpu().numpy()  # [B, n_track, L_fetch]
                        dna_batch_np_full = dna_batch.cpu().numpy()  # [B, L_fetch, 4]


                        if methy_data_process is not None:
                            methy_for_loss_batch_np_full = methy_data_process(methy_batch_raw_np_full)
                        else:
                            methy_for_loss_batch_np_full = methy_batch_raw_np_full

                        methy_for_loss_batch_np = methy_for_loss_batch_np_full
                        dna_batch_np = dna_batch_np_full
                        if calc_corr_using_original_myth:
                            orig_methy_batch_np = methy_batch_raw_np_full
                        else:
                            orig_methy_batch_np = None


                        for i in range(bsz):
                            seq_np = dna_batch_np[i]  # [L_eff, 4]


                            if np.all(seq_np == 0.25):
                                continue

                            # for track_idx, track_name in enumerate(track_names):
                            store = aggregated_data[track_name]

                            pred_item = final_pred_batch_np[i, 0, :]
                            methy_item = methy_for_loss_batch_np[i, 0, :]

                            if model_output_post_process:
                                pred_item = model_output_post_process(pred_item)

                            store["preds"].append(clean_seq_ndarray(pred_item))
                            store["methys"].append(clean_seq_ndarray(methy_item))
                            store["orig_seqs"].append(clean_seq_ndarray(seq_np))

                            if calc_corr_using_original_myth and orig_methy_batch_np is not None:
                                orig_methy_item = orig_methy_batch_np[i, track_idx:track_idx + 1, :]
                                store["orig_methys"].append(clean_seq_ndarray(orig_methy_item))

        logger.info(f"Calculating aggregated metrics for {mode}...")
        mode_results = {}

        for track_name in tqdm(track_names, desc=f"Calculating aggregating data for {mode}"):
            store = aggregated_data[track_name]
            if not store["preds"]:
                continue

            # Concatenate all data pieces into large arrays
            all_preds = np.concatenate(store["preds"], axis=0)
            all_methys = np.concatenate(store["methys"], axis=0)
            all_orig_seqs = np.concatenate(store["orig_seqs"], axis=0)

            corr_target_methys = all_methys
            # if calc_corr_using_original_myth and store["orig_methys"]:
            #     corr_target_methys = np.concatenate(store["orig_methys"], axis=0)

            # Call the new, efficient metric functions
            aucs_and_accs = get_aucs_and_accs_aggregated(all_preds, all_methys, all_orig_seqs)
            corrs = get_corrs_aggregated(all_preds, corr_target_methys, all_orig_seqs)

            avg_loss = np.mean(store["losses"]) if store["losses"] else None

            track_result = {
                "loss": avg_loss,
                "correlations": corrs,
                "aucs_and_accs": aucs_and_accs,
                "num_regions_processed": len(store["preds"]),
                "total_positions": all_preds.shape[0]
            }

            mode_results[track_name] = track_result

        final_results[mode] = mode_results

    # cleanup()
    return final_results

@cache
def get_all_run_splits_var_blocks(window_size: int) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:

    logger.info(f"--- Generating all data splits for window size {window_size} (will be cached) ---")


    train_blocks_full = get_scaled_blocks_from_variable_blocks_by_split_and_window(
        window=window_size,
        mode="train"
    )
    train_runs = [(chrom, start) for chrom, start, end in train_blocks_full]
    logger.info(f"Generated {len(train_runs)} runs for the TRAIN set.")


    valid_blocks_full = get_scaled_blocks_from_variable_blocks_by_split_and_window(
        window=window_size,
        mode="validation"
    )
    valid_runs = [(chrom, start) for chrom, start, end in valid_blocks_full]
    logger.info(f"Generated {len(valid_runs)} runs for the VALIDATION set.")


    test_blocks_full = get_scaled_blocks_from_variable_blocks_by_split_and_window(
        window=window_size,
        mode="test"
    )
    test_runs = [(chrom, start) for chrom, start, end in test_blocks_full]
    logger.info(f"Generated {len(test_runs)} runs for the TEST set.")

    return train_runs, valid_runs, test_runs


simgle_blood_b_ckpts = [
    ("scgpt_stage_1_415208", pdir + "/data/checkpoints/Melody-G1.pth"),
]

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

if __name__ == '__main__':
    # Example usage of the functions
    start_time_str = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--from_snapshot',
                        type=str, help='Path to the checkpoint directory to resume training from')
    parser.add_argument('--auto-resume', action='store_true',
                        help='Automatically resume latest checkpoint. If no checkpoint is available, do not set it.')
    parser.add_argument('--project', type=str, default='Melody', help='Base directory for data and checkpoints')
    parser.add_argument('--lab_name', type=str, default=f'scgpt_1_stage_{start_time_str}',
                        help='Name of the lab for logging')
    parser.add_argument('--from_ckpt', type=str,
                        default=None, help='Path to the checkpoint file to load')
    parser.add_argument('--appointed_csv_dataset', type=str, default=None, help='Path to the CSV file for the dataset')
    parser.add_argument('--use_cg_loss', default=True, action='store_true', help='Use CG loss in training')
    parser.add_argument('--use_avg_loss', default=True, action='store_true', help='Use average loss in training')
    parser.add_argument('--wcg', type=float, default=0.01, help='Weight for CG loss')
    parser.add_argument('--wavg', type=float, default=1, help='Weight for average loss')
    parser.add_argument('--use_mt_loss', action='store_true', help='Use MT loss in training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size, default: 64')
    parser.add_argument('--window_size', type=int, default=10000, help='Window size, default: 10000')
    parser.add_argument('--seed', type=int, default=5, help='Random seed, default: 4')
    parser.add_argument('--genome_path', type=str,
                        default=pdir + '/data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
                        help='Genome .fa file path')
    parser.add_argument('--bigwigs_dir', type=str, default=pdir + '/data/bigwigs/',
                        help='BigWigs directory')
    # parser.add_argument('--bigwigs_files', type=str, default=['GSM5652317_Blood-B-Z000000UB.hg38.bigwig'], nargs='+', help='BigWigs file')
    parser.add_argument('--cpg_cls_bin_width', type=int, default=5,
                        help='Bin width for CpG count classification, default: 5')
    parser.add_argument('--cpg_cls_n', type=int, default=7, help='Max number of classes for CpG counts, default: 7')
    parser.add_argument('--use_advanced_focal_loss', action='store_true', help='Use Advanced focal loss in training')
    parser.add_argument('--use_fixed_focal_cpg_loss', default=True, action='store_true',
                        help='Use Fixed focal loss in training')
    parser.add_argument('--low_methy_cpg_focal_weight', type=float, default=32.0,
                        help='Focal loss low_methy_cpg_focal_weight weight, default: 32.0')
    parser.add_argument('--any_cpg_focal_weight', type=float, default=8.0,
                        help='Focal loss any_cpg_focal_weight weight, default: 8.0')
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
    parser.add_argument('--use_39_track', action='store_true',
                        help='Use 39 track, default: False. If true, will use 39 track for training.')
    parser.add_argument('--model_cls', type=str, default='MelodyG', )
    parser.add_argument('--no-compile', default=False, action='store_false', dest='compile',
                        help='Do not compile the model, default: Compile. If --no-compile, will not compile the model.')
    parser.add_argument('--test', action='store_true',
                        help='Test mode, default: False. If true, will run the test mode.')
    parser.add_argument('--cell_embedding_path', type=str,
                        default=pdir + '/embeddings/scgpt',)

    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    args = parser.parse_args()
    args.bigwigs_files = track_39_bigwig_file_names
    args = check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    BATCH_SIZE, WINDOW_SIZE, SEED, DEVICE = args.batch_size, args.window_size, args.seed, 'cuda'
    bigwig_files, track_names = get_bigwig_filepaths(dir_=args.bigwigs_dir,
                                                     filenames=args.bigwigs_files,
                                                     fallback_dirs=[pdir + '/data/bigwigs/',])
    pre_process_func: Callable = get_pre_process_func(args.meth_pre_process_func)

    test_track = ['Pancreas-Delta', 'Blood-Granulocytes', 'Blood-Monocytes', 'Aorta-Endothel', 'Cortex-Neuron']

    genome = Genome(input_path=args.genome_path, cuda=True if DEVICE == 'cuda' else False)
    methy_data = [GenomicDataset([m], genome, storage="BigWig") for m in bigwig_files]

    cell_embeddings = get_embedding_list(bigwig_files, args.cell_embedding_path)

    args.from_ckpt = simgle_blood_b_ckpts[0][1] if args.from_ckpt is None else args.from_ckpt
    device_ = 'cuda'
    for ckpt_name, ckpt_path in simgle_blood_b_ckpts:
        logger.info("=" * 80)
        logger.info(f"Testing Model: {ckpt_name}")
        logger.info(f"checkpoint path: {ckpt_path}")

        args.from_ckpt = ckpt_path
        model = MelodyG(args, n_track=1)
        model = model.to(device=device_)
        load_ckpt(model, args.from_ckpt)

        if hasattr(torch, 'compile'):
            model = torch.compile(model)

        model.eval()

        check_loss_fn = nn.BCELoss(reduction='none')

        obj_ = check_runs_return_multiple_pic_dict_batched_calc_total_all(
                model=model,
                genome=genome,
                methy_data_list=methy_data,
                cell_embedding_list=cell_embeddings,
                see_length=args.window_size,
                track_names=track_names,
                test_track=test_track,
                device=DEVICE,
                batch_size=64,
                loss_fn=check_loss_fn,
                methy_data_process=pre_process_func if not (
                        args.meth_pre_process_func is None or args.meth_pre_process_func == 'None') else methy2bin,
                calc_corr_using_original_myth=True,
                tiny=False
            )

        data_native = convert_to_native(obj_)
        with open("stage_1_data_all_415208.json", "w", encoding="utf-8") as f:
            json.dump(data_native, f, ensure_ascii=False, indent=4)

        logger.info(f"--- model  '{ckpt_name}' results ---")
        for split, metrics in obj_.items():
            split_name = split.upper()
            loss = metrics.get('loss', 'N/A')
            corr = metrics.get('correlations', {})
            auc_acc = metrics.get('aucs_and_accs', {})

            logger.info(
                f"  [{split_name}] Correlations: All={corr.get('All', 0):.4f}, C/G={corr.get('C/G', 0):.4f}, CpG={corr.get('CpG', 0):.4f}")
            logger.info(
                f"  [{split_name}] AUC: All={auc_acc.get('auc_all', 0):.4f}, C/G={auc_acc.get('auc_C/G', 0):.4f}, CpG={auc_acc.get('auc_CpG', 0):.4f}")
            logger.info(
                f"  [{split_name}] Accuracy: All={auc_acc.get('acc_all', 0):.4f}, C/G={auc_acc.get('acc_C/G', 0):.4f}, CpG={auc_acc.get('acc_CpG', 0):.4f}")
        logger.info(f"--- --- ---")

        del model
        del obj_
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(3)

    logger.info("=" * 80)
