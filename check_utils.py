import argparse
import gc
import os
import time
from functools import cache
from typing import List, Tuple, Callable, Union, Dict, Any
import multiprocessing as mp
import loguru
import concurrent.futures
from echo_logger import dumps_json # pip install echo_logger
from loguru import logger
from torch import float32, nn
from numpy import ndarray
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from stateless import cleanup, sigmoid_first, load_ckpt, methy2bin
from variable_blocks_util import get_scaled_blocks_from_variable_blocks_by_split_and_window
from selene_sdk.sequences import Genome
from selene_util import GenomicSignalFeatures, RandomPositions, RandomPositionsSampler, SamplerDataLoader


import warnings
from sklearn.exceptions import UndefinedMetricWarning

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


class GenomicRegionDataset(Dataset):
    def __init__(self,
                 runs: List[Tuple[str, int]],
                 genome: Genome,
                 methy_data,
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


def genomic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Stack numerical data
    seq_for_model = torch.from_numpy(np.stack([item["seq_for_model"] for item in batch])).transpose(1, 2)
    methy_for_loss = torch.from_numpy(np.stack([item["methy_for_loss"] for item in batch]))

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
        "original_methy_for_corr_list": original_methy_for_corr_list,
        "original_seq_for_plotting_list": original_seq_for_plotting,
        "run_info_list": run_info,
        "end_pos_list": end_pos,
        "batch_size": len(batch)  # Actual batch size, useful for the last partial batch
    }


def process_single_track(track_name, track_data, calc_corr_using_original_myth):

    if not track_data["preds"]:

        return track_name, None

    # Concatenate all data pieces into large arrays
    all_preds = np.concatenate(track_data["preds"], axis=0)
    all_methys = np.concatenate(track_data["methys"], axis=0)
    all_orig_seqs = np.concatenate(track_data["orig_seqs"], axis=0)

    corr_target_methys = all_methys
    if calc_corr_using_original_myth and track_data["orig_methys"]:
        corr_target_methys = np.concatenate(track_data["orig_methys"], axis=0)


    aucs_and_accs = get_aucs_and_accs_aggregated(all_preds, all_methys, all_orig_seqs)
    corrs = get_corrs_aggregated(all_preds, corr_target_methys, all_orig_seqs)

    avg_loss = np.mean(track_data["losses"]) if track_data["losses"] else None

    track_result = {
        "loss": avg_loss,
        "correlations": corrs,
        "aucs_and_accs": aucs_and_accs,
        "num_regions_processed": len(track_data["preds"]),
        "total_positions": all_preds.shape[0]
    }

    return track_name, track_result


@logger.catch
def check_runs_return_multiple_pic_dict_batched_calc_total(
        model: nn.Module,
        genome: Genome,
        methy_data,
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
        test_sample_length: int = None,
        split_length=10000
):
    model_input_length = see_length
    test_region_length = split_length if test_sample_length is None else test_sample_length

    logger.info(f"Model Input Length: {model_input_length}, Test Region Length: {test_region_length}")

    train_runs, valid_runs, test_runs = get_all_run_splits_var_blocks(window_size=test_region_length)
    runs_constant_all = {
        'valid': valid_runs,
        'test': test_runs,
    }
    if tiny:
        runs_constant_all = {k: v[:batch_size * 2] for k, v in runs_constant_all.items() if v}

    final_results = {}
    model.eval()
    model.to(device=device)
    track_names = [track_name.replace(".hg38.bigwig", "") for track_name in track_names]

    for mode, runs in runs_constant_all.items():
        if not runs:
            logger.info(f"No runs specified for mode: {mode}. Skipping.")
            continue


        dataset_fetch_length = max(model_input_length, test_region_length)

        dataset = GenomicRegionDataset(
            runs=runs, genome=genome, methy_data=methy_data,
            see_length=dataset_fetch_length,
            methy_data_process=methy_data_process
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=genomic_collate_fn, num_workers=num_workers
        )

        # --- STAGE 1: Data Aggregation ---
        aggregated_data = {
            key: {"preds": [], "methys": [], "orig_seqs": [], "orig_methys": [], "losses": []}
            for key in track_names
        }

        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc=f"Aggregating data for {mode}"):
                seq_batch = batch_data["seq_for_model_batch"].to(device=device, dtype=dtype)
                bsz, _, fetched_len = seq_batch.shape

                final_pred_batch_np = None

                if test_region_length > model_input_length:
                    # Case 1: Stitching
                    if test_region_length % model_input_length != 0:
                        raise ValueError(
                            f"test_region_length ({test_region_length}) must be a multiple of model_input_length ({model_input_length}) for stitching.")
                    num_chunks = test_region_length // model_input_length
                    seq_chunks = seq_batch.view(bsz, 4, num_chunks, model_input_length)
                    seq_chunks_reshaped = seq_chunks.permute(0, 2, 1, 3).reshape(bsz * num_chunks, 4,
                                                                                 model_input_length)

                    pred_chunks_reshaped = model(seq_chunks_reshaped)
                    if isinstance(pred_chunks_reshaped, tuple):
                        pred_chunks_reshaped = pred_chunks_reshaped[0]

                    _, n_track, _ = pred_chunks_reshaped.shape
                    pred_batch_stitched = pred_chunks_reshaped.view(bsz, num_chunks, n_track, model_input_length)
                    final_pred_batch = pred_batch_stitched.permute(0, 2, 1, 3).reshape(bsz, n_track, test_region_length)
                    final_pred_batch_np = final_pred_batch.cpu().numpy()

                elif model_input_length > test_region_length:
                    # Case 2: Cropping
                    pred_batch = model(seq_batch)
                    if isinstance(pred_batch, tuple):
                        pred_batch = pred_batch[0]

                    trim = (model_input_length - test_region_length) // 2
                    start, end = trim, trim + test_region_length

                    # (seq_len is the last dim)
                    final_pred_batch_np = pred_batch.cpu().numpy()[:, :, start:end]

                else:  # model_input_length == test_region_length
                    # Case 3: Exact match
                    pred_batch = model(seq_batch)
                    if isinstance(pred_batch, tuple):
                        pred_batch = pred_batch[0]
                    final_pred_batch_np = pred_batch.cpu().numpy()


                methy_for_loss_batch_np_full = batch_data["methy_for_loss_batch"].cpu().numpy()
                orig_seq_batch_np_full = torch.stack(batch_data["original_seq_for_plotting_list"]).cpu().numpy()

                if model_input_length > test_region_length:
                    trim = (model_input_length - test_region_length) // 2
                    start, end = trim, trim + test_region_length

                    methy_for_loss_batch_np = methy_for_loss_batch_np_full[:, :, start:end]

                    orig_seq_batch_np = orig_seq_batch_np_full[:, start:end, :]
                else:
                    methy_for_loss_batch_np = methy_for_loss_batch_np_full
                    orig_seq_batch_np = orig_seq_batch_np_full

                orig_methy_batch_np = None
                can_use_orig_myth = calc_corr_using_original_myth
                if can_use_orig_myth:
                    if any(item is None for item in batch_data["original_methy_for_corr_list"]):
                        can_use_orig_myth = False
                        logger.warning(
                            "`original_methy_for_corr` contains None. Disabling `calc_corr_using_original_myth` for this run.")
                    else:
                        orig_methy_batch_np_full = torch.stack(batch_data["original_methy_for_corr_list"]).cpu().numpy()

                        if model_input_length > test_region_length:
                            orig_methy_batch_np = orig_methy_batch_np_full[:, :, start:end]
                        else:
                            orig_methy_batch_np = orig_methy_batch_np_full


                for i in range(batch_data["batch_size"]):
                    for track_idx, track_name in enumerate(track_names):
                        store = aggregated_data[track_name]
                        pred_item = final_pred_batch_np[i, track_idx:track_idx + 1, :]
                        methy_item = methy_for_loss_batch_np[i, track_idx:track_idx + 1, :]

                        if model_output_post_process:
                            pred_item = model_output_post_process(pred_item)

                        store["preds"].append(clean_seq_ndarray(pred_item))
                        store["methys"].append(clean_seq_ndarray(methy_item))
                        store["orig_seqs"].append(orig_seq_batch_np[i])

                        if can_use_orig_myth and orig_methy_batch_np is not None:
                            store["orig_methys"].append(
                                clean_seq_ndarray(orig_methy_batch_np[i, track_idx:track_idx + 1, :]))

                        if loss_fn:
                            loss = loss_fn(torch.from_numpy(pred_item).to(device),
                                           torch.from_numpy(methy_item).to(device)).mean().item()
                            store["losses"].append(loss)

        # --- STAGE 2: Aggregated Metric Calculation ---
        logger.info(f"Calculating aggregated metrics for {mode}...")
        mode_results = {}

        with concurrent.futures.ProcessPoolExecutor(16) as executor:

            future_to_track = {
                executor.submit(
                    process_single_track,
                    track_name,
                    aggregated_data[track_name],
                    calc_corr_using_original_myth
                ): track_name
                for track_name in track_names
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_track)):
                try:
                    track_name, track_result = future.result()

                    if track_result:
                        mode_results[track_name] = track_result
                        print(f"Finished processing track: {track_name}")
                except Exception as exc:
                    track_name_from_future = future_to_track[future]
                    print(f'Track {track_name_from_future} generated an exception: {exc}')

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