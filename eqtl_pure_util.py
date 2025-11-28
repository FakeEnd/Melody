import os
from functools import cache
from pathlib import Path
from typing import List, Dict, Tuple, Any, Callable, Optional, Union

import loguru
import matplotlib.pyplot as plt
from PIL import Image
from echo_logger import deprecated
from matplotlib.axes import Axes  # Import Axes type hint
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import logging  # Optional: for logging errors/warnings
import argparse
import io
import wandb
import torch
import numpy as np
from datetime import datetime
import torch.nn as nn
import seaborn as sns
from PIL import Image as PILImage
import copy

from _config import pdir

start_time_str = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
from selene_sdk.sequences import Genome
from selene_util import GenomicSignalFeatures, RandomPositions, RandomPositionsSampler, SamplerDataLoader

# Optimized GATC2onehot using a dictionary
BASE_TO_ONEHOT = {
    'A': [1, 0, 0, 0], 'a': [1, 0, 0, 0],
    'C': [0, 1, 0, 0], 'c': [0, 1, 0, 0],
    'G': [0, 0, 1, 0], 'g': [0, 0, 1, 0],
    'T': [0, 0, 0, 1], 't': [0, 0, 0, 1],
    # Handle 'N' or other characters if necessary, e.g., return zeros or raise error
    'N': [0, 0, 0, 0], 'n': [0, 0, 0, 0]
}
def GATC2onehot(base: str) -> list[int]:
    """Converts a single DNA base (A, C, G, T) to a one-hot encoded list."""
    return BASE_TO_ONEHOT.get(base, [0, 0, 0, 0])  # Return zeros for unknown bases



"""
```
EPIC.csv
skin.csv
GTEX_BreastMammaryTissue.csv
GTEX_ColonTransverse.csv
GTEX_KidneyCortex.csv
GTEX_Lung.csv
GTEX_MuscleSkeletal.csv
GTEX_Ovary.csv
GTEX_Prostate.csv
GTEX_Testis.csv
GTEX_WholeBlood.csv
CPG_units.csv
MDSs.csv
```
"""

# 3 directories, 13 files
# --- Load DataFrames ---

@cache
def warn_once(msg: str):
    loguru.logger.warning(msg)

df_dict = {}
eqtl_data_base_dir = Path(pdir) / 'meqtl/dataset/processed'
for df_name_dir in ["GTEX", "EPIGEN", "Olafur_2024"]:
    for filename in os.listdir(os.path.join(eqtl_data_base_dir, df_name_dir)):
        if filename.endswith(".csv"):
            df_path = os.path.join(eqtl_data_base_dir, df_name_dir, filename)
            df = pd.read_csv(df_path, header=0, dtype=str)
            # Convert column names to lowercase
            # df.columns = [col.lower() for col in df.columns]
            # Store the DataFrame in a dictionary with the filename as the key
            df_dict[filename.removesuffix('.csv')] = df
# --- MODIFIED SNPEffectDataset ---
class SNPEffectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, genome_accessor, half_model_input_len=5000, df_name=None):  # Removed margin
        self.df = df
        self.genome = genome_accessor
        self.half_model_input_len = half_model_input_len
        # required_columns = ['Chrom', 'SeqVariant_start', 'SeqVariant_end', 'CpG_start', 'CpG_end', 'SeqVariant_alt',
        #                     'CpG_alt_methylrate', 'CpG_ref_methylrate']
        required_columns = ['chrom', 'SNP_region_start', 'SNP_region_end', 'SNP_ref', 'SNP_alt', 'CPG_region_start',
                            'CPG_region_end', 'effect_size', 'cpg_number']
        for col in required_columns:
            if col not in self.df.columns:
                # if missing 'cpg_number', log warn and set to 1.
                if col == 'cpg_number':
                    warn_once(f"Dataset {df_name} Column '{col}' not found in DataFrame. Setting default value to 1.")
                    self.df[col] = 1
                else:
                    raise ValueError(f"Missing required column: {col}. Required columns are: {required_columns}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            chr_i = row['chrom']
            SeqVariant_start = int(float(row['SNP_region_start']))
            SeqVariant_end = int(float(row['SNP_region_end']))
            CpG_start = int(float(row['CPG_region_start']))
            CpG_end = int(float(row['CPG_region_end']))
            SeqVariant_alt = row['SNP_alt']
            SeqVariant_ref = row['SNP_ref']
            observed_effect = float(row['effect_size'])
            cpg_number = int(float(row['cpg_number']))

            if len(SeqVariant_alt) != 1: return None
            if SeqVariant_start != SeqVariant_end: return None

            center_point = (CpG_start + SeqVariant_start) // 2
            fetch_seq_start = center_point - self.half_model_input_len  # Genomic coordinate of fetched sequence start
            fetch_seq_end = center_point + self.half_model_input_len
            if fetch_seq_start < 0: fetch_seq_start = 0

            seq_test = self.genome.get(chr_i, fetch_seq_start, fetch_seq_end)
            if seq_test is None or seq_test.shape[0] == 0: return None
            if not isinstance(seq_test, np.ndarray): seq_test = np.array(seq_test)
            seq_test = seq_test.astype(np.float32)

            seq_test_mutated = seq_test.copy()
            mutation_rel_idx = SeqVariant_start - fetch_seq_start - 1  # 0-based relative to fetched seq

            if not (0 <= mutation_rel_idx < seq_test.shape[0]): return None

        # Convert nucleotide to one-hot encoding
            # A = [1,0,0,0], C = [0,1,0,0], G = [0,0,1,0], T = [0,0,0,1]
            nucleotide_to_onehot = {
                'A': [1, 0, 0, 0],
                'C': [0, 1, 0, 0],
                'G': [0, 0, 1, 0],
                'T': [0, 0, 0, 1]
            }

            # Check if reference sequence matches expected reference nucleotide
            seq_test[mutation_rel_idx:mutation_rel_idx + 1, :] = nucleotide_to_onehot[SeqVariant_ref]
            # ref_onehot = nucleotide_to_onehot[SeqVariant_ref]
            # current_base = seq_test_mutated[mutation_rel_idx].tolist()
            # if current_base != ref_onehot:
            #     print(f"Reference mismatch: Expected {idx} {SeqVariant_ref} {ref_onehot}, found {current_base} at position {mutation_rel_idx} {SeqVariant_alt} ")
            #     return None  # Skip this entry if reference doesn't match

            # Apply mutation at the variant position
            seq_test_mutated[mutation_rel_idx:mutation_rel_idx + 1, :] = nucleotide_to_onehot[SeqVariant_alt]

            cpg_rel_start_in_seq = CpG_start - fetch_seq_start
            cpg_rel_end_in_seq = CpG_end - fetch_seq_start


            return {
                'dist': abs(CpG_start - SeqVariant_start),  # Distance between CpG and SNP
                'ref_seq': seq_test,
                'alt_seq': seq_test_mutated,
                'observed_effect': observed_effect,
                'cpg_rel_start_in_seq': cpg_rel_start_in_seq,
                'cpg_rel_end_in_seq': cpg_rel_end_in_seq,
                'original_index': idx,
                'cpg_number': cpg_number
            }
        except Exception as e:
            print(f"Error processing index {idx} in Dataset: {e}")
            # return None


# --- MODIFIED Collate Function ---
def collate_snp_data(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None

    ref_seqs = np.stack([item['ref_seq'] for item in batch])
    alt_seqs = np.stack([item['alt_seq'] for item in batch])
    observed_effects = np.array([item['observed_effect'] for item in batch])
    # New fields
    dists = np.array([item['dist'] for item in batch])  # Distances
    cpg_rel_starts = np.array([item['cpg_rel_start_in_seq'] for item in batch])
    cpg_rel_ends = np.array([item['cpg_rel_end_in_seq'] for item in batch])
    cpg_numbers = np.array([item['cpg_number'] for item in batch])
    original_indices = [item['original_index'] for item in batch]

    return {
        'ref_seq_batch': torch.from_numpy(ref_seqs),
        'alt_seq_batch': torch.from_numpy(alt_seqs),
        'observed_batch': torch.from_numpy(observed_effects).float(),
        'cpg_rel_start_batch': cpg_rel_starts,  # New
        'cpg_rel_end_batch': cpg_rel_ends,  # New
        'index_batch': original_indices,
        'dists': dists,  # New
        'cpg_number_batch': cpg_numbers,
    }


# --- MODIFIED check_meqtl_batched (now for multiple margins) ---
def check_meqtl_batched_multi_margin(
        model: nn.Module,
        genome: Genome,
        data_frame: pd.DataFrame,
        margins_list: List[int],  # Takes a list of margins
        model_name: str,
        batch_size: int = 32,
        model_post_process: Callable = None,
        draw=False,
        df_name=None,
        **kwargs  # For half_model_input_len etc. for SNPEffectDataset
) -> Dict[int, Tuple[float, Image.Image]]:
    """
    Checks SNP effects for multiple margins by running the model once per batch.
    Returns a dictionary: {margin: (correlation, plot_image)}
    """
    snp_dataset = SNPEffectDataset(
        df=data_frame,
        genome_accessor=genome,
        df_name=df_name,  # Pass df_name if needed
        **kwargs  # Pass half_model_input_len
    )
    dataloader = DataLoader(
        snp_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
        collate_fn=collate_snp_data, pin_memory=torch.cuda.is_available()
    )

    # Initialize storage for effects for each margin
    # {margin: {'observed': [], 'predicted': []}}
    effects_data_by_margin: Dict[int, Dict[str, List[float]]] = {
        m: {'observed': [], 'predicted': []} for m in margins_list
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch_data in tqdm(dataloader, desc=f"Processing {model_name} (BS {batch_size}) for all margins", leave=False):
        if batch_data is None: continue

        seq_tensor = batch_data['ref_seq_batch'].to(device).transpose(1, 2)
        seq_mutated_tensor = batch_data['alt_seq_batch'].to(device).transpose(1, 2)

        observed_batch_np = batch_data['observed_batch'].numpy()
        cpg_rel_start_batch_np = batch_data['cpg_rel_start_batch']
        cpg_rel_end_batch_np = batch_data['cpg_rel_end_batch']
        cpg_number_batch_np = batch_data['cpg_number_batch']
        # original_indices_batch = batch_data['index_batch']

        try:
            with torch.no_grad():
                pred_test_output = model(seq_tensor)
                pred_test_mutated_output = model(seq_mutated_tensor)
                if model_post_process:
                    pred_test_output = model_post_process(pred_test_output)
                    pred_test_mutated_output = model_post_process(pred_test_mutated_output)
                pred_test = pred_test_output[0] if isinstance(pred_test_output, (list, tuple)) else pred_test_output
                pred_test_mutated = pred_test_mutated_output[0] if isinstance(pred_test_mutated_output, (list,
                                                                                                         tuple)) else pred_test_mutated_output

                if pred_test.dim() == 3 and pred_test.shape[1] == 1: pred_test = pred_test.squeeze(1)
                if pred_test_mutated.dim() == 3 and pred_test_mutated.shape[
                    1] == 1: pred_test_mutated = pred_test_mutated.squeeze(1)

                if pred_test.dim() < 2 or pred_test_mutated.dim() < 2:
                    raise ValueError(f"Model output tensor dimensions are incorrect: {pred_test.shape}")

                pred_len = pred_test.shape[1]

            # Process each item in the batch for ALL margins
            for j in range(len(observed_batch_np)):  # Iterate over items in batch
                item_cpg_rel_start = cpg_rel_start_batch_np[j]
                item_cpg_rel_end = cpg_rel_end_batch_np[j]
                item_observed_effect = observed_batch_np[j]
                item_cpg_number = cpg_number_batch_np[j]
                # original_idx = original_indices_batch[j]

                for margin_val in margins_list:
                    # Calculate check window for current margin
                    # Window is [CpG_start - margin, CpG_end + margin) relative to sequence start
                    # So, relative to prediction:
                    # (item_cpg_rel_start) is the 0-indexed start of CpG in the input sequence
                    # (item_cpg_rel_end) is the 0-indexed end of CpG (exclusive) in the input sequence
                    check_rel_start = item_cpg_rel_start - margin_val
                    check_rel_end = item_cpg_rel_end + margin_val  # Exclusive end

                    if not (0 <= check_rel_start < pred_len and \
                            0 < check_rel_end <= pred_len and \
                            check_rel_start < check_rel_end):
                        # print(f"Warning row {original_idx}, margin {margin_val}: Check window rel {check_rel_start}-{check_rel_end} out of bounds for pred_len {pred_len}. Skipping.")
                        continue

                    effect = pred_test_mutated[j, check_rel_start:check_rel_end] - \
                             pred_test[j, check_rel_start:check_rel_end]

                    effects_data_by_margin[margin_val]['predicted'].append(effect.sum().item()/(item_cpg_number * 1.0))
                    effects_data_by_margin[margin_val]['observed'].append(item_observed_effect)

        except Exception as e:
            # print(f"Error processing batch (indices ~{batch_data['index_batch'][0]}...): {e}")
            continue

    # --- Post-processing and Plotting for each margin ---
    results_for_model: Dict[int, Tuple[float, Image.Image]] = {}

    for margin_val in margins_list:
        if draw:
            fig, ax = plt.subplots(figsize=(8, 8))  # New figure for each margin

        model_prediction_list = effects_data_by_margin[margin_val]['predicted']
        effect_size_list = effects_data_by_margin[margin_val]['observed']

        if not model_prediction_list or not effect_size_list:
            # print(f"No valid data for margin {margin_val}. Cannot generate plot/correlation.")
            if draw:
                ax.text(0.5, 0.5, f"No data for margin {margin_val}", ha='center', va='center')
            corr = np.nan
        else:
            model_preds_np = np.array(model_prediction_list)
            observed_effects_np = np.array(effect_size_list)

            valid_indices = ~np.isnan(model_preds_np) & ~np.isnan(observed_effects_np)
            model_preds_no_nan = model_preds_np[valid_indices]
            observed_effects_no_nan = observed_effects_np[valid_indices]

            num_points = len(observed_effects_no_nan)
            corr = np.nan
            if num_points >= 2:
                if np.std(observed_effects_no_nan) > 1e-6 and np.std(model_preds_no_nan) > 1e-6:
                    corr = np.corrcoef(observed_effects_no_nan, model_preds_no_nan)[0, 1]
                # else: print(f"Warning margin {margin_val}: Zero std dev.")
            # elif num_points > 0: print(f"Warning margin {margin_val}: Only {num_points} points.")
            if draw:
                ax.scatter(observed_effects_no_nan, model_preds_no_nan, alpha=0.5, label=f'n={num_points}')
                ax.set_xlabel("Observed Methylation Change (ALT - REF)")
                ax.set_ylabel("Predicted Methylation Change (Sum over window)")
                ax.set_title(f'{model_name}, MARGIN={margin_val}\nCorrelation = {corr:.4f}')
                ax.grid(True)
            if num_points > 0:
                all_values = np.concatenate((observed_effects_no_nan, model_preds_no_nan))
                lims = [np.min(all_values), np.max(all_values)]
                padding = (lims[1] - lims[0]) * 0.05
                lims = [lims[0] - padding, lims[1] + padding]
                if lims[0] >= lims[1]: lims = [lims[0] - 1, lims[0] + 1]
                # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0) # y=x line
                # ax.set_xlim(lims)
                # ax.set_ylim(lims)
            if draw: ax.legend()
        if draw:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            pyplot_image = Image.open(buf)
            # plt.show() # Avoid showing plot in function
            plt.close(fig)

        results_for_model[margin_val] = (corr, None if not draw else pyplot_image)

    return results_for_model

def get_new_dist_dict():
    dict_dist = {
        "0-100": {'observed': [], 'predicted': []},
        "all": {'observed': [], 'predicted': []},
        "0-200": {'observed': [], 'predicted': []},
        "0-500": {'observed': [], 'predicted': []},
        "0-1000": {'observed': [], 'predicted': []},
        "0-2000": {'observed': [], 'predicted': []},
        "0-5000": {'observed': [], 'predicted': []},
        "0-10000": {'observed': [], 'predicted': []},
        "100-200": {'observed': [], 'predicted': []},
        "200-500": {'observed': [], 'predicted': []},
        "500-1000": {'observed': [], 'predicted': []},
        "1000-2000": {'observed': [], 'predicted': []},
        "2000-5000": {'observed': [], 'predicted': []},
        "5000-10000": {'observed': [], 'predicted': []},
        "10000+": {'observed': [], 'predicted': []},
        "0-20": {'observed': [], 'predicted': []},
        "0-21": {'observed': [], 'predicted': []},
        "0-40": {'observed': [], 'predicted': []},
        "0-41": {'observed': [], 'predicted': []},
        "0-75": {'observed': [], 'predicted': []},
        "0-150": {'observed': [], 'predicted': []},
    }
    return dict_dist

def check_meqtl_batched_multi_margin_multi_distance(
        model: nn.Module,
        genome: Genome,
        data_frame: pd.DataFrame,
        margins_list: List[int],  # Takes a list of margins
        model_name: str,
        batch_size: int = 32,
        model_post_process: Callable = None,
        df_name: str = None,
        **kwargs  # For half_model_input_len etc. for SNPEffectDataset
) -> Dict[int, Dict[str, float]]:
    """
    Checks SNP effects for multiple margins by running the model once per batch.
    Returns a dictionary: {margin: (correlation, plot_image)}
    """
    snp_dataset = SNPEffectDataset(
        df=data_frame,
        genome_accessor=genome,
        df_name=df_name,
        **kwargs  # Pass half_model_input_len
    )
    dataloader = DataLoader(
        snp_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
        collate_fn=collate_snp_data, pin_memory=torch.cuda.is_available()
    )

    # Initialize storage for effects for each margin
    # {margin: {'observed': [], 'predicted': []}}
    effects_data_by_margin_and_dist: Dict[int, Dict[str, Dict[str, List[float]]]] = {
    # margin->dist_str->{'observed': [], 'predicted': []}
        m: get_new_dist_dict() for m in margins_list
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for batch_data in tqdm(dataloader, desc=f"Processing {model_name} (BS {batch_size}) for all margins", leave=False):
        if batch_data is None: continue

        seq_tensor = batch_data['ref_seq_batch'].to(device).transpose(1, 2)
        seq_mutated_tensor = batch_data['alt_seq_batch'].to(device).transpose(1, 2)

        observed_batch_np = batch_data['observed_batch'].numpy()
        cpg_rel_start_batch_np = batch_data['cpg_rel_start_batch']
        cpg_rel_end_batch_np = batch_data['cpg_rel_end_batch']
        dists_np = batch_data['dists']
        cpg_number_batch_np = batch_data['cpg_number_batch']
        # original_indices_batch = batch_data['index_batch']

        try:
            with torch.no_grad():
                pred_test_output = model(seq_tensor)
                pred_test_mutated_output = model(seq_mutated_tensor)
                if model_post_process:
                    pred_test_output = model_post_process(pred_test_output)
                    pred_test_mutated_output = model_post_process(pred_test_mutated_output)
                pred_test = pred_test_output[0] if isinstance(pred_test_output, (list, tuple)) else pred_test_output
                pred_test_mutated = pred_test_mutated_output[0] if isinstance(pred_test_mutated_output, (list,
                                                                                                         tuple)) else pred_test_mutated_output

                if pred_test.dim() == 3 and pred_test.shape[1] == 1: pred_test = pred_test.squeeze(1)
                if pred_test_mutated.dim() == 3 and pred_test_mutated.shape[1] == 1:
                    pred_test_mutated = pred_test_mutated.squeeze(1)

                if pred_test.dim() < 2 or pred_test_mutated.dim() < 2:
                    raise ValueError(f"Model output tensor dimensions are incorrect: {pred_test.shape}")

                pred_len = pred_test.shape[1]

            # Process each item in the batch for ALL margins
            for j in range(len(observed_batch_np)):  # Iterate over items in batch
                item_cpg_rel_start = cpg_rel_start_batch_np[j]
                item_cpg_rel_end = cpg_rel_end_batch_np[j]
                item_observed_effect = observed_batch_np[j]
                item_cpg_number = cpg_number_batch_np[j]
                dist = dists_np[j]
                # original_idx = original_indices_batch[j]

                for margin_val in margins_list:
                    # Calculate check window for current margin
                    # Window is [CpG_start - margin, CpG_end + margin) relative to sequence start
                    # So, relative to prediction:
                    # (item_cpg_rel_start) is the 0-indexed start of CpG in the input sequence
                    # (item_cpg_rel_end) is the 0-indexed end of CpG (exclusive) in the input sequence
                    check_rel_start = item_cpg_rel_start - margin_val
                    check_rel_end = item_cpg_rel_end + margin_val  # Exclusive end

                    if not (0 <= check_rel_start < pred_len and \
                            0 < check_rel_end <= pred_len and \
                            check_rel_start < check_rel_end):
                        # print(f"Warning row {original_idx}, margin {margin_val}: Check window rel {check_rel_start}-{check_rel_end} out of bounds for pred_len {pred_len}. Skipping.")
                        continue

                    effect = ((pred_test_mutated[j, check_rel_start:check_rel_end] - \
                             pred_test[j, check_rel_start:check_rel_end]).sum().item()) / (1.0 * float(item_cpg_number))
                    current_margin_data = effects_data_by_margin_and_dist[margin_val]
                    current_margin_data['all']['predicted'].append(effect)
                    current_margin_data['all']['observed'].append(item_observed_effect)
                    if dist < 100:
                        current_margin_data['0-100']['predicted'].append(effect)
                        current_margin_data['0-100']['observed'].append(item_observed_effect)
                    elif 100 <= dist < 200:
                        current_margin_data['100-200']['predicted'].append(effect)
                        current_margin_data['100-200']['observed'].append(item_observed_effect)
                    elif 200 <= dist < 500:
                        current_margin_data['200-500']['predicted'].append(effect)
                        current_margin_data['200-500']['observed'].append(item_observed_effect)
                    elif 500 <= dist < 1000:
                        current_margin_data['500-1000']['predicted'].append(effect)
                        current_margin_data['500-1000']['observed'].append(item_observed_effect)
                    elif 1000 <= dist < 2000:
                        current_margin_data['1000-2000']['predicted'].append(effect)
                        current_margin_data['1000-2000']['observed'].append(item_observed_effect)
                    elif 2000 <= dist < 5000:
                        current_margin_data['2000-5000']['predicted'].append(effect)
                        current_margin_data['2000-5000']['observed'].append(item_observed_effect)
                    elif 5000 <= dist < 10000:
                        current_margin_data['5000-10000']['predicted'].append(effect)
                        current_margin_data['5000-10000']['observed'].append(item_observed_effect)
                    elif dist >= 10000:
                        current_margin_data['10000+']['predicted'].append(effect)
                        current_margin_data['10000+']['observed'].append(item_observed_effect)

                    # Populate cumulative bins (an item can fall into multiple cumulative bins)
                    # Note: '0-100' is already covered by the specific bin logic above if dist < 100.
                    # The keys "0-200", "0-500", etc., in dict_dist refer to cumulative ranges.
                    if dist < 200:  # For "0-200"
                        current_margin_data['0-200']['predicted'].append(effect)
                        current_margin_data['0-200']['observed'].append(item_observed_effect)
                    if dist < 500:  # For "0-500"
                        current_margin_data['0-500']['predicted'].append(effect)
                        current_margin_data['0-500']['observed'].append(item_observed_effect)
                    if dist < 1000:  # For "0-1000"
                        current_margin_data['0-1000']['predicted'].append(effect)
                        current_margin_data['0-1000']['observed'].append(item_observed_effect)
                    if dist < 2000:  # For "0-2000"
                        current_margin_data['0-2000']['predicted'].append(effect)
                        current_margin_data['0-2000']['observed'].append(item_observed_effect)
                    if dist < 5000:  # For "0-5000"
                        current_margin_data['0-5000']['predicted'].append(effect)
                        current_margin_data['0-5000']['observed'].append(item_observed_effect)
                    if dist < 10000:  # For "0-10000"
                        current_margin_data['0-10000']['predicted'].append(effect)
                        current_margin_data['0-10000']['observed'].append(item_observed_effect)
                    if dist <= 20:
                        current_margin_data['0-20']['predicted'].append(effect)
                        current_margin_data['0-20']['observed'].append(item_observed_effect)
                    if dist <= 21:
                        current_margin_data['0-21']['predicted'].append(effect)
                        current_margin_data['0-21']['observed'].append(item_observed_effect)
                    if dist <= 40:
                        current_margin_data['0-40']['predicted'].append(effect)
                        current_margin_data['0-40']['observed'].append(item_observed_effect)
                    if dist <= 41:
                        current_margin_data['0-41']['predicted'].append(effect)
                        current_margin_data['0-41']['observed'].append(item_observed_effect)
                    if dist <= 75:
                        current_margin_data['0-75']['predicted'].append(effect)
                        current_margin_data['0-75']['observed'].append(item_observed_effect)
                    if dist <= 150:
                        current_margin_data['0-150']['predicted'].append(effect)
                        current_margin_data['0-150']['observed'].append(item_observed_effect)

        except Exception as e:
            print(f"Error processing batch (indices ~{batch_data['index_batch'][0]}...): {e}")
            # continue

    # --- Post-processing and Plotting for each margin ---
    results_for_model: Dict[int, Dict[str, float]] = {}
    for margin_val in margins_list:
        results_for_model[margin_val] = {}
        for dist_str in get_new_dist_dict().keys():
            results_for_model[margin_val][dist_str] = np.nan
    for dist_str in get_new_dist_dict().keys():
        for margin_val in margins_list:
            model_prediction_list = effects_data_by_margin_and_dist[margin_val][dist_str]['predicted']
            effect_size_list = effects_data_by_margin_and_dist[margin_val][dist_str]['observed']

            if not model_prediction_list or not effect_size_list:
                corr = np.nan
            else:
                model_preds_np = np.array(model_prediction_list)
                observed_effects_np = np.array(effect_size_list)

                valid_indices = ~np.isnan(model_preds_np) & ~np.isnan(observed_effects_np)
                model_preds_no_nan = model_preds_np[valid_indices]
                observed_effects_no_nan = observed_effects_np[valid_indices]

                num_points = len(observed_effects_no_nan)
                corr = np.nan
                if num_points >= 2:
                    if np.std(observed_effects_no_nan) > 1e-6 and np.std(model_preds_no_nan) > 1e-6:
                        corr = np.corrcoef(observed_effects_no_nan, model_preds_no_nan)[0, 1]

            results_for_model[margin_val][dist_str] = corr

    return results_for_model

def check_meqtl_batched_multi_margin_multi_distance_multi_track_model(
        model: nn.Module,
        genome: Genome,
        data_frame: pd.DataFrame,
        margins_list: List[int],  # Takes a list of margins
        model_name: str,
        batch_size: int = 32,
        model_post_process: Callable = None,
        track_names: List[str] = None,
        df_name: str = None,
        **kwargs  # For half_model_input_len etc. for SNPEffectDataset
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Checks SNP effects for multiple margins by running the model once per batch.
    Returns a dictionary: {margin: (correlation, plot_image)}
    """
    assert track_names, "track_names must be provided for multi-track models."
    snp_dataset = SNPEffectDataset(
        df=data_frame,
        genome_accessor=genome,
        df_name=df_name,
        **kwargs  # Pass half_model_input_len
    )
    dataloader = DataLoader(
        snp_dataset, batch_size=batch_size, shuffle=False, num_workers=16,
        collate_fn=collate_snp_data, pin_memory=torch.cuda.is_available()
    )
    track_name2idx = [track_names.index(tn) for tn in track_names]
    # Initialize storage for effects for each margin
    # {margin: {'observed': [], 'predicted': []}}
    effects_data_by_margin_and_dist: Dict[str, Dict[int, Dict[str, Dict[str, List[float]]]]] = {tn: {
    # track_name->margin->dist_str->{'observed': [], 'predicted': []}
        m: get_new_dist_dict() for m in margins_list
    } for tn in track_names
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Processing {model_name} (BS {batch_size}) for all margins", leave=False):
            if batch_data is None: continue

            seq_tensor = batch_data['ref_seq_batch'].to(device).transpose(1, 2)
            seq_mutated_tensor = batch_data['alt_seq_batch'].to(device).transpose(1, 2)

            observed_batch_np = batch_data['observed_batch'].numpy()
            cpg_rel_start_batch_np = batch_data['cpg_rel_start_batch']
            cpg_rel_end_batch_np = batch_data['cpg_rel_end_batch']
            cpg_number_batch_np = batch_data['cpg_number_batch']
            dists_np = batch_data['dists']

            pred_test_output = model(seq_tensor)
            pred_test_mutated_output = model(seq_mutated_tensor)
            if model_post_process:
                pred_test_output = model_post_process(pred_test_output)
                pred_test_mutated_output = model_post_process(pred_test_mutated_output)
            pred_test = pred_test_output[0] if isinstance(pred_test_output, (list, tuple)) else pred_test_output
            pred_test_mutated = pred_test_mutated_output[0] if isinstance(pred_test_mutated_output, (list,
                                                                                                     tuple)) else pred_test_mutated_output

            for track_idx, tn in enumerate(track_names):
                track_index = track_name2idx[track_idx]
                pred_test_track = pred_test[:, track_index, :]  # Shape: (batch_size, seq_len)
                pred_test_mutated_track = pred_test_mutated[:, track_index, :]  # Shape: (batch_size, seq_len)
                if pred_test_track.dim() == 3 and pred_test_track.shape[1] == 1: pred_test_track = pred_test_track.squeeze(1)
                if pred_test_mutated_track.dim() == 3 and pred_test_mutated_track.shape[1] == 1:
                    pred_test_mutated_track = pred_test_mutated_track.squeeze(1)

                if pred_test_track.dim() < 2 or pred_test_mutated.dim() < 2:
                    raise ValueError(f"Model output tensor dimensions are incorrect: {pred_test_track.shape}")

                pred_len = pred_test_track.shape[1]

                # Process each item in the batch for ALL margins
                for j in range(len(observed_batch_np)):  # Iterate over items in batch
                    item_cpg_rel_start = cpg_rel_start_batch_np[j]
                    item_cpg_rel_end = cpg_rel_end_batch_np[j]
                    item_observed_effect = observed_batch_np[j]
                    item_cpg_number = cpg_number_batch_np[j]
                    dist = dists_np[j]
                    # original_idx = original_indices_batch[j]

                    for margin_val in margins_list:
                        # Calculate check window for current margin
                        # Window is [CpG_start - margin, CpG_end + margin) relative to sequence start
                        # So, relative to prediction:
                        # (item_cpg_rel_start) is the 0-indexed start of CpG in the input sequence
                        # (item_cpg_rel_end) is the 0-indexed end of CpG (exclusive) in the input sequence
                        check_rel_start = item_cpg_rel_start - margin_val
                        check_rel_end = item_cpg_rel_end + margin_val  # Exclusive end

                        if not (0 <= check_rel_start < pred_len and \
                                0 < check_rel_end <= pred_len and \
                                check_rel_start < check_rel_end):
                            # print(f"Warning row {original_idx}, margin {margin_val}: Check window rel {check_rel_start}-{check_rel_end} out of bounds for pred_len {pred_len}. Skipping.")
                            continue

                        effect = ((pred_test_mutated_track[j, check_rel_start:check_rel_end] - \
                                 pred_test_track[j, check_rel_start:check_rel_end]).sum().item()) / (1.0 * float(item_cpg_number))
                        current_margin_data = effects_data_by_margin_and_dist[tn][margin_val]
                        current_margin_data['all']['predicted'].append(effect)
                        current_margin_data['all']['observed'].append(item_observed_effect)
                        if dist < 100:
                            current_margin_data['0-100']['predicted'].append(effect)
                            current_margin_data['0-100']['observed'].append(item_observed_effect)
                        elif 100 <= dist < 200:
                            current_margin_data['100-200']['predicted'].append(effect)
                            current_margin_data['100-200']['observed'].append(item_observed_effect)
                        elif 200 <= dist < 500:
                            current_margin_data['200-500']['predicted'].append(effect)
                            current_margin_data['200-500']['observed'].append(item_observed_effect)
                        elif 500 <= dist < 1000:
                            current_margin_data['500-1000']['predicted'].append(effect)
                            current_margin_data['500-1000']['observed'].append(item_observed_effect)
                        elif 1000 <= dist < 2000:
                            current_margin_data['1000-2000']['predicted'].append(effect)
                            current_margin_data['1000-2000']['observed'].append(item_observed_effect)
                        elif 2000 <= dist < 5000:
                            current_margin_data['2000-5000']['predicted'].append(effect)
                            current_margin_data['2000-5000']['observed'].append(item_observed_effect)
                        elif 5000 <= dist < 10000:
                            current_margin_data['5000-10000']['predicted'].append(effect)
                            current_margin_data['5000-10000']['observed'].append(item_observed_effect)
                        elif dist >= 10000:
                            current_margin_data['10000+']['predicted'].append(effect)
                            current_margin_data['10000+']['observed'].append(item_observed_effect)

                        # Populate cumulative bins (an item can fall into multiple cumulative bins)
                        # Note: '0-100' is already covered by the specific bin logic above if dist < 100.
                        # The keys "0-200", "0-500", etc., in dict_dist refer to cumulative ranges.
                        if dist < 200:  # For "0-200"
                            current_margin_data['0-200']['predicted'].append(effect)
                            current_margin_data['0-200']['observed'].append(item_observed_effect)
                        if dist < 500:  # For "0-500"
                            current_margin_data['0-500']['predicted'].append(effect)
                            current_margin_data['0-500']['observed'].append(item_observed_effect)
                        if dist < 1000:  # For "0-1000"
                            current_margin_data['0-1000']['predicted'].append(effect)
                            current_margin_data['0-1000']['observed'].append(item_observed_effect)
                        if dist < 2000:  # For "0-2000"
                            current_margin_data['0-2000']['predicted'].append(effect)
                            current_margin_data['0-2000']['observed'].append(item_observed_effect)
                        if dist < 5000:  # For "0-5000"
                            current_margin_data['0-5000']['predicted'].append(effect)
                            current_margin_data['0-5000']['observed'].append(item_observed_effect)
                        if dist < 10000:  # For "0-10000"
                            current_margin_data['0-10000']['predicted'].append(effect)
                            current_margin_data['0-10000']['observed'].append(item_observed_effect)
                        if dist <= 20:
                            current_margin_data['0-20']['predicted'].append(effect)
                            current_margin_data['0-20']['observed'].append(item_observed_effect)
                        if dist <= 21:
                            current_margin_data['0-21']['predicted'].append(effect)
                            current_margin_data['0-21']['observed'].append(item_observed_effect)
                        if dist <= 40:
                            current_margin_data['0-40']['predicted'].append(effect)
                            current_margin_data['0-40']['observed'].append(item_observed_effect)
                        if dist <= 41:
                            current_margin_data['0-41']['predicted'].append(effect)
                            current_margin_data['0-41']['observed'].append(item_observed_effect)
                        if dist <= 75:
                            current_margin_data['0-75']['predicted'].append(effect)
                            current_margin_data['0-75']['observed'].append(item_observed_effect)
                        if dist <= 150:
                            current_margin_data['0-150']['predicted'].append(effect)
                            current_margin_data['0-150']['observed'].append(item_observed_effect)


        # --- Post-processing and Plotting for each margin ---
        # results_for_model: Dict[int, Dict[str, float]] = {} # margin -> dist_str -> corr
        results_for_model: Dict[str, Dict[int, Dict[str, float]]] = {tn: {} for tn in track_names}
        for tn in track_names:
            for margin_val in margins_list:
                results_for_model[tn][margin_val] = {}
                for dist_str in get_new_dist_dict().keys():
                    results_for_model[tn][margin_val][dist_str] = np.nan

            for dist_str in get_new_dist_dict().keys():
                for margin_val in margins_list:
                    model_prediction_list = effects_data_by_margin_and_dist[tn][margin_val][dist_str]['predicted']
                    effect_size_list = effects_data_by_margin_and_dist[tn][margin_val][dist_str]['observed']

                    if not model_prediction_list or not effect_size_list:
                        corr = np.nan
                    else:
                        model_preds_np = np.array(model_prediction_list)
                        observed_effects_np = np.array(effect_size_list)

                        valid_indices = ~np.isnan(model_preds_np) & ~np.isnan(observed_effects_np)
                        model_preds_no_nan = model_preds_np[valid_indices]
                        observed_effects_no_nan = observed_effects_np[valid_indices]

                        num_points = len(observed_effects_no_nan)
                        corr = np.nan
                        if num_points >= 2:
                            if np.std(observed_effects_no_nan) > 1e-6 and np.std(model_preds_no_nan) > 1e-6:
                                corr = np.corrcoef(observed_effects_no_nan, model_preds_no_nan)[0, 1]

                    results_for_model[tn][margin_val][dist_str] = corr

        return results_for_model



def split_by_distance(
        data_frame: pd.DataFrame,
        distance_bars: List[int],
        limit_left: True,   # if False, the first distance bar will be 0. TODO: IF
):
    assert limit_left, "Currently only support limit_left=True, which means the first distance bar is 0."
    data_frame['SNP_region_start'] = pd.to_numeric(data_frame['SNP_region_start'], errors='coerce')
    data_frame['CPG_region_start'] = pd.to_numeric(data_frame['CPG_region_start'], errors='coerce')
    data_frame['distance'] = abs(data_frame['SNP_region_start'] - data_frame['CPG_region_start'])
    data_frames = []
    for i in range(len(distance_bars)):
        if i == 0:
            condition = (data_frame['distance'] >= 0) & (data_frame['distance'] < distance_bars[i])
        else:
            if limit_left:
                condition = (data_frame['distance'] >= distance_bars[i - 1]) & (data_frame['distance'] < distance_bars[i])
            else:

                condition = data_frame['distance'] < distance_bars[i]
        filtered_df = data_frame[condition]

        data_frames.append(filtered_df)

    final_condition = data_frame['distance'] >= distance_bars[-1]
    final_df = data_frame[final_condition]
    data_frames.append(final_df)

    return data_frames

@loguru.logger.catch
def check_meqtl_batched_multi_margin_multi_dist_multi_frame(
        model: nn.Module,
        genome: Genome,
        data_frame_names: List[str],
        margins_list: List[int],  # Takes a list of margins
        model_name: str,
        batch_size: int = 32,
        model_post_process: Callable = None,
        **kwargs  # For half_model_input_len etc. for SNPEffectDataset
) -> Dict[str, Dict[int, Dict[str, float]]]:   # df_name-> margin -> distance_str -> correlation
    """
    Checks SNP effects for multiple margins by running the model once per batch for multiple data frames.
    Returns a dictionary where the key is the data frame name and the value is another dictionary
    with margins as keys and distance_str as sub-keys, containing correlation values.
    """
    results_for_all_frames: Dict[str, Dict[int, Dict[str, float]]] = {}

    # Iterate over each data frame
    for df_name in data_frame_names:
        df = df_dict.get(df_name)
        if df is None:
            logging.error(f"Data frame {df_name} not found in df_dict.")
            continue

        # Call the check_meqtl_batched_multi_margin function for each data frame
        results_for_model = check_meqtl_batched_multi_margin_multi_distance(
            model=model,
            genome=genome,
            data_frame=df,
            margins_list=margins_list,
            model_name=model_name,
            batch_size=batch_size,
            model_post_process=model_post_process,
            df_name=df_name,
            **kwargs  # Pass additional arguments
        )

        # Store results for this data frame in the dictionary
        results_for_all_frames[df_name] = results_for_model

    return results_for_all_frames


@loguru.logger.catch
def new_check_meqtl_batched_multi_margin_multi_dist_multi_frame(
        model: nn.Module,
        genome: Genome,
        data_frame_names: List[str],
        margins_list: List[int],  # Takes a list of margins
        model_name: str,
        batch_size: int = 32,
        model_post_process: Callable = None,
        track_names: List[str] = None,
        **kwargs  # For half_model_input_len etc. for SNPEffectDataset
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:   # df_name-> margin -> distance_str -> correlation
    """
    Checks SNP effects for multiple margins by running the model once per batch for multiple data frames.
    Returns a dictionary where the key is the data frame name and the value is another dictionary
    with margins as keys and distance_str as sub-keys, containing correlation values.
    """
    results_for_all_frames: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {tn : {} for tn in track_names}
    # eqtl_dataset_name->track_name->margin->dist_str->correlation
    # Iterate over each data frame
    for df_name in tqdm(data_frame_names):
        df = df_dict.get(df_name)
        if df is None:
            logging.error(f"Data frame {df_name} not found in df_dict.")
            continue
        results_for_model = check_meqtl_batched_multi_margin_multi_distance_multi_track_model(
            model=model,
            genome=genome,
            data_frame=df,
            margins_list=margins_list,
            model_name=model_name,
            batch_size=batch_size,
            model_post_process=model_post_process,
            track_names=track_names,
            **kwargs  # Pass additional arguments
        )
        results_for_all_frames[df_name] = results_for_model
    # make eqtl_dataset_name->track_name->margin->dist_str->correlation into track_name->eqtl_dataset_name->margin->dist_str->correlation
    new_results_for_all_frames: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {tn : {} for tn in track_names}
    for df_name, track_dict in results_for_all_frames.items():
        for track_name, margin_dict in track_dict.items():
            if track_name not in new_results_for_all_frames:
                new_results_for_all_frames[track_name] = {}
            new_results_for_all_frames[track_name][df_name] = margin_dict
    results_for_all_frames = new_results_for_all_frames
    return results_for_all_frames

def plot_correlation_heatmap_to_image(
    results: Dict[str, Dict[int, Tuple[float, PILImage.Image]]],
    title: str = "Correlation Heatmap: Data Frames vs. Margins",
    image_format: str = "jpg",  # 'png' or 'jpg'
    dpi: int = 150 # Dots per inch for the output image
) -> PILImage.Image:
    """
    Generates a heatmap of correlations and returns it as a PIL Image object.

    Args:
        results: The dictionary from check_meqtl_batched_multi_margin_multi_frame.
        title: The title for the heatmap.
        image_format: The format for the output image ('png', 'jpg').
        dpi: Resolution of the output image.

    Returns:
        PIL.Image.Image: The heatmap as a PIL Image object.
                         Returns a placeholder image if input is invalid.
    """
    if not results:
        print("Warning: No data provided to plot_correlation_heatmap_to_image.")
        img = PILImage.new('RGB', (400, 100), color = 'lightgrey')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()
        draw.text((10, 10), "No data to display for heatmap.", fill='black', font=font)
        return img

    data_frame_names = list(results.keys())
    if not data_frame_names: # Should be caught by `if not results` but good practice
        margins = []
    else:
        first_df_results = results[data_frame_names[0]]
        margins = sorted(list(first_df_results.keys()))

    if not margins:
        print("Warning: No margins found in the results.")
        img = PILImage.new('RGB', (400, 100), color = 'lightgrey')
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()
        draw.text((10, 10), "No margin data to display for heatmap.", fill='black', font=font)
        return img

    heatmap_data = []
    for df_name in data_frame_names:
        row_correlations = [results[df_name].get(margin, (float('nan'), None))[0] for margin in margins]
        heatmap_data.append(row_correlations)

    df_heatmap = pd.DataFrame(heatmap_data, index=data_frame_names, columns=margins)

    # Dynamically adjust figsize
    fig_width = max(8, 2 + len(margins) * 0.7)
    fig_height = max(6, 1 + len(data_frame_names) * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    sns.heatmap(
        df_heatmap,
        annot=True, fmt=".3f", cmap="Blues",
        linewidths=.5, cbar_kws={'label': 'Correlation Score'}, ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Margin", fontsize=12)
    ax.set_ylabel("Data Frame Name", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout(pad=1.2)

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format=image_format) # dpi is already set in subplots
    buf.seek(0)

    # Create a PIL Image from the BytesIO object
    pil_img = PILImage.open(buf)

    # Close the matplotlib figure to free up memory
    plt.close(fig)

    return pil_img

def parse_distance_bar_str(index_, distance_bar_li, limit_left=True):
    # e.g., distance_bar_li = [100 500 1000 2000 10000]
    # 0, distance_bar_li -> "0-100"
    # 1, distance_bar_li -> "100-500"
    # ...
    # 5, distance_bar_li -> "10000+"
    if index_ == 0:
        return f"0-{distance_bar_li[index_]}"
    elif limit_left:
        if index_ == len(distance_bar_li):
            return f"{distance_bar_li[-1]}+"
        else:
            return f"{distance_bar_li[index_ - 1]}-{distance_bar_li[index_]}"
    else:
        if index_ == len(distance_bar_li):
            return f"{distance_bar_li[-1]}+"
        else:
            return f"{distance_bar_li[index_ - 1]}-{distance_bar_li[index_]}"
