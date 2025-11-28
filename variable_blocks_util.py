import csv
import sys
from functools import cache
from time import sleep
from typing import Tuple, List
from loguru import logger


from selene_sdk.sequences import Genome
from _config import pdir

GENOME_PATH = pdir + '/data/fasta/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
var_blocks_path = pdir + '/data/sample.csv'
try:
    logger.info("Initializing Genome for validation (one-time setup)...")
    genome = Genome(input_path=GENOME_PATH, blacklist_regions='hg38')
    genome.get = genome.get_encoding_from_coords
    logger.info("Waiting for genome object to initialize...")
    sleep(2)
    logger.success("Genome reader initialized globally.")
except Exception as e:
    logger.critical(f"Fatal: Failed to initialize global Genome object. Error: {e}")
    sys.exit(1)
@cache
def get_variable_blocks() -> List[Tuple[str, int, int]]:
    """
    Load variable blocks from a CSV file and cache the results.
    Returns a list of tuples (chr_hg38, start_hg38, end_hg38).
    """
    logger.info(f"Loading variable blocks from {var_blocks_path}...")
    variable_blocks = []

    try:
        with open(var_blocks_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                chr_hg38 = row['chr_hg38']
                start_hg38 = int(row['start_hg38'])
                end_hg38 = int(row['end_hg38'])
                variable_blocks.append((chr_hg38, start_hg38, end_hg38))
        logger.success("Variable blocks loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load variable blocks: {e}")
        sys.exit(1)

    return variable_blocks

@cache
def get_scaled_blocks_from_variable_blocks(
    variable_blocks: Tuple[Tuple[str, int, int], ...],
    window_size: int
) -> List[Tuple[str, int, int]]:
    logger.info(f"Scaling {len(variable_blocks)} blocks to window size {window_size} and validating...")

    valid_scaled_blocks = []
    invalid_count = 0

    for chr_hg38, start_hg38, end_hg38 in variable_blocks:
        center = (start_hg38 + end_hg38) // 2
        start_scaled = center - window_size // 2
        end_scaled = start_scaled + window_size
        if start_scaled < 0:
            invalid_count += 1
            logger.debug(f"Invalid scaled block: {chr_hg38}:{start_hg38}-{end_hg38} resulted in negative start ({start_scaled}).")
            continue
        try:
            genome.get(chr_hg38, start_scaled, end_scaled)
            valid_scaled_blocks.append((chr_hg38, start_scaled, end_scaled))
        except Exception as e:
            invalid_count += 1
            logger.debug(f"Invalid scaled block {chr_hg38}:{start_scaled}-{end_scaled}. Reason: {e}")

    if invalid_count > 0:
        logger.warning(f"Discarded {invalid_count} scaled blocks that were invalid (negative start or out of bounds).")

    logger.success(f"Found {len(valid_scaled_blocks)} valid scaled blocks.")
    return valid_scaled_blocks

@cache
def get_scaled_blocks_from_variable_blocks_by_split_and_window(
    window: int,
    validation_holdout=('chr10',),
    test_holdout=('chr8', 'chr9'),
    blacklist_chrs=('chrY',),
    mode="train",
) -> List[Tuple[str, int, int]]:
    var_blocks_list = get_variable_blocks()
    var_blocks_tuple = tuple(var_blocks_list)
    all_scaled_blocks = get_scaled_blocks_from_variable_blocks(var_blocks_tuple, window_size=window)
    # print(all_scaled_blocks)
    validation_set = set(validation_holdout)
    test_set = set(test_holdout)
    blacklist_set = set(blacklist_chrs)
    print("Validation set:", validation_set)
    filtered_blocks = []
    # import pdb
    if mode == "train":
        forbidden_chrs = validation_set.union(test_set, blacklist_set)
        for block in all_scaled_blocks:
            if block[0] not in forbidden_chrs:
                filtered_blocks.append(block)
        return filtered_blocks

    elif mode == "validation":
        for block in all_scaled_blocks:
            if block[0] in validation_set:
                filtered_blocks.append(block)
        return filtered_blocks

    elif mode == "test":
        for block in all_scaled_blocks:
            if block[0] in test_set:
                filtered_blocks.append(block)
        return filtered_blocks
    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of 'train', 'validation', or 'test'.")