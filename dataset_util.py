import json
from typing import Iterable

import numpy as np
import torch

from selene_sdk.sequences import Genome # original selene_sdk

class RandomPositionsSamplerMultiCell:
    """
    RandomPositionsSampler is a class that samples data from random genomic positions
    from specified datasets, applying optional filters to exclude certain regions. It is designed
    to work with multiple datasets and a position sampler, and it supports multiprocessing through
    the SamplerDataLoader.

    Attributes
    ----------
    seed : int
        The seed for random number generation to ensure reproducibility.
    datasets : list
        A list of datasets from which samples are drawn.
    filters : list or None
        Optional filters to exclude certain genomic regions.
    position_sampler : object
        An object responsible for generating random positions.
    position_postprocess_funs : list or None
        Optional functions or lambda expressions to postprocess the sampled positions. Each function should take the
        chromosome, start position, and end position as input and return a tuple of the same.
        Example usage includes `[lambda chrm, start, end: random_shift(chrm, start, end, 100), None]`.
        to randomly shift the positions for selected datasets.

    Methods
    -------
    uninitialize()
        Uninitializes the position sampler and datasets, closing any open resources.

    filter(query)
        Applies filters to a given query to determine if it should be excluded.

    sample(batch_size=None)
        Samples a batch of data from the datasets, applying filters to exclude certain positions.
        If batch_size is specified, it returns a batch of that size.
    """

    def __init__(self,
                 datasets,
                 position_sampler,
                 position_postprocess_funs=None,
                 filters=None,
                 seed=436):

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        if isinstance(datasets, Iterable):
            self.datasets = datasets
        else:
            self.datasets = [datasets]

        self.filters = filters
        self.position_sampler = position_sampler
        self.position_postprocess_funs = position_postprocess_funs

    def uninitialize(self):
        self.position_sampler.uninitialize()
        for d in self.datasets:
            d.uninitialize()
        if self.filters is not None:
            for f in self.filters:
                f.uninitialize()

    def filter(self, query):
        if self.filters:
            if isinstance(self.filters, Iterable):
                for f in self.filters:
                    if f.get(*query):
                        return True
            else:
                return self.filters.get(*query)

    def sample(self, batch_size=None):
        if batch_size is not None:
            data_all = []
            for i in range(batch_size):
                while True:
                    try:
                        query = self.position_sampler.sample()
                        if self.filter(query):
                            continue
                        if self.position_postprocess_funs:
                            data = [d.get(*fun(*query))[None, ...] if fun is not None else d.get(*query)[None, ...] for
                                    d, fun in
                                    zip(self.datasets, self.position_postprocess_funs)]
                        else:
                            data = [d.get(*query) for d in self.datasets]
                            # data = [d.get(*query)[None, ...] for d in self.datasets]
                            # data = self.datasets[1].get(*query)[None, ...]
                        break
                    except Exception as e:
                        print(f"Exception occurred: {e}")
                tup = data[1]
                data = data[:1] + list(tup)
                data = [d[None, ...] for d in data]
                data_all.append(data)
            data_all = [np.concatenate(arrays) for arrays in zip(*data_all)]
            return tuple(data_all)
        else:
            while True:
                try:
                    query = self.position_sampler.sample()
                    if self.filter(query):
                        print('Position filtered:', query)
                        continue
                    if self.position_postprocess_funs:
                        data = [d.get(*fun(*query))[None, ...] if fun is not None else d.get(*query)[None, ...] for
                                d, fun in
                                zip(self.datasets, self.position_postprocess_funs)]
                    else:
                        data = [d.get(*query) for d in self.datasets]
                    break
                except Exception as e:
                    print(f"Exception occurred: {e}")

            return tuple(data)




from torch.utils.data import Dataset

class SequentialGenomeDataset(Dataset):
    """
    A PyTorch Dataset to iterate sequentially over genomic coordinates.

    This dataset generates fixed-size windows from specified chromosomes with a
    defined stride, ensuring no repetition and complete sequential coverage.

    Parameters
    ----------
    genome : Genome
        The genome object, expected to have a `get_chr_lens()` method that
        returns a list of (chr_name, length) tuples, and a `get(chrom, start, end)`
        method to retrieve sequence data.
    methy_data : GenomicDataset
        The methylation data object, also expected to have a `get(chrom, start, end)`
        method.
    chromosomes_to_scan : list of str
        A list of chromosome names to be scanned sequentially (e.g., ['chr8', 'chr9']).
    window_size : int
        The length of the genomic window to sample (e.g., 10000).
    stride : int
        The step size to move along the chromosome. For non-overlapping windows,
        set stride >= window_size.
    """

    def __init__(self, genome, methy_data, chromosomes_to_scan, window_size, stride):
        self.genome = genome
        self.methy_data = methy_data
        self.window_size = window_size
        self.stride = stride

        # A list to store all valid (chromosome, start_pos) tuples
        self.positions = []

        # Get chromosome lengths from the genome object
        chr_lengths = dict(self.genome.get_chr_lens())

        print("Pre-calculating all valid sequential positions...")
        for chrom in chromosomes_to_scan:
            if chrom not in chr_lengths:
                print(f"Warning: Chromosome '{chrom}' not found in genome. Skipping.")
                continue

            chrom_len = chr_lengths[chrom]

            # The last possible start position is `chrom_len - window_size`.
            # We iterate from 0 to this position with the given stride.
            for start_pos in range(0, chrom_len - self.window_size + 1, self.stride):
                self.positions.append((chrom, start_pos))

        print(f"Done. Found {len(self.positions)} total positions to sample.")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        A sample consists of the DNA sequence and methylation data for a
        specific genomic window.
        """
        # Get the chromosome and start position for the given index
        chrom, start_pos = self.positions[idx]
        end_pos = start_pos + self.window_size

        # Fetch data from the genome and methylation objects
        dna_sequence = self.genome.get(chrom, start_pos, end_pos)
        methylation_sequence = self.methy_data.get(chrom, start_pos, end_pos)

        # PyTorch models usually expect tensors
        # You might need to adjust the conversion based on your data types
        return torch.from_numpy(dna_sequence).float(), torch.from_numpy(methylation_sequence).float()



class scRNADataset:
    def __init__(self, embedding, cell_sample_size, random_cells=True):
        self.embedding = embedding
        self.cell_sample_size = cell_sample_size
        self.random_cells = random_cells
        self.len = len(self.embedding)

    def get(self, idx_cells=None):
        if idx_cells is None:
            if self.random_cells:
                if self.cell_sample_size > self.len:
                    embeddings = torch.from_numpy(np.vstack(self.embedding))
                else:
                    # idx_cells = np.arange(self.cell_sample_size)
                    idx_cells = np.random.randint(0, self.len, size=self.cell_sample_size).tolist()
                    # print(idx_cells)
                    embeddings = torch.from_numpy(np.vstack(self.embedding[idx_cells]))
        return embeddings



class SequentialGenomeDatasetWithEmbedding(Dataset):
    """
    A PyTorch Dataset to iterate sequentially over genomic coordinates.

    This dataset generates fixed-size windows from specified chromosomes with a
    defined stride, ensuring no repetition and complete sequential coverage.

    Parameters
    ----------
    genome : Genome
        The genome object, expected to have a `get_chr_lens()` method that
        returns a list of (chr_name, length) tuples, and a `get(chrom, start, end)`
        method to retrieve sequence data.
    methy_data : GenomicDataset
        The methylation data object, also expected to have a `get(chrom, start, end)`
        method.
    chromosomes_to_scan : list of str
        A list of chromosome names to be scanned sequentially (e.g., ['chr8', 'chr9']).
    window_size : int
        The length of the genomic window to sample (e.g., 10000).
    stride : int
        The step size to move along the chromosome. For non-overlapping windows,
        set stride >= window_size.
    """

    def __init__(self, genome: Genome, methy_data, cell_embedding, chromosomes_to_scan, window_size, stride):
        self.genome = genome
        self.methy_data = methy_data
        self.cell_embedding = np.mean(cell_embedding, axis=0)
        self.window_size = window_size
        self.stride = stride

        # A list to store all valid (chromosome, start_pos) tuples
        self.positions = []

        # Get chromosome lengths from the genome object
        chr_lengths = dict(self.genome.get_chr_lens())

        print("Pre-calculating all valid sequential positions...")
        for chrom in chromosomes_to_scan:
            if chrom not in chr_lengths:
                print(f"Warning: Chromosome '{chrom}' not found in genome. Skipping.")
                continue

            chrom_len = chr_lengths[chrom]

            # The last possible start position is `chrom_len - window_size`.
            # We iterate from 0 to this position with the given stride.
            for start_pos in range(0, chrom_len - self.window_size + 1, self.stride):
                self.positions.append((chrom, start_pos))

        print(f"Done. Found {len(self.positions)} total positions to sample.")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.positions)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        A sample consists of the DNA sequence and methylation data for a
        specific genomic window.
        """
        # Get the chromosome and start position for the given index
        chrom, start_pos = self.positions[idx]
        end_pos = start_pos + self.window_size

        # Fetch data from the genome and methylation objects
        dna_sequence = self.genome.get(chrom, start_pos, end_pos)
        methylation_sequence = self.methy_data.get(chrom, start_pos, end_pos)

        # PyTorch models usually expect tensors
        # You might need to adjust the conversion based on your data types
        return torch.from_numpy(dna_sequence).float(), torch.from_numpy(methylation_sequence).float(), torch.from_numpy(self.cell_embedding).float()

chromosome_lengths = {
    'chr1': 248956422,
    'chr2': 242193529,
    'chr3': 198295559,
    'chr4': 190214555,
    'chr5': 181538259,
    'chr6': 170805979,
    'chr7': 159345973,
    'chr8': 145138636,
    'chr9': 138394717,
    'chr10': 133797422,
    'chr11': 135086622,
    'chr12': 133275309,
    'chr13': 114364328,
    'chr14': 107043718,
    'chr15': 101991189,
    'chr16': 90338345,
    'chr17': 83257441,
    'chr18': 80373285,
    'chr19': 58617616,
    'chr20': 64444167,
    'chr21': 46709983,
    'chr22': 50818468,
    'chrX': 156040895,
    'chrY': 57227415
}

class CellRegionDataset(Dataset):
    def __init__(self, json_file, genome, methy_dataset_dict, embedding_dict, test_track, included_chrs=None):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)

        self.data_list = [item for item in self.data_list if item[0] not in test_track]

        self.genome = genome
        # self.methy_data = methy_data
        self.methy_dataset_dict = methy_dataset_dict
        self.embedding_dict = embedding_dict

        self.samples = []
        # for chr, positions in self.cg_dict.items():
        #     if included_chrs is None or chr in included_chrs:
        #         for pos in positions:
        #             self.samples.append((chr, pos))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        type, chr, region_start, region_end = self.data_list[idx]
        # length = region_end - region_start

        pos = (int(region_start) + int(region_end)) // 2
        start = pos - 10000 // 2
        end = pos + 10000 // 2

        length = region_end - region_start
        mask_start = 5000 - length // 2
        mask_end = 5000 + length // 2

        cell_region_mask = torch.zeros(10000, dtype=torch.long)
        cell_region_mask[mask_start:mask_end + 1] = 1

        if end <= chromosome_lengths[chr] and start >= 0:
            seq = self.genome.get(chr, start, end)  # shape: (length, 4

            methy_data = self.methy_dataset_dict[type]
            methy = methy_data.get(chr, start,end) # shape: (length,)
            embedding = self.embedding_dict[type]


            return seq, methy, np.mean(embedding, axis=0), cell_region_mask
