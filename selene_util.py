import numpy as np
import torch

from torch.utils.data import DataLoader, IterableDataset
from typing import Iterable, List, Optional, Sequence, Tuple

from selene_sdk.sequences import Genome
from selene_sdk.targets import Target
import pyBigWig
import tabix

class GenomicSignalFeatures(Target):
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True
        wigmat = np.vstack([c.values(chrom, start, end, numpy=True)
                            for c in self.data])
        if self.blacklists is not None:
            if self.replacement_indices is None:
                for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                    replacement_indices, np.fmax(
                                                                                                        int(s) - start,
                                                                                                        0): int(
                                                                                                        e) - start] * replacement_scaling_factor
        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat

    def get(self, *args, **kwargs):
        return self.get_feature_data(*args, **kwargs)


class RandomPositions:
    VALID_MODES = {"train", "validate", "test"}

    def __init__(self,
                 genome: Genome,
                 sample_length: int = 10000,
                 position_resolution: int = 1,
                 validation_holdout: Sequence[str] = ("chr10",),
                 test_holdout: Sequence[str] = ("chr8", "chr9"),
                 blacklist_chroms: Sequence[str] = ("chrY",),
                 mode: str = "train"):

        self.genome = genome
        self.sample_length = int(sample_length)
        self.position_resolution = int(position_resolution)

        self.validation_holdout = set(validation_holdout)
        self.test_holdout = set(test_holdout)
        self.blacklist_chroms = set(blacklist_chroms)

        half = self.sample_length // 2
        self._start_radius = half
        self._end_radius = self.sample_length - half

        chrom_lens = list(self.genome.get_chr_lens())
        self._all_chroms = [c for c, _ in chrom_lens]
        self._all_lengths = np.array([l for _, l in chrom_lens], dtype=np.int64)

        self._mode = None
        self._active_chroms: List[str] = []
        self._active_lengths: np.ndarray = np.array([], dtype=np.int64)
        self._chrom_sampling_probs: np.ndarray = np.array([], dtype=np.float64)

        self.set_mode(mode)

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str):
        mode = str(mode).lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self._mode = mode
        active = []

        for chrom, length in zip(self._all_chroms, self._all_lengths):
            if chrom in self.blacklist_chroms:
                continue

            if mode == "train":

                if chrom in self.validation_holdout:
                    continue
                if chrom in self.test_holdout:
                    continue
            elif mode == "validate":

                if chrom not in self.validation_holdout:
                    continue
            elif mode == "test":

                if chrom not in self.test_holdout:
                    continue

            active.append((chrom, length))

        if not active:
            raise RuntimeError(
                f"No chromosomes are eligible for mode={mode}. "
                f"Check your holdout and blacklist settings."
            )

        self._active_chroms = [c for c, _ in active]
        self._active_lengths = np.array([l for _, l in active], dtype=np.int64)

        probs = self._active_lengths.astype(np.float64)
        probs /= probs.sum()
        self._chrom_sampling_probs = probs

    def _choose_chrom_and_center(self) -> Tuple[str, int]:

        idx = np.random.choice(len(self._active_chroms), p=self._chrom_sampling_probs)
        chrom = self._active_chroms[idx]
        chrom_len = int(self._active_lengths[idx])

        start_min = self._start_radius
        start_max = chrom_len - self._end_radius
        if start_max <= start_min:
            raise RuntimeError(
                f"Chromosome {chrom} (len={chrom_len}) is too short for "
                f"sample_length={self.sample_length}"
            )

        center = np.random.randint(start_min, start_max)
        return chrom, center

    def sample(self) -> Tuple[str, int, int]:

        chrom, center = self._choose_chrom_and_center()

        start = center - self._start_radius
        end = start + self.sample_length

        if self.position_resolution > 1:
            start = (start // self.position_resolution) * self.position_resolution
            end = start + self.sample_length

        return chrom, start, end


class RandomPositionsSampler:

    def __init__(self,
                 reference_sequence,
                 target: Optional[Target] = None,
                 features: Optional[Sequence[str]] = None,
                 sequence_length: int = 10000,
                 position_resolution: int = 1,
                 validation_holdout: Sequence[str] = ("chr10",),
                 test_holdout: Sequence[str] = ("chr8", "chr9"),
                 blacklist_chroms: Sequence[str] = ("chrY",),
                 random_shift: int = 0,
                 random_strand: bool = False,
                 mode: str = "train",
                 seed: int = 436):

        if isinstance(reference_sequence, (list, tuple)) and len(reference_sequence) == 2:
            if isinstance(target, RandomPositions):
                self.reference_sequence = reference_sequence[0]
                self.target = reference_sequence[1]
                self._position_sampler = target
            else:
                self.reference_sequence = reference_sequence[0]
                self.target = reference_sequence[1]
        else:
            self.reference_sequence = reference_sequence
            self.target = target

        self.features = list(features) if features is not None else None

        self.sequence_length = int(sequence_length)
        self.position_resolution = int(position_resolution)
        self.random_shift = int(random_shift)
        self.random_strand = bool(random_strand)

        np.random.seed(seed)
        torch.manual_seed(seed)

        if not hasattr(self, '_position_sampler'):
            self._position_sampler = RandomPositions(
                genome=self.reference_sequence,
                sample_length=sequence_length,
                position_resolution=position_resolution,
                validation_holdout=validation_holdout,
                test_holdout=test_holdout,
                blacklist_chroms=blacklist_chroms,
                mode=mode,
            )

    @property
    def mode(self) -> str:
        return self._position_sampler.mode

    @mode.setter
    def mode(self, value: str):
        self._position_sampler.set_mode(value)

    def _maybe_shift(self, chrom: str, start: int, end: int) -> Tuple[str, int, int]:
        if self.random_shift <= 0:
            return chrom, start, end

        max_shift = self.random_shift
        shift = np.random.randint(-max_shift, max_shift + 1)

        chrom_len = dict(self.reference_sequence.get_chr_lens())[chrom]
        new_start = start + shift
        new_end = end + shift

        if new_start < 0:
            new_start = 0
            new_end = new_start + (end - start)
        if new_end > chrom_len:
            new_end = chrom_len
            new_start = new_end - (end - start)

        return chrom, new_start, new_end

    def _pick_strand(self) -> str:
        if not self.random_strand:
            return "+"
        return "+" if np.random.rand() < 0.5 else "-"

    def _single_sample(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        chrom, start, end = self._position_sampler.sample()
        chrom, start, end = self._maybe_shift(chrom, start, end)
        strand = self._pick_strand()

        seq = self.reference_sequence.get_encoding_from_coords(
            chrom,
            start,
            end,
            strand
        )
        seq = np.asarray(seq, dtype=np.float32)

        if self.target is None:
            return seq, None

        y = self.target.get_feature_data(chrom, start, end)
        y = np.asarray(y, dtype=np.float32)

        return seq, y

    def sample(self,
               batch_size: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        sequences = []
        targets = [] if self.target is not None else None

        for _ in range(batch_size):
            seq, y = self._single_sample()
            sequences.append(seq)
            if targets is not None:
                targets.append(y)

        sequences = np.stack(sequences, axis=0)

        if targets is None:
            return sequences, None

        targets = np.stack(targets, axis=0)
        return sequences, targets


class _SamplerIterableDataset(IterableDataset):

    def __init__(self, sampler: RandomPositionsSampler, batch_size: int):
        super().__init__()
        self.sampler = sampler
        self.batch_size = int(batch_size)

    def __iter__(self):
        while True:
            yield self.sampler.sample(batch_size=self.batch_size)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + worker_id)


class SamplerDataLoader(DataLoader):

    def __init__(self,
                 sampler: RandomPositionsSampler,
                 num_workers: int = 1,
                 batch_size: int = 1,
                 seed: int = 436,
                 **kwargs):
        np.random.seed(seed)
        torch.manual_seed(seed)

        dataset = _SamplerIterableDataset(sampler, batch_size=batch_size)

        super().__init__(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            **kwargs,
        )
