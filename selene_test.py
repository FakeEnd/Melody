import time
import os
import numpy as np
import tabix
import torch
import selene_sdk
import pyBigWig
from torch import nn
from scipy.special import softmax
from matplotlib import pyplot as plt
from selene_sdk.targets import Target
from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader

torch.set_default_tensor_type('torch.FloatTensor')

class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
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


tfeature = GenomicSignalFeatures(["./resources/agg.plus.bw.bedgraph.bw",
                                  "./resources/agg.encodecage.plus.v2.bedgraph.bw",
                                  "./resources/agg.encoderampage.plus.v2.bedgraph.bw",
                                  "./resources/agg.plus.grocap.bedgraph.sorted.merged.bw",
                                  "./resources/agg.plus.allprocap.bedgraph.sorted.merged.bw",
                                  "./resources/agg.minus.allprocap.bedgraph.sorted.merged.bw",
                                  "./resources/agg.minus.grocap.bedgraph.sorted.merged.bw",
                                  "./resources/agg.encoderampage.minus.v2.bedgraph.bw",
                                  "./resources/agg.encodecage.minus.v2.bedgraph.bw",
                                  "./resources/agg.minus.bw.bedgraph.bw"],
                                 ['cage_plus', 'encodecage_plus', 'encoderampage_plus', 'grocap_plus', 'procap_plus',
                                  'procap_minus', 'grocap_minus'
                                     , 'encoderampage_minus', 'encodecage_minus',
                                  'cage_minus'],
                                 (10000,))
genome = selene_sdk.sequences.Genome(
    input_path='./resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
    blacklist_regions='hg38'
)

sampler = RandomPositionsSampler(
    reference_sequence=genome,
    target=tfeature,
    features=[''],
    test_holdout=['chr8', 'chr9'],
    validation_holdout=['chr10'],
    sequence_length=10000,
    center_bin_to_predict=10000,
    position_resolution=1,
    random_shift=0,
    random_strand=False
)
sampler.mode = "train"
dataloader = SamplerDataLoader(sampler, num_workers=32, batch_size=32, seed=42)
