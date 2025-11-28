from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from basic_blocks import ConvBlock
from global_constants import track_39_names

from selene_sdk.sequences import Genome
from selene_util import GenomicSignalFeatures, RandomPositions, RandomPositionsSampler, SamplerDataLoader

class CSVGnomeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        genome: Genome,
        methy_data,
        randi=True,
        shift_range=500,
        sample_length=10000
    ):
        self.data = pd.read_csv(path)
        self.genome = genome
        self.methy_data = methy_data
        self.pos_list = self.data["pos"].tolist()
        self.chr_list = self.data["chr"].tolist()
        self.shift_range = shift_range
        self.sample_length = sample_length
        self.randi = randi

    def __len__(self):
        return len(self.pos_list)

    def __getitem__(self, idx):
        chr = self.chr_list[idx]
        pos = self.pos_list[idx]
        # add a random shift
        if self.randi:
            pos = pos + np.random.randint(-self.shift_range, self.shift_range)
        else:
            pos = pos + self.shift_range
        seq = self.genome.get(chr, pos, pos + self.sample_length)
        methy = self.methy_data.get(chr, pos, pos + self.sample_length)
        return seq, methy

class Melody(nn.Module):
    def __init__(self, args: Namespace=None, n_track=1, track_names=None):
        super(Melody, self).__init__()
        self.args = args
        self.n_track = n_track
        self.uplblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(4, 256, kernel_size=17, padding=8),
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Conv1d(256, 256, stride=4, kernel_size=17, padding=8),  # 2500
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Conv1d(256, 256, stride=5, kernel_size=17, padding=8),  # 500
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Conv1d(256, 256, stride=5, kernel_size=17, padding=8),  # 100
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Conv1d(256, 256, stride=4, kernel_size=17, padding=8),  # 25
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Conv1d(256, 256, stride=5, kernel_size=17, padding=8),  # 5
                nn.GroupNorm(1, 256)),

        ])

        self.upblocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

        ])

        self.downlblocks = nn.ModuleList([

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(256, 256, kernel_size=17, padding=8),
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(256, 256, kernel_size=17, padding=8),
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(256, 256, kernel_size=17, padding=8),
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Upsample(scale_factor=5),
                nn.Conv1d(256, 256, kernel_size=17, padding=8),
                nn.GroupNorm(1, 256)),

            nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv1d(256, 256, kernel_size=17, padding=8),
                nn.GroupNorm(1, 256)),

        ])

        self.downblocks = nn.ModuleList([

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True)),

            nn.Sequential(
                ConvBlock(256, 256, fused=True),
                ConvBlock(256, 256, fused=True))

        ])

        self.final = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.n_track, kernel_size=1),  # 0: is CG or not. 1: output value. 2. 1 minus output value
            # nn.Softplus() # now removed, should have softplus somewhere outside
        )
        cpg_cls_n = args.cpg_cls_n if args is not None else 7
        self.final_100_cg_count = nn.Sequential( # this head predicts CG count in each 100bp bin. every output bit corresponds to one bin
            nn.MaxPool1d(kernel_size=100, stride=100), # note the shape here
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, cpg_cls_n, kernel_size=1),
        )

        self.final_100_methy_avg = nn.Sequential( # this head predicts methylation average in each 100bp bin. every output bit corresponds to one bin
            nn.MaxPool1d(kernel_size=100, stride=100),  # note the shape here
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.n_track, kernel_size=1)
        )
        self.track_39_names = track_39_names if track_names is None else track_names

    def forward(self, x):
        """Forward propagation of a batch."""
        out = x
        encodings = []
        for i, lconv, conv in zip(np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out_final = self.final(out)
        pred_cg = self.final_100_cg_count(out)
        pred_methy_avg = self.final_100_methy_avg(out)
        return out_final, pred_cg, pred_methy_avg

    def get_output_with_cell_type__return_idx_and_sliced_tensor(self, cell_type_str, x):
        assert self.n_track == 39, "Only support 39 tracks for now."
        # cell_type_str may be buzzword or full name
        if cell_type_str in self.track_39_names:
            idx = self.track_39_names.index(cell_type_str)
            with torch.no_grad():
                pred_all, _, _ = self.forward(x)
            return idx, pred_all[:, idx, :]  # shape: [bsz, L]
        else:
            # try to find a similar name
            found_indices = []
            found_cell_track_names = []
            for i, name in enumerate(self.track_39_names):
                if cell_type_str in name or name in cell_type_str:
                    found_indices.append(i)
                    found_cell_track_names.append(name)
            if len(found_indices) == 1:
                idx = found_indices[0]
                with torch.no_grad():
                    pred_all, _, _ = self.forward(x)
                return idx, pred_all[:, idx, :] # shape: [bsz, L]
            elif len(found_indices) > 1:
                raise ValueError(f"Found multiple matches for {cell_type_str}: {found_cell_track_names}. Please specify more clearly.")
            elif len(found_indices) == 0:
                raise ValueError(f"Cannot find cell type {cell_type_str} in the track names. Please check the input.")
            else:
                raise ValueError(f"Unexpected error when searching for cell type {cell_type_str} in the track names.")