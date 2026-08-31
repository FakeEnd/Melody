from argparse import Namespace

import torch
import numpy as np
import torch.nn as nn

class CellConditionedModulation(nn.Module):
    def __init__(self, cell_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cell_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim * 2)
        )

    def forward(self, x, cell_embed):
        """
        x: [B, C, L]
        cell_embed: [B, K, cell_dim]
        """
        B, C, L = x.shape
        pooled = cell_embed
        cond = self.mlp(pooled)
        scale, shift = cond.chunk(2, dim=1)
        scale = scale.view(B, C, 1)
        shift = shift.view(B, C, 1)

        return x * (1 + scale) + shift

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 9, 1, padding=4, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.GroupNorm(1, oup),
        )

    def forward(self, x):
        return x + self.conv(x)


class MelodyG(nn.Module):
    def __init__(self, args: Namespace, n_track=1, n_channels=4,):
        """
        Parameters
        ----------
        """
        super(MelodyG, self).__init__()
        torch.set_float32_matmul_precision('high')
        self.cell_mod = CellConditionedModulation(cell_dim=512, out_dim=256)
        # cell_embedding = np.load(args.cell_embedding_file)
        # self.cell_embedding_global = torch.tensor(cell_embedding, device='cuda', requires_grad=False)
        self.uplblocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_channels, 256, kernel_size=17, padding=8),
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
            nn.Conv1d(256, n_track, kernel_size=1),  # batch_size, 5, 10000
        )

        self.final_100_cg_count = nn.Sequential(
            nn.MaxPool1d(kernel_size=100, stride=100),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 7, kernel_size=1),  # batch_size, 7, 100
        )

        self.final_100_methy_avg = nn.Sequential(
            nn.MaxPool1d(kernel_size=100, stride=100),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(1, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x, cell_embedding=None, cg_count=False, methy_avg=False):
        """Forward propagation of a batch."""
        out = x
        encodings = []
        for i, lconv, conv in zip(np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        out = self.cell_mod(out, cell_embedding)

        for enc, lconv, conv in zip(reversed(encodings[:-1]), self.downlblocks, self.downblocks):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out_final = self.final(out)
        out_100_cg_count = self.final_100_cg_count(out)
        out_100_methy_avg = self.final_100_methy_avg(out)

        return out_final, out_100_cg_count, out_100_methy_avg


    def get_label(self, x):
        out = self.forward(x)
        label = torch.sigmoid(out)
        return label