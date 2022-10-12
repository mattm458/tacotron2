from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from model.resblock import MultiReceptiveFieldFusion


class Generator(nn.Module):
    def __init__(
        self,
        resblock_kernel_sizes: List[int] = [3, 5, 7],
        resblock_dilation_sizes: List[Tuple[int, int]] = [(1, 2), (2, 6), (3, 12)],
    ):
        super().__init__()

        prenet = torch.nn.Conv1d(
            in_channels=80, out_channels=256, kernel_size=7, stride=1, padding=3
        )

        upsample1 = torch.nn.ConvTranspose1d(
            in_channels=256,
            out_channels=128,
            kernel_size=16,
            stride=8,
            padding=(16 - 8) // 2,
        )

        mrf1 = MultiReceptiveFieldFusion(
            128, resblock_kernel_sizes, resblock_dilation_sizes
        )

        upsample2 = torch.nn.ConvTranspose1d(
            in_channels=128,
            out_channels=64,
            kernel_size=16,
            stride=8,
            padding=(16 - 8) // 2,
        )

        mrf2 = MultiReceptiveFieldFusion(
            64, resblock_kernel_sizes, resblock_dilation_sizes
        )

        upsample3 = torch.nn.ConvTranspose1d(
            in_channels=64,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=(8 - 4) // 2,
        )

        mrf3 = MultiReceptiveFieldFusion(
            32, resblock_kernel_sizes, resblock_dilation_sizes
        )

        postnet = torch.nn.Conv1d(
            in_channels=32, out_channels=1, kernel_size=7, stride=1, padding=3
        )

        self.weight_norms = [prenet, upsample1, upsample2, upsample3, postnet]
        self.weight_norm_calls = [mrf1, mrf2, mrf3]

        self.generator = nn.Sequential(
            prenet,
            nn.LeakyReLU(0.1),
            upsample1,
            mrf1,
            nn.LeakyReLU(0.1),
            # Upsample 2
            upsample2,
            mrf2,
            nn.LeakyReLU(0.1),
            # Upsample 3
            upsample3,
            mrf3,
            nn.LeakyReLU(0.1),
            postnet,
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.swapaxes(1, 2)
        return self.generator(x)

    def weight_norm(self):
        for i in self.weight_norms:
            weight_norm(i)
        for i in self.weight_norm_calls:
            i.weight_norm()

    def remove_weight_norm(self):
        for i in self.weight_norms:
            remove_weight_norm(i)
        for i in self.weight_norm_calls:
            i.remove_weight_norm()
