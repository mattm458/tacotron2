from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm


class ResBlock2(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int] = (1, 3)
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=int((kernel_size * dilation[0] - dilation[0]) / 2),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=int((kernel_size * dilation[1] - dilation[1]) / 2),
                ),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

    def weight_norm(self):
        for i in self.convs:
            weight_norm(i)

    def remove_weight_norm(self):
        for i in self.convs:
            remove_weight_norm(i)


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        resblock_kernel_sizes: List[int] = [3, 5, 7],
        resblock_dilation_sizes: List[Tuple[int, int]] = [(1, 2), (2, 6), (3, 12)],
    ):
        super().__init__()

        self.resblock_len = len(resblock_kernel_sizes)

        self.resblock = nn.ModuleList(
            [
                ResBlock2(channels=channels, kernel_size=k, dilation=d)
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.stack([r(x) for r in self.resblock]).sum(dim=0)
        x /= self.resblock_len

        return x

    def weight_norm(self):
        for i in self.resblock:
            i.weight_norm()

    def remove_weight_norm(self):
        for i in self.resblock:
            i.remove_weight_norm()
