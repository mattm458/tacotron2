import torch
from torch import nn

from model.modules_legacy import XavierConv1d


class GatedConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        nonlinearity="linear",
    ):
        super().__init__()

        self.conv = XavierConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            nonlinearity=nonlinearity,
        )

        self.gate = XavierConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            nonlinearity="sigmoid",
        )

    def forward(self, x):
        # x = nn.functional.pad(x, self.dilation, 0)
        torch.mul(self.conv(x), torch.sigmoid(self.gate(x)))


class GatedResidualBlock(nn.Module):
    def __init__(
        self,
        output_width,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        nonlinearity="linear",
    ):
        super().__init__()

        self.output_width = output_width

        self.gatedconv = GatedConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            nonlinearity=nonlinearity,
        )
        self.skip = nn.XavierConv1d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            nonlinearity=nonlinearity,
        )

    def forward(self, x):
        skip = self.skip(self.gatedconv(x))
        residual = torch.add(skip, x)

        skip_cut = skip.shape[-1] - self.output_width
        skip = skip.narrow(-1, skip_cut, self.output_width)
        return residual, skip
