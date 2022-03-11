from torch import nn


class XavierLinear(nn.Module):
    """An extension of the PyTorch Linear class that applies Xavier weight initialization"""

    def __init__(self, in_dim, out_dim, bias=True, nonlinearity="linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(nonlinearity)
        )

    def forward(self, x):
        return self.linear_layer(x)


class XavierConv1d(nn.Module):
    """An extension of the PyTorch Conv1d class that applies Xavier weight initialization"""

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
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(nonlinearity)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class XavierConv2d(nn.Module):
    """An extension of the PyTorch Conv1d class that applies Xavier weight initialization"""

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
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(nonlinearity)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class AlwaysDropout(nn.Module):
    """An extension of the PyTorch Dropout class that applies dropout during inference"""

    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, X):
        self.dropout.training = True
        return self.dropout(X)
