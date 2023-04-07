from torch import nn
from model.modules_legacy import XavierConv1d


class Postnet(nn.Module):
    def __init__(self, num_layers, num_mels, postnet_dim, dropout):
        super().__init__()

        postnet_convs = [
            XavierConv1d(
                num_mels,
                postnet_dim,
                5,
                bias=False,
                padding="same",
                nonlinearity="tanh",
            ),
            nn.BatchNorm1d(postnet_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        ]
        for i in range(num_layers - 2):
            postnet_convs.extend(
                [
                    XavierConv1d(
                        postnet_dim,
                        postnet_dim,
                        5,
                        bias=False,
                        padding="same",
                        nonlinearity="tanh",
                    ),
                    nn.BatchNorm1d(postnet_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                ]
            )
        postnet_convs.extend(
            [
                XavierConv1d(
                    postnet_dim,
                    num_mels,
                    5,
                    bias=False,
                    padding="same",
                ),
                nn.BatchNorm1d(num_mels),
                nn.Dropout(dropout),
            ]
        )

        self.postnet = nn.Sequential(*postnet_convs)

    def forward(self, X):
        return self.postnet(X)
