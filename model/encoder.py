from torch import nn
from math import sqrt

from model.modules import XavierConv1d


class Encoder(nn.Module):
    """A class implementing a Tacotron2 encoder. It accepts a sequence of character indices and
    returns a Tensor of encoded character embeddings.
    """

    def __init__(self, num_chars, embedding_dim, encoder_kernel_size, dropout):
        """Create an Encoder object.

        Args:
            num_chars -- The number of characters used in the dataset
            embedding_dim -- The character embedding size
            encoder_kernel_size -- size of the character input convolving kernel
            dropout -- the probability of elements to be zeroed out where dropout is applied
        """
        super().__init__()

        # Character embeddings
        self.embedding = nn.Embedding(num_chars + 1, embedding_dim, padding_idx=0)
        std = sqrt(2.0 / (num_chars + embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # 3 character convolutions. The paper describes these as working like n-grams, mixing data
        # from embedding_kernel_size neighboring characters
        convolutions = []
        for _ in range(3):
            conv_layer = [
                XavierConv1d(
                    embedding_dim,
                    embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    nonlinearity="relu",
                ),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            convolutions.extend(conv_layer)
        self.convolutions = nn.Sequential(*convolutions)

        # Final encoding step: a bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, embedding_dim // 2, batch_first=True, bidirectional=True
        )

    def forward(self, char_idx, char_idx_len):
        embedded = self.embedding(char_idx)
        embedded = embedded.transpose(1, 2)

        conv = self.convolutions(embedded)
        conv = conv.transpose(1, 2)

        packed = nn.utils.rnn.pack_padded_sequence(
            conv, char_idx_len.cpu(), batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        encoded, _ = self.lstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)

        return encoded
