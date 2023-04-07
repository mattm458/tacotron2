import logging
from random import random
from typing import Dict, Optional, Tuple

import lightning as pl
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F

from model.decoder import Decoder
from model.encoder import Encoder
from model.modules_v2 import AlwaysDropout
from model.postnet import Postnet
from utils.hifi_gan import get_random_segment


class Tacotron2(pl.LightningModule):
    def __init__(
        self,
        num_chars: int,
        char_embedding_dim: int,
        encoder_kernel_size: int,
        num_mels: int,
        prenet_dim: int,
        att_rnn_dim: int,
        att_dim: int,
        rnn_hidden_dim: int,
        postnet_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding_dim = char_embedding_dim
        self.num_mels = num_mels
        self.att_rnn_dim = att_rnn_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.char_embedding_dim = char_embedding_dim

        # Tacotron 2 encoder
        self.encoder = Encoder(
            num_chars=num_chars,
            embedding_dim=char_embedding_dim,
            encoder_kernel_size=encoder_kernel_size,
            dropout=dropout,
        )

        # Prenet - a preprocessing step over the Mel spectrogram from the previous frame.
        # The network uses AlwaysDropout, which forces dropout to occur even during inference. This
        # method is adopted by the Tacotron 2 paper to introduce variation in the output speech.
        self.prenet = nn.Sequential(
            nn.Linear(num_mels, prenet_dim, bias=False),
            nn.ReLU(),
            AlwaysDropout(dropout),
            nn.Linear(prenet_dim, prenet_dim, bias=False),
            nn.ReLU(),
            AlwaysDropout(dropout),
        )

        # Additional encoder layer for attention. Done here since it applies to the entire
        # character input, and is only applied once before invoking the decoder
        self.att_encoder = nn.Linear(self.embedding_dim, att_dim, bias=False)

        # Tacotron 2 decoder
        self.decoder = Decoder(
            num_mels=num_mels,
            embedding_dim=self.embedding_dim,
            prenet_dim=prenet_dim,
            att_rnn_dim=att_rnn_dim,
            att_dim=att_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            dropout=dropout,
        )

        # Postnet layer. Done here since it applies to the entire Mel spectrogram output.
        self.postnet = Postnet(
            num_layers=5, num_mels=num_mels, postnet_dim=postnet_dim, dropout=dropout
        )

        self.tacotron_modules = [
            self.encoder,
            self.prenet,
            self.att_encoder,
            self.decoder,
            self.postnet,
        ]

    def init_hidden(self, encoded_len: int, batch_size: int, device: torch.device):
        """Generates initial hidden states, output tensors, and attention vectors.

        Args:
            encoded_len -- Length of the input character tensor
            batch_size -- Number of samples per batch
        """
        att_rnn_hidden = (
            torch.zeros(batch_size, self.att_rnn_dim, device=device),
            torch.zeros((batch_size, self.att_rnn_dim), device=device),
        )

        att_context = torch.zeros(batch_size, self.embedding_dim, device=device)
        att_weights = torch.zeros(batch_size, encoded_len, device=device)
        att_weights_cum = torch.zeros(batch_size, encoded_len, device=device)

        rnn_hidden = (
            torch.zeros(batch_size, self.rnn_hidden_dim, device=device),
            torch.zeros(batch_size, self.rnn_hidden_dim, device=device),
        )

        return (
            att_rnn_hidden,
            att_context,
            att_weights,
            att_weights_cum,
            rnn_hidden,
        )

    def forward(
        self,
        chars_idx: Tensor,
        chars_idx_len: Tensor,
        teacher_forcing: bool,
        mel_spectrogram: Optional[Tensor] = None,
        mel_spectrogram_len: Optional[Tensor] = None,
    ):
        if teacher_forcing:
            assert (
                mel_spectrogram is not None
            ), "Ground-truth Mel spectrogram is required for teacher forcing"
            assert (
                mel_spectrogram_len is not None
            ), "Ground-truth Mel spectrogram lengths are required for teacher forcing"

            logging.info(
                f"Tacotron 2: Teacher forcing enabled"
            )
        else:
            logging.info("Tacotron 2: Teacher forcing disabled")

        device = chars_idx.device

        # Encoding --------------------------------------------------------------------------------
        encoded = self.encoder(chars_idx, chars_idx_len)

        # Create a mask for the encoded characters
        encoded_mask = (
            torch.arange(chars_idx.shape[1], device=encoded.device)[None, :]
            < chars_idx_len[:, None]
        )

        # Transform the encoded characters for attention
        att_encoded = self.att_encoder(encoded)

        # Decoding --------------------------------------------------------------------------------
        batch_size = chars_idx.shape[0]

        # Get empty initial states
        (
            att_rnn_hidden,
            att_context,
            att_weights,
            att_weights_cum,
            rnn_hidden,
        ) = self.init_hidden(
            encoded_len=encoded.shape[1], batch_size=batch_size, device=encoded.device
        )

        max_len = mel_spectrogram.shape[1] if mel_spectrogram is not None else 2000

        if teacher_forcing:
            decoder_in = F.pad(mel_spectrogram, (0, 0, 1, 0))

            if teacher_forcing:
                decoder_in = self.prenet(decoder_in)

            decoder_in = [x.squeeze(1) for x in torch.split(decoder_in, 1, dim=1)]

            prev_mel = decoder_in[0]
        else:
            prev_mel = self.prenet(
                torch.zeros((batch_size, self.num_mels), device=device)
            )
            done = torch.zeros((batch_size), dtype=torch.bool, device=device)

        mels = []
        gates = []
        alignments = []

        if teacher_forcing:
            lengths = mel_spectrogram_len
        else:
            lengths = torch.zeros((batch_size), dtype=torch.long, device=device)

        # Iterate through all decoder inputs
        for i in range(0, max_len):
            # Run the decoder
            (
                mel_out,
                gate_out,
                att_rnn_hidden,
                att_context,
                att_weights,
                att_weights_cum,
                rnn_hidden,
            ) = self.decoder(
                prev_mel_prenet=prev_mel,
                att_rnn_hidden=att_rnn_hidden,
                att_context=att_context,
                att_weights=att_weights,
                att_weights_cum=att_weights_cum,
                rnn_hidden=rnn_hidden,
                encoded=encoded,
                att_encoded=att_encoded,
                encoded_mask=encoded_mask,
            )

            # Save decoder output
            mels.append(mel_out)
            gates.append(gate_out)
            alignments.append(att_weights)

            # Prepare for the next iteration
            if teacher_forcing:
                prev_mel = decoder_in[i + 1]
            else:
                done[gate_out.squeeze(-1) < 0.0] = True
                lengths[gate_out.squeeze(-1) >= 0.0] += 1
                if done.all():
                    break

                prev_mel = self.prenet(mel_out.detach())

        mels = torch.stack(mels, dim=1)
        gates = torch.stack(gates, dim=1)
        alignments = torch.stack(alignments, dim=1)

        # Run mel output through the postnet as a residual
        mels_post = self.postnet(mels.transpose(1, 2)).transpose(1, 2)
        mels_post = mels + mels_post

        mel_mask = (
            torch.arange(mels_post.shape[1], device=mels_post.device)[None, :]
            >= lengths[:, None]
        )

        mels = mels.swapaxes(1, 2).swapaxes(0, 1)
        mels_post = mels_post.swapaxes(1, 2).swapaxes(0, 1)
        gates = gates.swapaxes(1, 2).swapaxes(0, 1)
        mels = mels.masked_fill(mel_mask, 0.0).swapaxes(0, 1).swapaxes(1, 2)
        mels_post = mels_post.masked_fill(mel_mask, 0.0).swapaxes(0, 1).swapaxes(1, 2)
        gates = gates.masked_fill(mel_mask, -1000.0).swapaxes(0, 1).swapaxes(1, 2)

        return mels, mels_post, gates, alignments
