from typing import List, Optional, Tuple

import lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model.decoder import Decoder
from model.encoder import Encoder
from model.modules_v2 import AlwaysDropout
from model.postnet import Postnet


class Tacotron2(nn.Module):
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
        speaker_tokens: bool = False,
        num_speakers: int = 1,
        controls: bool = False,
        controls_dim: int = 0,
    ):
        super().__init__()

        self.embedding_dim = char_embedding_dim
        self.num_mels = num_mels
        self.att_rnn_dim = att_rnn_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.char_embedding_dim = char_embedding_dim

        self.controls = controls
        self.controls_dim = controls_dim
        if self.controls:
            print(f"Tacotron2: Controls enabled with a size of {controls_dim}")
        else:
            print(f"Tacotron2: Controls disabled")

        self.speaker_tokens = speaker_tokens
        assert not speaker_tokens or (
            speaker_tokens and num_speakers is not None
        ), "If speaker tokens are enabled, you must give a num_speakers!"
        if self.speaker_tokens:
            print(f"Tacotron2: Speaker tokens enabled with {num_speakers} speakers")
        else:
            print("Tacotron2: Speaker tokens disabled")

        if speaker_tokens:
            self.speaker_embedding = nn.Embedding(
                num_embeddings=num_speakers, embedding_dim=char_embedding_dim
            )
            self.speaker_embedding.weight.data.normal_(mean=0, std=0.5)

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
            extra_decoder_in_dim=controls_dim,
        )

        # Postnet layer. Done here since it applies to the entire Mel spectrogram output.
        self.postnet = Postnet(
            num_layers=5, num_mels=num_mels, postnet_dim=postnet_dim, dropout=dropout
        )

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
        speaker_id: Optional[Tensor] = None,
        controls: Optional[Tensor] = None,
        max_len_override: Optional[int] = None,
    ):
        if teacher_forcing:
            assert (
                mel_spectrogram is not None
            ), "Ground-truth Mel spectrogram is required for teacher forcing"
            assert (
                mel_spectrogram_len is not None
            ), "Ground-truth Mel spectrogram lengths are required for teacher forcing"

        assert not self.speaker_tokens or (
            self.speaker_tokens and speaker_id is not None
        ), "speaker_id tensor required when speaker tokens are active!"

        assert not self.controls or (
            self.controls and controls is not None
        ), "Controls are enabled, but no control vector was passed to the model!"

        assert self.controls or (
            not self.controls and not controls
        ), "Controls are disabled, but a control vector was passed to the model!"

        device = chars_idx.device
        longest_chars = chars_idx.shape[1]

        speaker_token: Optional[Tensor] = None
        if self.speaker_tokens:
            speaker_token = F.tanh(self.speaker_embedding(speaker_id))

        # Encoding --------------------------------------------------------------------------------
        encoded = self.encoder(chars_idx, chars_idx_len)

        # Create a mask for the encoded characters
        encoded_mask = (
            torch.arange(longest_chars, device=device)[None, :]
            >= chars_idx_len[:, None]
        )

        if speaker_token is not None:
            encoded += speaker_token.unsqueeze(1)

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

        if max_len_override is not None:
            max_len = max_len_override

        if max_len_override is None and mel_spectrogram is None:
            raise Exception(
                "If Mel spectrogram is not given, max_len_override is required!"
            )
        elif max_len_override is None and mel_spectrogram is not None:
            max_len = mel_spectrogram.shape[1]

        if teacher_forcing:
            decoder_in = F.pad(mel_spectrogram, (0, 0, 1, 0))

            decoder_in = self.prenet(decoder_in)
            decoder_in = [x.squeeze(1) for x in torch.split(decoder_in, 1, dim=1)]

            prev_mel = decoder_in[0]
        else:
            prenet_in = torch.zeros((batch_size, self.num_mels), device=device)
            prev_mel = self.prenet(prenet_in)
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
            args = {}

            extra_decoder_in: List[Tensor] = []

            if self.controls and controls is not None:
                extra_decoder_in.append(controls)

            args["extra_decoder_in"] = None
            if len(extra_decoder_in):
                args["extra_decoder_in"] = torch.concat(extra_decoder_in, dim=-1)

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
                **args,
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

                prenet_in = mel_out.detach()
                prev_mel = self.prenet(prenet_in)

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
