from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from model.attention_legacy import Attention
from model.modules_legacy import XavierLinear, XavierConv1d


class Decoder(nn.Module):
    """A class implementing a Tacotron2 decoder. It accepts a sequence of encoded input text, and
    returns four Tensors: an output Mel spectrogram frame, a gate output indicating whether the
    decoder has just produced the final Mel spectrogram frame, and the attention weights for this
    timestep.
    """
    def __init__(
        self,
        num_mels: int,
        embedding_dim: int,
        prenet_dim: int,
        att_rnn_dim: int,
        att_dim: int,
        rnn_hidden_dim: int,
        dropout: float,
        speech_feature_dim: Optional[int] = None,
    ):
        """Create a Decoder object.

        Args:
            num_mels -- number of Mel filterbanks to produce
            embedding_dim -- The character embedding size
            prenet_dim -- size of the Mel prenet output
            att_rnn_dim -- size of the hidden layer of the attention RNN
            att_dim -- size of hidden attention layers
            rnn_hidden_dim -- size of the hidden layer of the decoder RNN
            dropout -- the probability of elements to be zeroed out where dropout is applied
        """
        super().__init__()

        # Attention components - a LSTM cell and attention module
        self.att_rnn = nn.LSTMCell(prenet_dim + embedding_dim, att_rnn_dim)
        self.att_rnn_dropout = nn.Dropout(0.1)
        self.attention = Attention(
            attention_rnn_dim=att_rnn_dim,
            embedding_dim=embedding_dim,
            attention_dim=att_dim,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
        )

        speech_feature_dim = 0 if speech_feature_dim is None else speech_feature_dim
        self.speech_features = speech_feature_dim is not None

        # Decoder LSTM cell
        print(att_rnn_dim + embedding_dim + speech_feature_dim)
        self.lstm1 = nn.LSTMCell(
            att_rnn_dim + embedding_dim + speech_feature_dim, rnn_hidden_dim, bias=1
        )
        self.lstm_dropout = nn.Dropout(0.1)

        # Final layer producing Mel output
        self.mel_out = XavierLinear(rnn_hidden_dim + embedding_dim, num_mels)

        # Final layer producing gate output
        self.gate = XavierLinear(rnn_hidden_dim + embedding_dim, 1)

    def forward(
        self,
        prev_mel_prenet: Tensor,
        att_rnn_hidden: Tuple[Tensor, Tensor],
        att_context,
        att_weights,
        att_weights_cum,
        rnn_hidden: Tuple[Tensor, Tensor],
        encoded,
        att_encoded,
        encoded_mask,
        speech_features: Optional[Tensor] = None,
    ):
        """Perform a decoder forward pass.

        Args:
            prev_mel -- the previously generated Mel spectrogram frame
            prev_mel_prenet -- the previously generated Mel spectrogram frame processed by the
                               prenet
            att_rnn_hidden -- accumulated hidden and cell state for the attention RNN
            att_context -- accumulated attention context vectors
            att_weights -- attention weights from the previous Mel spectrogram frame
            att_weights_cum -- cumulative attention weights
            rnn_hidden -- accumulated hidden and cell state for both layers of the decoder RNN
            att_encoded - encoder output processed for attention
            dropout -- the probability of elements to be zeroed out where dropout is applied
            encoded_mask -- a mask for batched encoder input
        """
        # Attention -------------------------------------------------------------------------------
        # Attention RNN
        att_rnn_input = torch.concat([prev_mel_prenet, att_context], -1)
        att_h, att_c = self.att_rnn(att_rnn_input, att_rnn_hidden)
        att_h = self.att_rnn_dropout(att_h)

        # Attention module
        att_weights_cat = torch.concat(
            [att_weights.unsqueeze(1), att_weights_cum.unsqueeze(1)], 1
        )
        att_context, att_weights = self.attention(
            att_h, encoded, att_encoded, att_weights_cat, encoded_mask
        )

        # Save cumulative attention weights
        att_weights_cum += att_weights

        # Decoder ---------------------------------------------------------------------------------
        # Run attention output through the decoder RNN
        decoder_input = [att_h, att_context]
        if speech_features is not None:
            decoder_input.append(speech_features)

        decoder_input = torch.concat(decoder_input, -1)

        rnn_h, rnn_c = self.lstm1(decoder_input, rnn_hidden)
        rnn_h = self.lstm_dropout(rnn_h)

        rnn_out_att_context = torch.cat([rnn_h, att_context], dim=1)

        # Produce Mel spectrogram and gate outputs
        mel_out = self.mel_out(rnn_out_att_context)
        gate_out = self.gate(rnn_out_att_context)

        return (
            mel_out,
            gate_out,
            (att_h, att_c),
            att_context,
            att_weights,
            att_weights_cum,
            (rnn_h, rnn_c),
        )
