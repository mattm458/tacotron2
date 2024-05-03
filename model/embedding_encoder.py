import torch
from torch import Tensor, jit, nn


def lengths_to_mask(lengths: Tensor, max_size: int) -> Tensor:
    mask = torch.arange(max_size, device=lengths.device)
    mask = mask.unsqueeze(0).repeat(len(lengths), 1)
    mask = mask >= lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)
    return mask


class Attention(jit.ScriptModule):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.context_dim = context_dim

        self.history = nn.Linear(history_in_dim, att_dim, bias=False)
        self.context = nn.Linear(context_dim, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)

    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        history_att = self.history(history)
        context_att = self.context(context).unsqueeze(1)
        score = self.v(torch.tanh(history_att + context_att))

        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)

        score = score.masked_fill(mask, 0.0)
        score_out = score.detach().clone()

        score = score.squeeze(-1).unsqueeze(1)
        att_applied = torch.bmm(score, history)
        att_applied = att_applied.squeeze(1)

        return att_applied, score_out


class EmbeddingEncoder(jit.ScriptModule):
    def __init__(
        self,
        embedding_dim: int,
        encoder_out_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        attention_dim: int,
        pack_sequence=True,
    ):
        super().__init__()

        lstm_out_dim = encoder_out_dim // 2

        self.encoder_out_dim = encoder_out_dim
        self.encoder_num_layers = encoder_num_layers
        self.pack_sequence = pack_sequence

        self.encoder = nn.GRU(
            embedding_dim,
            lstm_out_dim,
            bidirectional=True,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            batch_first=True,
        )
        self.encoder.flatten_parameters()

        self.attention = Attention(
            history_in_dim=encoder_out_dim,
            context_dim=encoder_out_dim * 2,
            att_dim=attention_dim,
        )

    @jit.script_method
    def forward(self, encoder_in: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = encoder_in.shape[0]

        if self.pack_sequence:
            encoder_in_packed = nn.utils.rnn.pack_padded_sequence(
                encoder_in,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            encoder_out_tmp, h = self.encoder(encoder_in_packed)

            encoder_out, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_out_tmp, batch_first=True
            )
        else:
            encoder_out, h = self.encoder(encoder_in)

        h = h.swapaxes(0, 1).reshape(batch_size, -1)

        return self.attention(
            history=encoder_out,
            context=h,
            mask=lengths_to_mask(lengths, encoder_out.shape[1]),
        )
