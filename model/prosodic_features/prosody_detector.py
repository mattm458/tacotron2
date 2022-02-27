from model.modules import XavierConv2d, XavierLinear
from torch import nn
from torch import functional as F
import torch
import pytorch_lightning as pl


class ProsodyPredictor(pl.LightningModule):
    def __init__(self, output_dim):
        super().__init__()

        convolutions = []
        for _ in range(5):
            conv_layer = [
                XavierConv2d(
                    1,
                    1,
                    kernel_size=(1, 5),
                    stride=1,
                    padding="same",
                    nonlinearity="relu",
                ),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            ]
            convolutions.extend(conv_layer)
        self.convolutions = nn.Sequential(*convolutions)

        self.lstm = nn.LSTM(80, 80 // 2, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

        self.linear = XavierLinear(80, output_dim)

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        conv = self.convolutions(mel_spectrogram.unsqueeze(1)).squeeze(1)

        packed = nn.utils.rnn.pack_padded_sequence(
            conv, mel_spectrogram_len.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h, c) = self.lstm(packed)
        h = h.transpose(1, 0).contiguous().view(mel_spectrogram.shape[0], -1)

        return self.linear(h)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        pred_features = self(
            tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
        )

        loss = F.mse_loss(pred_features, tts_metadata["features_log_norm"])

        return loss

    def validation_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        pred_features = self(
            tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
        )

        loss = F.mse_loss(pred_features, tts_metadata["features_log_norm"])

        return loss

    def predict_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        pred_features = self(
            tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
        )

        return pred_features, tts_metadata["features_log_norm"]

    def validation_step_end(self, outputs):
        self.log("val_loss", outputs.detach(), on_step=False, on_epoch=True)

    def training_step_end(self, outputs):
        self.log("training_loss", outputs.detach(), on_step=False, on_epoch=True)
