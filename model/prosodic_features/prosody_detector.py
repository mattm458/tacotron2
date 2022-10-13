from model.modules import XavierConv2d, XavierLinear
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_lightning as pl
import torchaudio
import torchmetrics


class ProsodyPredictorV2(nn.Module):
    def __init__(self, use_deltas=True, use_lstm=False, rnn_layers=1, rnn_dropout=0.0):
        super().__init__()

        self.use_deltas = use_deltas

        self.delta = torchaudio.transforms.ComputeDeltas()

        self.conv = nn.Sequential(
            nn.Conv2d(3 if use_deltas else 1, 128, (5, 3), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(128, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (5, 3), (1, 1), padding=(2, 1)),
            nn.LeakyReLU(),
        )

        self.pre_rnn = nn.Sequential(nn.Linear(10 * 256, 768), nn.LeakyReLU())

        if use_lstm:
            self.rnn = nn.LSTM(
                768,
                128,
                batch_first=True,
                bidirectional=True,
                num_layers=rnn_layers,
                dropout=rnn_dropout,
            )
        else:
            self.rnn = nn.GRU(
                768,
                128,
                batch_first=True,
                bidirectional=True,
                num_layers=rnn_layers,
                dropout=rnn_dropout,
            )

        self.frame_weight = nn.Linear(256, 1)

        self.features_out = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, 7), nn.Tanh()
        )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        batch_size = mel_spectrogram.shape[0]

        mel_spectrogram = mel_spectrogram.swapaxes(1, 2)
        if mel_spectrogram.shape[2] % 2 == 1:
            mel_spectrogram = torch.cat(
                [
                    mel_spectrogram,
                    torch.zeros(
                        (mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1),
                        device=mel_spectrogram.device,
                    ),
                ],
                2,
            )

        if self.use_deltas:
            d1 = self.delta(mel_spectrogram)
            d2 = self.delta(d1)

            x = torch.cat(
                [mel_spectrogram.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)], dim=1
            ).swapaxes(2, 3)
        else:
            x = mel_spectrogram.unsqueeze(1).swapaxes(2, 3)

        output = self.conv(x)
        output = output.permute(0, 2, 3, 1).reshape(
            mel_spectrogram.shape[0], mel_spectrogram.shape[2], 256 * 10
        )

        output_low = output

        output = self.pre_rnn(output)

        rnn_input = nn.utils.rnn.pack_padded_sequence(
            output, mel_spectrogram_len.cpu(), batch_first=True, enforce_sorted=False
        )
        output, h = self.rnn(rnn_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output_middle = output

        weights = self.frame_weight(output).squeeze(-1)

        mask = torch.arange(mel_spectrogram_len.max(), device=mel_spectrogram.device)
        mask = mask.repeat(batch_size, 1)
        mask = mask < mel_spectrogram_len.unsqueeze(1)

        weights = torch.masked_fill(weights, ~mask, -float("inf"))
        weights = torch.softmax(weights, dim=1)

        output_high = torch.bmm(weights.unsqueeze(1), output).squeeze(1)

        return (
            torch.tanh(self.features_out(output_high)),
            output_low,
            output_middle,
            output_high,
        )


class ProsodyPredictorLightning(pl.LightningModule):
    def __init__(
        self,
        use_deltas=True,
        use_lstm=False,
        rnn_layers=1,
        rnn_dropout=0.0,
        features=[
            "pitch_mean_norm_clip",
            "pitch_range_norm_clip",
            "intensity_mean_norm_clip",
            "jitter_norm_clip",
            "shimmer_norm_clip",
            "nhr_norm_clip",
            "rate_norm_clip",
        ],
        lr=0.00001,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.features = features
        self.lr = lr

        self.prosody_predictor = ProsodyPredictorV2(
            use_deltas=use_deltas,
            use_lstm=use_lstm,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
        )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        out, _, _, _ = self.prosody_predictor(mel_spectrogram, mel_spectrogram_len)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        pred_features = self(
            tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
        )

        return pred_features, tts_metadata["features"]

    def validation_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        pred_features = self(
            tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
        )

        loss = F.mse_loss(pred_features, tts_metadata["features"])
        self.log("val_loss", loss, on_epoch=True, on_step=False)

        ccc = torchmetrics.functional.concordance_corrcoef(
            pred_features, tts_metadata["features"]
        )
        for feature, p in zip(self.features, ccc):
            self.log(f"val_{feature}", p, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        pred_features = self(
            tts_data["mel_spectrogram"],
            tts_metadata["mel_spectrogram_len"],
        )

        loss = F.mse_loss(pred_features, tts_metadata["features"])
        self.log("train_loss", loss.detach(), on_epoch=True, on_step=True)

        ccc = torchmetrics.functional.concordance_corrcoef(
            pred_features.detach(), tts_metadata["features"]
        )
        for feature, p in zip(self.features, ccc):
            self.log(
                f"train_{feature}", p, on_epoch=True, on_step=False, sync_dist=True
            )

        return loss


class ProsodyPredictor(pl.LightningModule):
    def __init__(self, output_dim):
        super().__init__()

        convolutions = []
        for _ in range(4):
            conv_layer = [
                XavierConv2d(
                    3,
                    3,
                    kernel_size=(5, 5),
                    stride=1,
                    padding="same",
                    nonlinearity="relu",
                ),
                nn.BatchNorm2d(3),
                nn.ReLU(),
            ]
            convolutions.extend(conv_layer)
        convolutions.extend(
            [
                XavierConv2d(
                    3,
                    1,
                    kernel_size=(5, 5),
                    stride=1,
                    padding="same",
                    nonlinearity="relu",
                ),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            ]
        )
        self.convolutions = nn.Sequential(*convolutions)

        self.lstm = nn.LSTM(80, 80 // 2, batch_first=True, bidirectional=True)
        self.lstm.flatten_parameters()

        self.linear = XavierLinear(80, output_dim)

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        mel_spectrogram = torch.exp(mel_spectrogram).swapaxes(1, 2)
        delta = torchaudio.functional.compute_deltas(mel_spectrogram)
        delta_delta = torchaudio.functional.compute_deltas(delta)

        x = torch.cat(
            [
                mel_spectrogram.unsqueeze(1),
                delta.unsqueeze(1),
                delta_delta.unsqueeze(1),
            ],
            dim=1,
        ).swapaxes(2, 3)

        conv = self.convolutions(x).squeeze(1)

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
