from model.modules import XavierConv2d, XavierLinear
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_lightning as pl
import torchaudio
import torchmetrics


class ProsodyPredictorV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.delta = torchaudio.transforms.ComputeDeltas()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, (5, 3), padding=(2, 1)),
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

        self.rnn = nn.GRU(768, 128, batch_first=True, bidirectional=True)

        self.frame_weight = nn.Linear(256, 256)
        self.context_weight = nn.Linear(256, 1)

        self.emotion_out = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, 7)
        )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        mel_spectrogram = mel_spectrogram.swapaxes(1, 2)
        if mel_spectrogram.shape[2] % 2 == 1:
            mel_spectrogram = torch.cat(
                [
                    mel_spectrogram,
                    torch.zeros(
                        (mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1),
                        device="cuda",
                    ),
                ],
                2,
            )

        d1 = self.delta(mel_spectrogram)
        d2 = self.delta(d1)

        x = torch.cat(
            [mel_spectrogram.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)], dim=1
        ).swapaxes(2, 3)

        output = self.conv(x)
        output = output.permute(0, 2, 3, 1).reshape(
            mel_spectrogram.shape[0], mel_spectrogram.shape[2], 256 * 10
        )

        output = self.pre_rnn(output)

        output, _ = self.rnn(output)

        att_output = self.frame_weight(output)
        att_output = self.context_weight(att_output)

        att_output = torch.softmax(att_output, 1)
        output = (output * att_output).sum(1)

        return self.emotion_out(output)

features = [
    "pitch_mean",
    "pitch_range",
    "intensity_mean",
    "jitter",
    "shimmer",
    "nhr",
    "duration",
]

class ProsodyPredictorLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.prosody_predictor = ProsodyPredictorV2()

        # self.loss = ConcordanceCorrelationCoefficientLoss()
        self.loss = nn.MSELoss()

        # self.train_pearsons = nn.ModuleList(
        #     [torchmetrics.PearsonCorrcoef() for x in features]
        # )
        # self.val_pearsons = nn.ModuleList(
        #     [torchmetrics.PearsonCorrcoef() for x in features]
        # )

    def forward(self, mel_spectrogram, mel_spectrogram_len):
        out = torch.tanh(self.prosody_predictor(mel_spectrogram, mel_spectrogram_len))
        return out

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=0.00001)

    # def training_step(self, batch, batch_idx):
    #     tts_data, tts_metadata = batch

    #     pred_features = self(
    #         tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
    #     )

    #     loss = self.loss(pred_features, tts_metadata["features_log_norm"])

    #     for i, (feature, p) in enumerate(zip(features, self.train_pearsons)):
    #         p(pred_features[:, i], tts_metadata["features_log_norm"][:, i])
    #         self.log(f"train_pearson_{feature}_step", p, prog_bar=True)

    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     tts_data, tts_metadata = batch

    #     pred_features = self(
    #         tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
    #     )

    #     loss = self.loss(pred_features, tts_metadata["features_log_norm"])

    #     for i, (feature, p) in enumerate(zip(features, self.val_pearsons)):
    #         p(pred_features[:, i], tts_metadata["features_log_norm"][:, i])
    #         self.log(f"train_pearson_{feature}_step", p, prog_bar=True)

    #     return loss

    # def predict_step(self, batch, batch_idx):
    #     tts_data, tts_metadata = batch

    #     pred_features = self(
    #         tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
    #     )

    #     return pred_features, tts_metadata["features_log_norm"]

    # def validation_step_end(self, outputs):
    #     self.log("val_loss", outputs.detach(), on_step=False, on_epoch=True)

    # def training_step_end(self, outputs):
    #     self.log("training_loss", outputs.detach(), on_step=False, on_epoch=True)

    # def validation_epoch_end(self, outputs):
    #     for feature, p in zip(features, self.val_pearsons):
    #         self.log(f"val_pearson_{feature}_epoch", p)

    # def training_epoch_end(self, outputs):
    #     for feature, p in zip(features, self.train_pearsons):
    #         self.log(f"train_pearson_{feature}_epoch", p)




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
