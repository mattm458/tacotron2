from typing import List, Optional

import lightning as pl
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from prosody_modeling.model.prosody_model import ProsodyModel
from torch import Tensor
from torch.nn import functional as F

from model.tacotron2 import Tacotron2

matplotlib.use("Agg")


class TTSModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
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
        scheduler_milestones: List[int],
        speaker_tokens: bool = False,
        num_speakers: int = 1,
        controls: bool = False,
        controls_dim: int = 0,
        max_len_override: Optional[int] = None,
        prosody_model: Optional[ProsodyModel] = None,
        prosody_model_after: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_milestones = scheduler_milestones
        self.speaker_tokens = speaker_tokens
        self.controls = controls
        self.max_len_override = max_len_override

        self.prosody_model = prosody_model
        self.prosody_model_after = prosody_model_after

        self.tacotron2 = Tacotron2(
            num_chars=num_chars,
            char_embedding_dim=char_embedding_dim,
            encoder_kernel_size=encoder_kernel_size,
            num_mels=num_mels,
            prenet_dim=prenet_dim,
            att_rnn_dim=att_rnn_dim,
            att_dim=att_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            postnet_dim=postnet_dim,
            dropout=dropout,
            speaker_tokens=speaker_tokens,
            num_speakers=num_speakers,
            controls=controls,
            controls_dim=controls_dim,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.tacotron2.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        optimizer_config = {"optimizer": optimizer}

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.scheduler_milestones, gamma=0.1
        )
        lr_scheduler_config = {"scheduler": scheduler, "interval": "step"}
        optimizer_config["lr_scheduler"] = lr_scheduler_config

        return optimizer_config

    def forward(
        self,
        chars_idx: Tensor,
        chars_idx_len: Tensor,
        teacher_forcing: bool = True,
        mel_spectrogram: Optional[Tensor] = None,
        mel_spectrogram_len: Optional[Tensor] = None,
        speaker_id: Optional[Tensor] = None,
        controls: Optional[Tensor] = None,
        max_len_override: Optional[int] = None,
    ):
        return self.tacotron2(
            chars_idx=chars_idx,
            chars_idx_len=chars_idx_len,
            teacher_forcing=teacher_forcing,
            mel_spectrogram=mel_spectrogram,
            mel_spectrogram_len=mel_spectrogram_len,
            speaker_id=speaker_id,
            controls=controls,
            max_len_override=max_len_override,
        )

    def validation_step(self, batch, batch_idx):
        tts_data, tts_metadata, _ = batch

        args = {}
        if self.speaker_tokens:
            args["speaker_id"] = tts_metadata["speaker_id"]

        if self.controls:
            args["controls"] = tts_metadata["features"]

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=tts_data["chars_idx"],
            chars_idx_len=tts_metadata["chars_idx_len"],
            teacher_forcing=True,
            mel_spectrogram=tts_data["mel_spectrogram"],
            mel_spectrogram_len=tts_metadata["mel_spectrogram_len"],
            **args,
        )

        gate_loss = F.binary_cross_entropy_with_logits(gate, tts_data["gate"])
        mel_loss = F.mse_loss(mel_spectrogram, tts_data["mel_spectrogram"])
        mel_post_loss = F.mse_loss(mel_spectrogram_post, tts_data["mel_spectrogram"])

        loss = gate_loss + mel_loss + mel_post_loss
        self.log("val_mel_loss", loss, on_step=False, on_epoch=True)

        mel_spectrogram_len = tts_metadata["mel_spectrogram_len"]
        chars_idx_len = tts_metadata["chars_idx_len"]

        mel_spectrogram = tts_data["mel_spectrogram"]
        mel_spectrogram_pred = mel_spectrogram_post

        out = {
            "mel_spectrogram_pred": mel_spectrogram_pred[0, : mel_spectrogram_len[0]],
            "mel_spectrogram": mel_spectrogram[0, : mel_spectrogram_len[0]],
            "alignment": alignment[0, : mel_spectrogram_len[0], : chars_idx_len[0]],
            "gate": tts_data["gate"][0],
            "gate_pred": gate[0],
        }

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx):
        tts_data, tts_metadata, _ = batch

        args = {}
        if self.speaker_tokens:
            args["speaker_id"] = tts_metadata["speaker_id"]

        if self.controls:
            args["controls"] = tts_metadata["features"]

        if (
            self.prosody_model is not None
            and self.global_step >= self.prosody_model_after
        ):
            _, low, mid, high = self.prosody_model(
                tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
            )

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=tts_data["chars_idx"],
            chars_idx_len=tts_metadata["chars_idx_len"],
            teacher_forcing=True,
            mel_spectrogram=tts_data["mel_spectrogram"],
            mel_spectrogram_len=tts_metadata["mel_spectrogram_len"],
            **args,
        )

        loss = 0

        gate_loss = F.binary_cross_entropy_with_logits(gate, tts_data["gate"])
        mel_loss = F.mse_loss(mel_spectrogram, tts_data["mel_spectrogram"])
        mel_post_loss = F.mse_loss(mel_spectrogram_post, tts_data["mel_spectrogram"])

        tacotron_loss = gate_loss + mel_loss + mel_post_loss
        loss = tacotron_loss

        if (
            self.prosody_model is not None
            and self.global_step >= self.prosody_model_after
        ):
            _, low_pred, mid_pred, high_pred = self.prosody_model(
                mel_spectrogram_post, tts_metadata["mel_spectrogram_len"]
            )

            style_loss = (
                F.mse_loss(low_pred, low)
                + F.mse_loss(mid_pred, mid)
                + F.mse_loss(high_pred, high)
            )

            self.log(
                "training_style_loss",
                style_loss.detach(),
                on_step=True,
                on_epoch=True,
            )

            loss += style_loss

        self.log(
            "training_gate_loss",
            gate_loss.detach(),
            on_step=True,
            on_epoch=True,
        )
        self.log("training_mel_loss", mel_loss.detach(), on_step=True, on_epoch=True)
        self.log(
            "training_mel_post_loss",
            mel_post_loss.detach(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "training_tacotron_loss",
            tacotron_loss.detach(),
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "training_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
        )

        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if batch_idx > 0:
            return

        self.logger.experiment.add_image(
            "val_mel_spectrogram",
            plot_spectrogram_to_numpy(outputs["mel_spectrogram"].cpu().T.numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "val_mel_spectrogram_predicted",
            plot_spectrogram_to_numpy(
                outputs["mel_spectrogram_pred"].cpu().swapaxes(0, 1).numpy()
            ),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "val_alignment",
            plot_alignment_to_numpy(outputs["alignment"].cpu().swapaxes(0, 1).numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "val_gate",
            plot_gate_outputs_to_numpy(
                outputs["gate"].cpu().squeeze().numpy(),
                torch.sigmoid(outputs["gate_pred"]).squeeze().cpu().numpy(),
            ),
            self.global_step,
            dataformats="HWC",
        )

    def training_step_end(self, outputs):
        if self.global_step % 1000 == 0:
            for name, parameter in self.named_parameters():
                self.logger.experiment.add_histogram(name, parameter, self.global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        tts_data, tts_metadata, tts_extra = batch

        text = tts_extra["text"]

        args = {}
        if self.speaker_tokens:
            args["speaker_id"] = tts_metadata["speaker_id"]

        if self.controls:
            args["controls"] = tts_metadata["features"]

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            chars_idx=tts_data["chars_idx"],
            chars_idx_len=tts_metadata["chars_idx_len"],
            teacher_forcing=False,
            mel_spectrogram=tts_data["mel_spectrogram"],
            mel_spectrogram_len=tts_metadata["mel_spectrogram_len"],
            max_len_override=self.max_len_override,
            **args,
        )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment, text


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    fig.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Encoder timestep")
    fig.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(
        range(len(gate_targets)),
        gate_targets,
        alpha=0.5,
        color="green",
        marker="+",
        s=1,
        label="target",
    )
    ax.scatter(
        range(len(gate_outputs)),
        gate_outputs,
        alpha=0.5,
        color="red",
        marker=".",
        s=1,
        label="predicted",
    )

    ax.set_xlabel("Frames (Green target, Red predicted)")
    ax.set_ylabel("Gate State")
    fig.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close(fig)
    return data
