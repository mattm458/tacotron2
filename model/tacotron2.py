import matplotlib

matplotlib.use("Agg")

from random import random

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

from model.decoder import Decoder
from model.encoder import Encoder
from model.gst import GST
from model.modules import AlwaysDropout, XavierLinear
from model.postnet import Postnet


class Tacotron2(pl.LightningModule):
    """This class implements Tacotron 2. It does so as a Lightning module, so it can be used as
    part of a Lightning training loop. The class is responsible for several things, including:

        - Running the encoder on input characters
        - Applying an attention-specific transformation to the encoded characters
        - Applying a prenet transformation previous Mel spectrogram steps
        - Invoking the decoder in a loop to produce output Mel spectrogram frames
        - Applying a post-net residual to the output Mel spectrogram
    """

    def __init__(
        self,
        lr,
        weight_decay,
        num_chars,
        char_embedding_dim,
        encoder_kernel_size,
        num_mels,
        prenet_dim,
        att_rnn_dim,
        att_dim,
        rnn_hidden_dim,
        postnet_dim,
        dropout,
        gst=None,
        gst_dim=None,
        speaker_embeddings=None,
        speaker_embeddings_dim=None,
        speech_features=False,
        speech_feature_dim=None,
        feature_detector=None,
        teacher_forcing=True,
    ):
        """Create a Tacotron2 object.

        Args:
            lr -- learning rate for the optimizer
            weight_decay -- weight decay for the optimizer
            num_chars -- The number of characters used in the dataset
            char_embedding_dim -- The character embedding size
            encoder_kernel_size -- size of the character input convolving kernel
            dropout -- the probability of elements to be zeroed out where dropout is applied
            num_mels -- number of Mel filterbanks to produce
            prenet_dim -- size of the Mel prenet output
            att_rnn_dim -- size of the hidden layer of the attention RNN
            att_dim -- size of hidden attention layers
            rnn_hidden_dim -- size of the hidden layer of the decoder RNN
            postnet_dim -- size of hidden layers in the postnet
            dropout -- the probability of elements to be zeroed out where dropout is applied
        """
        super().__init__()

        self.teacher_forcing = teacher_forcing

        self.gst = None
        self.speaker_embeddings = None

        self.embedding_dim = char_embedding_dim

        if gst_dim is not None and gst is not None:
            self.embedding_dim += gst_dim
            self.gst = gst

        if speaker_embeddings is not None and speaker_embeddings_dim is not None:
            self.embedding_dim += speaker_embeddings_dim
            self.speaker_embeddings = speaker_embeddings

        self.speech_features = speech_features

        self.prosody_predictor = feature_detector

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_mels = num_mels
        self.att_rnn_dim = att_rnn_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.char_embedding_dim = char_embedding_dim

        self.train_i = 0

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
            XavierLinear(num_mels, prenet_dim, bias=False, nonlinearity="linear"),
            nn.ReLU(),
            AlwaysDropout(dropout),
            XavierLinear(prenet_dim, prenet_dim, bias=False, nonlinearity="linear"),
            nn.ReLU(),
            AlwaysDropout(dropout),
        )

        # Additional encoder layer for attention. Done here since it applies to the entire
        # character input, and is only applied once before invoking the decoder
        self.att_encoder = XavierLinear(
            self.embedding_dim, att_dim, bias=False, nonlinearity="tanh"
        )

        # Tacotron 2 decoder
        self.decoder = Decoder(
            num_mels=num_mels,
            embedding_dim=self.embedding_dim,
            prenet_dim=prenet_dim,
            att_rnn_dim=att_rnn_dim,
            att_dim=att_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            dropout=dropout,
            speech_feature_dim=speech_feature_dim,
        )

        # Postnet layer. Done here since it applies to the entire Mel spectrogram output.
        self.postnet = Postnet(
            num_layers=5, num_mels=num_mels, postnet_dim=postnet_dim, dropout=dropout
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def __init_hidden(self, encoded_len, batch_size):
        """Generates initial hidden states, output tensors, and attention vectors.

        Args:
            encoded_len -- Length of the input character tensor
            batch_size -- Number of samples per batch
        """
        att_rnn_hidden = (
            torch.zeros(batch_size, self.att_rnn_dim, device=self.device),
            torch.zeros(batch_size, self.att_rnn_dim, device=self.device),
        )

        att_context = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        att_weights = torch.zeros(batch_size, encoded_len, device=self.device)
        att_weights_cum = torch.zeros(batch_size, encoded_len, device=self.device)

        rnn_hidden = (
            torch.zeros(batch_size, self.rnn_hidden_dim, device=self.device),
            torch.zeros(batch_size, self.rnn_hidden_dim, device=self.device),
        )

        return (
            att_rnn_hidden,
            att_context,
            att_weights,
            att_weights_cum,
            rnn_hidden,
        )

    def forward(self, data, gst=None, teacher_forcing=True):
        if self.gst is not None and gst is None:
            raise Exception("Tacotron2 is expecting a GST, but none was given")
        elif self.gst is None and gst is not None:
            raise Exception(
                "Tacotron2 was given a GST, but is not configured to use it"
            )

        tts_data, tts_metadata = data

        if isinstance(teacher_forcing, bool):
            all_teacher_forcing = teacher_forcing
            teacher_forcing = 1.0 if teacher_forcing else 0.0
        elif isinstance(teacher_forcing, float):
            all_teacher_forcing = teacher_forcing == 1.0

        # Encoding --------------------------------------------------------------------------------
        encoded = self.encoder(tts_data["chars_idx"], tts_metadata["chars_idx_len"])

        # if self.gst is not None:
        #     gst = gst.repeat(1, encoded.shape[1], 1)
        #     encoded = torch.cat([encoded, gst], dim=2)

        # Create a mask for the encoded characters
        encoded_mask = (
            torch.arange(tts_data["chars_idx"].shape[1], device=self.device)[None, :]
            < tts_metadata["chars_idx_len"][:, None]
        )

        if self.speaker_embeddings is not None:
            speaker_embeddings = self.speaker_embeddings(
                tts_metadata["speaker_id"]
            ).unsqueeze(1)
            speaker_embeddings = speaker_embeddings.repeat(1, encoded.shape[1], 1)
            encoded = torch.cat([encoded, speaker_embeddings], dim=2)

        # Transform the encoded characters for attention
        att_encoded = self.att_encoder(encoded)

        # Decoding --------------------------------------------------------------------------------
        batch_size = tts_data["mel_spectrogram"].shape[0]
        num_mels = tts_data["mel_spectrogram"].shape[2]

        # Get empty initial states
        (
            att_rnn_hidden,
            att_context,
            att_weights,
            att_weights_cum,
            rnn_hidden,
        ) = self.__init_hidden(
            encoded_len=encoded.shape[1],
            batch_size=batch_size,
        )

        max_len = 0

        # Decoder input: go frame (all ones), the complete Mel spectrogram, and a final 0
        # padding frame for iteration purposes
        decoder_in = torch.concat(
            [
                torch.zeros(batch_size, 1, num_mels, device=self.device),
                tts_data["mel_spectrogram"],
            ],
            1,
        )
        max_len = decoder_in.shape[1] - 1

        prev_mel = decoder_in[:, 0]

        mels = []
        gates = []
        alignments = []

        # Iterate through all decoder inputs
        for i in range(0, max_len):
            teacher_force = all_teacher_forcing or random() < teacher_forcing

            # Get the frame processed by the prenet
            prev_mel_prenet = self.prenet(prev_mel)

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
                prev_mel_prenet=prev_mel_prenet,
                att_rnn_hidden=att_rnn_hidden,
                att_context=att_context,
                att_weights=att_weights,
                att_weights_cum=att_weights_cum,
                rnn_hidden=rnn_hidden,
                encoded=encoded,
                att_encoded=att_encoded,
                encoded_mask=encoded_mask,
                speech_features=tts_metadata["features"],
            )

            # Save decoder output
            mels.append(mel_out)
            gates.append(gate_out)
            alignments.append(att_weights)

            # Prepare for the next iteration
            if teacher_force:
                prev_mel = decoder_in[:, i + 1]
            else:
                prev_mel = mel_out.detach()

        mels = torch.stack(mels, dim=1)
        gates = torch.stack(gates, dim=1)
        alignments = torch.stack(alignments, dim=1)

        # Run mel output through the postnet as a residual
        mels_post = self.postnet(mels.transpose(1, 2)).transpose(1, 2)
        mels_post = mels + mels_post

        mel_mask = (
            torch.arange(mels_post.shape[1], device=self.device)[None, :]
            >= tts_metadata["mel_spectrogram_len"][:, None]
        )

        mels = mels.swapaxes(1, 2).swapaxes(0, 1)
        mels_post = mels_post.swapaxes(1, 2).swapaxes(0, 1)
        gates = gates.swapaxes(1, 2).swapaxes(0, 1)
        mels = mels.masked_fill(mel_mask, 0.0).swapaxes(0, 1).swapaxes(1, 2)
        mels_post = mels_post.masked_fill(mel_mask, 0.0).swapaxes(0, 1).swapaxes(1, 2)
        gates = gates.masked_fill(mel_mask, -1000.0).swapaxes(0, 1).swapaxes(1, 2)

        return mels, mels_post, gates, alignments

    def training_step(self, batch, batch_idx):
        self.train_i += 1
        tts_data, tts_metadata = batch

        gst = None
        if self.gst is not None:
            gst = self.gst(tts_data.mel_spectrogram)
            mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
                batch, gst=gst, teacher_forcing=self.teacher_forcing
            )
        else:
            mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
                batch, teacher_forcing=self.teacher_forcing
            )

        gate_loss = F.binary_cross_entropy_with_logits(gate, tts_data["gate"])
        mel_loss = F.mse_loss(mel_spectrogram, tts_data["mel_spectrogram"])
        mel_post_loss = F.mse_loss(mel_spectrogram_post, tts_data["mel_spectrogram"])

        loss = gate_loss + mel_loss + mel_post_loss

        out = {
            "mel_spectrogram_pred": mel_spectrogram_post[0].detach(),
            "mel_spectrogram": tts_data["mel_spectrogram"][0].detach(),
            "alignment": alignment[0][
                : tts_metadata["mel_spectrogram_len"][0],
                : tts_metadata["chars_idx_len"][0],
            ].detach(),
            "gate": tts_data["gate"][0].detach(),
            "gate_pred": gate[0].detach(),
            "log": {"loss_train": loss.detach()},
        }

        if self.prosody_predictor is not None:
            self.prosody_predictor.requires_grad_(False)

            pred_mel_prosody = self.prosody_predictor(
                mel_spectrogram_post, tts_metadata["mel_spectrogram_len"]
            )
            mel_prosody = self.prosody_predictor(
                tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
            )

            prosody_loss = F.mse_loss(pred_mel_prosody, mel_prosody)
            out["prosody_loss"] = prosody_loss

            loss += prosody_loss

        out["loss"] = loss
        return out

    def validation_step(self, batch, batch_idx):
        tts_data, tts_metadata = batch

        gst = None
        if self.gst is not None:
            gst = self.gst(tts_data["mel_spectrogram"])
            mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
                batch, gst=gst, teacher_forcing=False
            )
        else:
            mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
                batch, teacher_forcing=False
            )

        gate_loss = F.binary_cross_entropy_with_logits(gate, tts_data["gate"])
        mel_loss = F.mse_loss(mel_spectrogram, tts_data["mel_spectrogram"])
        mel_post_loss = F.mse_loss(mel_spectrogram_post, tts_data["mel_spectrogram"])

        loss = gate_loss + mel_loss + mel_post_loss

        out = {
            "mel_spectrogram_pred": mel_spectrogram_post[0].detach(),
            "mel_spectrogram": tts_data["mel_spectrogram"][0].detach(),
            "alignment": alignment[0].detach(),
            "gate": tts_data["gate"][0].detach(),
            "gate_pred": gate[0].detach(),
        }

        if self.prosody_predictor is not None:
            pred_mel_prosody = self.prosody_predictor(
                mel_spectrogram_post, tts_metadata["mel_spectrogram_len"]
            )
            mel_prosody = self.prosody_predictor(
                tts_data["mel_spectrogram"], tts_metadata["mel_spectrogram_len"]
            )

            prosody_loss = F.mse_loss(pred_mel_prosody, mel_prosody)
            out["prosody_loss"] = prosody_loss

            loss += prosody_loss

        out["loss"] = loss
        return out

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        tts_data, tts_metadata = batch

        gst = None
        if self.gst is not None:
            gst = self.gst(tts_data["mel_spectrogram"])
            mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
                batch, gst=gst, teacher_forcing=False
            )
        else:
            mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
                batch, teacher_forcing=False
            )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment

    def validation_step_end(self, outputs):
        self.log("val_loss", outputs["loss"], on_step=True)

        if self.prosody_predictor is not None:
            self.log("val_prosody_loss", outputs["prosody_loss"].detach(), on_step=True)

        self.logger.experiment.add_image(
            "val_mel_spectrogram",
            plot_spectrogram_to_numpy(outputs["mel_spectrogram"].cpu().T.numpy()),
            self.train_i,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "val_mel_spectrogram_predicted",
            plot_spectrogram_to_numpy(outputs["mel_spectrogram_pred"].cpu().T.numpy()),
            self.train_i,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "val_alignment",
            plot_alignment_to_numpy(outputs["alignment"].cpu().T.numpy()),
            self.train_i,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "val_gate",
            plot_gate_outputs_to_numpy(
                outputs["gate"].cpu().squeeze().T.numpy(),
                torch.sigmoid(outputs["gate_pred"]).squeeze().cpu().T.numpy(),
            ),
            self.train_i,
            dataformats="HWC",
        )

    def training_step_end(self, outputs):
        self.log("training_loss", outputs["loss"].detach(), on_step=True)

        if self.prosody_predictor is not None:
            self.log(
                "prosody_loss",
                outputs["prosody_loss"].detach(),
                on_step=True,
                prog_bar=True,
            )

        if self.train_i % 1000 == 0:
            for name, parameter in self.named_parameters():
                self.logger.experiment.add_histogram(name, parameter, self.train_i)


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
