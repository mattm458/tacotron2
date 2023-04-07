import matplotlib

matplotlib.use("Agg")

from random import random
from typing import Dict, Optional, Tuple

from random import random
import numpy as np
import lightning as pl
import torch
import torchaudio
from hifi_gan.model.hifi_gan import discriminator_loss, generator_loss, feature_loss
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F

from model.decoder_legacy import Decoder
from model.encoder_legacy import Encoder
from model.modules_legacy import AlwaysDropout, XavierLinear
from model.postnet_legacy import Postnet
from utils.hifi_gan import get_random_segment


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
        att_rnn_dim: int,
        att_dim,
        rnn_hidden_dim,
        postnet_dim,
        dropout,
        speech_features=False,
        speech_feature_dim=None,
        prosody_model=None,
        post_prosody_model=None,
        hifi_gan_generator=None,
        hifi_gan_mpd=None,
        hifi_gan_msd=None,
        hifi_gan_spectrogram=None,
        fine_tune_style: bool = False,
        fine_tune_features: bool = False,
        fine_tune_hifi_gan: bool = False,
        fine_tune_hifi_gan_segment_size: int = 96,
        fine_tune_hifi_gan_style: bool = False,
        fine_tune_tacotron: bool = False,
        fine_tune_tacotron_limit: int = -1,
        teacher_forcing: bool = True,
        freeze_tacotron: bool = False,
        teacher_forcing_prob=1.0,  # 0.0,
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
        self.embedding_dim = char_embedding_dim

        self.speech_features = speech_features

        # Conditioning
        self.prosody_model = prosody_model
        self.post_prosody_model = post_prosody_model
        self.hifi_gan_generator = hifi_gan_generator
        self.hifi_gan_mpd = hifi_gan_mpd
        self.hifi_gan_msd = hifi_gan_msd
        self.hifi_gan_spectrogram = hifi_gan_spectrogram
        self.fine_tune_style = fine_tune_style
        self.fine_tune_features = fine_tune_features
        self.fine_tune_hifi_gan = fine_tune_hifi_gan
        self.fine_tune_hifi_gan_segment_size = fine_tune_hifi_gan_segment_size

        self.fine_tune_tacotron = fine_tune_tacotron
        self.fine_tune_tacotron_limit = fine_tune_tacotron_limit

        self.teacher_forcing_prob = teacher_forcing_prob

        if self.fine_tune_tacotron:
            if self.fine_tune_tacotron_limit < 0:
                print("Tacotron: Fine-tuning Tacotron model")
            else:
                print(
                    f"Tacotron: Fine-tuning Tacotron model for {fine_tune_tacotron_limit} iterations"
                )

        if self.fine_tune_hifi_gan:
            print("Tacotron: Fine-tuning a HiFi-GAN model")
            self.fine_tune_tacotron_limit *= 2
            self.automatic_optimization = False

        self.lr = lr
        self.weight_decay = weight_decay
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

        self.tacotron_modules = [
            self.encoder,
            self.prenet,
            self.att_encoder,
            self.decoder,
            self.postnet,
        ]

        self.freeze_tacotron = freeze_tacotron
        if freeze_tacotron:
            print("Tacotron: Freezing Tacotron weights")
            for x in self.tacotron_modules:
                x.requires_grad_(False)
        else:
            print("Tacotron: Training Tacotron weights")

    def configure_optimizers(self):
        if not self.freeze_tacotron:
            tacotron_parameters = sum(
                [list(x.parameters()) for x in self.tacotron_modules], []
            )
        else:
            tacotron_parameters = []

        if self.fine_tune_hifi_gan:
            return torch.optim.Adam(
                tacotron_parameters + list(self.hifi_gan_generator.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            ), torch.optim.Adam(
                list(self.hifi_gan_mpd.parameters())
                + list(self.hifi_gan_msd.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            return torch.optim.Adam(
                tacotron_parameters, lr=self.lr, weight_decay=self.weight_decay
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
        data: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        teacher_forcing: bool = True,
        sigmoid_gates: bool = False,
        save_mel: Optional[str] = None,
        force_to_gpu: Optional[str] = None,
        teacher_forcing_prob=1.0
    ):
        tts_data, tts_metadata, _ = data

        tts_data_mel_spectrogram = tts_data["mel_spectrogram"]
        tts_metadata_mel_spectrogram_len = tts_metadata["mel_spectrogram_len"]
        tts_data_chars_idx = tts_data["chars_idx"]

        tts_metadata_features = tts_metadata["features"]
        tts_metadata_chars_idx_len = tts_metadata["chars_idx_len"]

        device = tts_data_chars_idx.device

        if force_to_gpu is not None:
            if teacher_forcing is True:
                tts_data_mel_spectrogram = tts_data_mel_spectrogram.to(force_to_gpu)

            tts_metadata_mel_spectrogram_len = tts_metadata_mel_spectrogram_len.to(
                force_to_gpu
            )
            tts_data_chars_idx = tts_data_chars_idx.to(force_to_gpu)

            tts_metadata_features = tts_metadata_features.to(force_to_gpu)
            tts_metadata_chars_idx_len = tts_metadata_chars_idx_len.to(force_to_gpu)

        # Encoding --------------------------------------------------------------------------------
        encoded = self.encoder(tts_data_chars_idx, tts_metadata_chars_idx_len)

        # Create a mask for the encoded characters
        encoded_mask = (
            torch.arange(tts_data_chars_idx.shape[1], device=encoded.device)[None, :]
            < tts_metadata_chars_idx_len[:, None]
        )

        # Transform the encoded characters for attention
        att_encoded = self.att_encoder(encoded)

        # Decoding --------------------------------------------------------------------------------
        batch_size = tts_data_chars_idx.shape[0]

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

        max_len = tts_data_mel_spectrogram.shape[1]

        if teacher_forcing:
            decoder_in = F.pad(tts_data_mel_spectrogram, (0, 0, 1, 0))

            if self.teacher_forcing:
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
            lengths = tts_metadata_mel_spectrogram_len
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
                speech_features=tts_metadata_features if self.speech_features else None,
            )

            # Save decoder output
            mels.append(mel_out)
            gates.append(gate_out)
            alignments.append(att_weights)

            # Prepare for the next iteration
            teacher_forcing_rand = random() < teacher_forcing_prob
            if teacher_forcing and teacher_forcing_rand:
                prev_mel = decoder_in[i + 1]
            elif teacher_forcing and not teacher_forcing_rand:
                prev_mel = self.prenet(mel_out.detach())
            else:
                done[gate_out.squeeze(-1) < 0.0] = True
                lengths[gate_out.squeeze(-1) >= 0.0] += 1
                #print(lengths)
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

        if save_mel is not None:
            torch.save(mels_post.cpu().detach(), save_mel)

        if sigmoid_gates:
            gates = torch.sigmoid(gates)

        return mels, mels_post, gates, alignments

    def validation_step(self, batch, batch_idx):
        tts_data, tts_metadata, _ = batch

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            batch,
            teacher_forcing=self.teacher_forcing,
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

        if self.fine_tune_tacotron and self.prosody_model is not None:
            (
                output_features,
                output_features_low,
                output_features_medium,
                output_features_high,
            ) = self.prosody_model(
                mel_spectrogram_post,
                tts_metadata["mel_spectrogram_len"],
            )
            (
                ground_truth_features,
                ground_truth_features_low,
                ground_truth_features_medium,
                ground_truth_features_high,
            ) = self.prosody_model(
                tts_data["mel_spectrogram"],
                tts_metadata["mel_spectrogram_len"],
            )

            style_loss = (
                F.mse_loss(output_features_low, ground_truth_features_low)
                + F.mse_loss(output_features_medium, ground_truth_features_medium)
                + F.mse_loss(output_features_high, ground_truth_features_high)
            )
            loss = loss + style_loss
            self.log("val_style_loss", style_loss, on_step=False, on_epoch=True)

            feature_loss_val = F.mse_loss(output_features, ground_truth_features)
            loss = loss + feature_loss_val
            self.log("val_feature_loss", feature_loss_val, on_step=False, on_epoch=True)

        if self.hifi_gan_generator is not None:
            if "wav" not in tts_data:
                raise Exception("Ground-truth waveform data is not in TTS data!")
            if "wav_len" not in tts_metadata:
                raise Exception("Waveform lengths unavailable in TTS metadata!")

            mel_segment_pred, mel_segment_len, wav_segment_true = get_random_segment(
                mel=mel_spectrogram_post,
                mel_len=mel_spectrogram_len,
                wav=tts_data["wav"],
                wav_len=tts_metadata["wav_len"],
                segment_size=self.fine_tune_hifi_gan_segment_size,
                hop_len=256,
            )

            wav_segment_true_mel = F.pad(
                wav_segment_true, ((1024 - 256) // 2, (1024 - 256) // 2)
            )
            mel_segment_true = self.hifi_gan_spectrogram(wav_segment_true_mel)

            # Compute the output waveform
            wav_segment_pred = self.hifi_gan_generator(mel_segment_pred)

            # Create a Mel spectrogram from the predicted waveform
            wav_segment_pred_mel = F.pad(
                wav_segment_pred, ((1024 - 256) // 2, (1024 - 256) // 2)
            )
            mel_segment_pred = self.hifi_gan_spectrogram(wav_segment_pred_mel)

            hifi_gan_mel_loss = F.l1_loss(mel_segment_pred, mel_segment_true)
            loss = loss + hifi_gan_mel_loss

            self.log(
                "val_hifi_gan_mel_loss", hifi_gan_mel_loss, on_epoch=True, on_step=True
            )

            if self.post_prosody_model is not None:
                _, low_pred, medium_pred, high_pred = self.post_prosody_model(
                    mel_segment_pred, mel_segment_len
                )
                _, low, medium, high = self.post_prosody_model(
                    mel_segment_true, mel_segment_len
                )

                style_loss = (
                    F.mse_loss(low_pred, low)
                    + F.mse_loss(medium_pred, medium)
                    + F.mse_loss(high_pred, high)
                )

                self.log("val_style_loss", style_loss, on_step=False, on_epoch=True)
                loss = loss + style_loss

        self.log("val_loss", loss, on_step=False, on_epoch=True)

        out["loss"] = loss
        return out

    def training_step(self, batch, batch_idx):
        if (
            self.fine_tune_tacotron
            and self.fine_tune_tacotron_limit > -1
            and self.global_step > self.fine_tune_tacotron_limit
        ):
            if self.global_step == self.fine_tune_tacotron_limit:
                print("Tacotron: Disabling Tacotron gradients...")
            for x in self.tacotron_modules:
                x.requires_grad_(False)

        if self.fine_tune_hifi_gan:
            tacotron_optimizer, discriminator_optimizer = self.optimizers()

        tts_data, tts_metadata, _ = batch

        teacher_forcing=True
        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            batch,
            teacher_forcing=True,teacher_forcing_prob=self.teacher_forcing_prob
        )

        loss = 0

        if not self.freeze_tacotron:
            if teacher_forcing:
                gate_loss = F.binary_cross_entropy_with_logits(gate, tts_data["gate"])
                mel_loss = F.mse_loss(mel_spectrogram, tts_data["mel_spectrogram"])
                mel_post_loss = F.mse_loss(
                    mel_spectrogram_post, tts_data["mel_spectrogram"]
                )

                tacotron_loss = gate_loss + mel_loss + mel_post_loss
                loss = tacotron_loss

                self.log(
                    "training_gate_loss",
                    gate_loss.detach(),
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    "training_mel_loss", mel_loss.detach(), on_step=True, on_epoch=True
                )
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

        if teacher_forcing:
            mel_spectrogram_len = tts_metadata["mel_spectrogram_len"]
            #print("TEACHER FORCING")
        else:
            #print("NOT TEACHER FORCING")
            #print((gate < 0.0).type(torch.long).squeeze(-1))
            mel_spectrogram_len = torch.argmax(
                (gate < 0.0).type(torch.long).squeeze(-1), dim=1
            )
            mel_spectrogram_len[mel_spectrogram_len == 0] = mel_spectrogram_post.shape[
                1
            ]
        # print("len", mel_spectrogram_len)
        # print("max len", mel_spectrogram_len.max())
        # print("len actual", tts_metadata["mel_spectrogram_len"])
        # print(mel_spectrogram_post.shape)
        # print()
        mel_spectrogram = tts_data["mel_spectrogram"]

        out = {
            "mel_spectrogram_pred": mel_spectrogram_post.detach(),
            "mel_spectrogram": mel_spectrogram.detach(),
            "alignment": alignment.detach(),
            "gate": tts_data["gate"][0].detach(),
            "gate_pred": gate[0].detach(),
        }

        if not self.freeze_tacotron and self.prosody_model is not None:
            (
                output_features,
                output_features_low,
                output_features_medium,
                output_features_high,
            ) = self.prosody_model(
                mel_spectrogram_post,
                mel_spectrogram_len,
            )
            (
                ground_truth_features,
                ground_truth_features_low,
                ground_truth_features_medium,
                ground_truth_features_high,
            ) = self.prosody_model(
                tts_data["mel_spectrogram"],
                tts_metadata["mel_spectrogram_len"],
            )

            if self.fine_tune_style:
                style_loss = (
                    F.mse_loss(output_features_low, ground_truth_features_low)
                    + F.mse_loss(output_features_medium, ground_truth_features_medium)
                    + F.mse_loss(output_features_high, ground_truth_features_high)
                )
                loss = loss + style_loss
                self.log(
                    "training_style_loss",
                    style_loss.detach(),
                    on_step=True,
                    on_epoch=True,
                )
            if self.fine_tune_features:
                feature_loss_val = F.mse_loss(output_features, ground_truth_features)
                loss = loss + feature_loss_val
                self.log(
                    "training_feature_loss",
                    feature_loss_val.detach(),
                    on_step=True,
                    on_epoch=True,
                )

        if self.hifi_gan_generator is not None:
            if "wav" not in tts_data:
                raise Exception("Ground-truth waveform data is not in TTS data!")
            if "wav_len" not in tts_metadata:
                raise Exception("Waveform lengths unavailable in TTS metadata!")

            # Get a random segment from the input audio to give to HiFi-GAN
            mel_segment_pred, mel_segment_len, wav_segment_true = get_random_segment(
                mel=mel_spectrogram_post,
                mel_len=mel_spectrogram_len,
                wav=tts_data["wav"],
                wav_len=tts_metadata["wav_len"],
                segment_size=self.fine_tune_hifi_gan_segment_size,
                hop_len=256,
            )

            wav_segment_true_mel = F.pad(
                wav_segment_true, ((1024 - 256) // 2, (1024 - 256) // 2)
            )
            mel_segment_true = self.hifi_gan_spectrogram(wav_segment_true_mel)

            # Compute the output waveform
            wav_segment_pred = self.hifi_gan_generator(mel_segment_pred)

            if self.fine_tune_hifi_gan:
                discriminator_optimizer.zero_grad()

            # Train the discriminators
            # =====================================================================
            y_df_hat_r, y_df_hat_g, _, _ = self.hifi_gan_mpd(
                wav_segment_true, wav_segment_pred.detach()
            )
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = self.hifi_gan_msd(
                wav_segment_true, wav_segment_pred.detach()
            )
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_discriminator = loss_disc_s + loss_disc_f
            self.manual_backward(loss_discriminator)
            discriminator_optimizer.step()

            self.log(
                "training_hifi_gan_discriminator_loss",
                loss_discriminator.detach(),
                on_epoch=True,
                on_step=True,
            )

            # Train the generator and Tacotron
            # =====================================================================
            if self.fine_tune_hifi_gan:
                tacotron_optimizer.zero_grad()

            # Create a Mel spectrogram from the predicted waveform
            wav_segment_pred_mel = F.pad(
                wav_segment_pred, ((1024 - 256) // 2, (1024 - 256) // 2)
            )
            mel_segment_pred = self.hifi_gan_spectrogram(wav_segment_pred_mel)

            (
                _,
                output_features_low,
                output_features_medium,
                output_features_high,
            ) = self.post_prosody_model(
                torch.log(torch.clamp(mel_segment_pred, min=1e-5)),
                mel_segment_len,
            )
            (
                _,
                ground_truth_features_low,
                ground_truth_features_medium,
                ground_truth_features_high,
            ) = self.post_prosody_model(
                torch.log(torch.clamp(mel_segment_true, min=1e-5)),
                mel_segment_len,
            )

            post_style_loss = (
                F.mse_loss(output_features_low, ground_truth_features_low)
                + F.mse_loss(output_features_medium, ground_truth_features_medium)
                + F.mse_loss(output_features_high, ground_truth_features_high)
            ) * 2400
            loss = loss + post_style_loss
            self.log(
                "training_post_style_loss",
                post_style_loss.detach(),
                on_step=True,
                on_epoch=True,
            )

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.hifi_gan_mpd(
                wav_segment_true, wav_segment_pred
            )
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.hifi_gan_msd(
                wav_segment_true, wav_segment_pred
            )

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            # Mel spectrogram loss
            hifi_gan_mel_loss = F.l1_loss(mel_segment_true, mel_segment_pred) * 45

            self.log(
                "training_hifi_gan_mel_loss",
                hifi_gan_mel_loss.detach(),
                on_epoch=True,
                on_step=True,
            )

            loss_generator = (
                loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + hifi_gan_mel_loss
            )

            loss = loss + loss_generator

            self.log(
                "training_hifi_gan_generator_loss",
                loss_generator.detach(),
                on_epoch=True,
                on_step=True,
            )

            self.manual_backward(loss)
            tacotron_optimizer.step()

        self.log("training_loss", loss.detach(), on_step=True, on_epoch=True)

        out["loss"] = loss
        return out

    def validation_step_end(self, outputs):
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
        tts_data, tts_metadata, _ = batch

        mel_spectrogram, mel_spectrogram_post, gate, alignment = self(
            batch, teacher_forcing=False, save_mel="test"
        )

        return mel_spectrogram, mel_spectrogram_post, gate, alignment


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
