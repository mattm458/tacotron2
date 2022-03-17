import csv
import os
from os import path

import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile
import torch
import torchaudio
import yaml
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from scipy.stats import zscore
from sklearn.preprocessing import OrdinalEncoder
from torch import nn
from tqdm import tqdm

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.gst import GST
from model.prosodic_features import prosody_detector
from model.speaker_embeddings import utils as speaker_embedding_utils
from model.tacotron2 import Tacotron2
from utils.args import args


def load_dataset(
    df, config, dataset_dir, base_dir="./", df_train=None, feature_override=None
):
    args = {"filenames": list(df.wav), "texts": list(df.text_normalized)}

    if "model" in config and "extensions" in config["model"]:
        if "speaker_embeddings" in config["model"]["extensions"]:
            encoder = speaker_embedding_utils.get_encoder(
                path.join(
                    base_dir, config["extensions"]["speaker_embeddings"]["speaker_ids"]
                ),
                base_dir,
            )

            args["speaker_ids"] = encoder.transform(df.speaker_id)

        if "features" in config["model"]["extensions"]:
            features = df[config["extensions"]["features"]["allowed_features"]]
            args["features"] = features.values.tolist()

    return TTSDataset(
        **args,
        base_dir=dataset_dir,
        feature_override=feature_override,
        max_mel_len=None,
        max_text_len=None,
    )


if __name__ == "__main__":
    with open(args.config, "r") as infile:
        config = yaml.load(infile, Loader=yaml.Loader)

    if args.mode == "train":
        dataset_dir = args.dataset_dir

        df_train = pd.read_csv(
            config["data"]["train"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )
        df_test = pd.read_csv(
            config["data"]["test"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )
        df_val = pd.read_csv(
            config["data"]["val"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )

        train_dataset = load_dataset(df_train, config, dataset_dir)
        val_dataset = load_dataset(df_val, config, dataset_dir, df_train=df_train)

        train_dataloader = TTSDataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            prefetch_factor=config["data"]["prefetch_factor"],
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )
        val_dataloader = TTSDataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

    tacotron_args = {}

    use_trainer_checkpoint = True
    prosody_predictor = None

    if "model" in config and "extensions" in config["model"]:
        if "speaker_embeddings" in config["model"]["extensions"]:
            tacotron_args["speaker_embeddings"] = nn.Embedding(
                config["extensions"]["speaker_embeddings"]["num_speakers"],
                config["extensions"]["speaker_embeddings"]["embedding_dim"],
            )
            tacotron_args["speaker_embeddings_dim"] = config["extensions"][
                "speaker_embeddings"
            ]["embedding_dim"]
        if "features" in config["model"]["extensions"]:
            tacotron_args["speech_features"] = True
            tacotron_args["speech_features_key"] = config["extensions"]["features"][
                "features_key"
            ]
            tacotron_args["speech_feature_dim"] = len(
                config["extensions"]["features"]["allowed_features"]
            )

            if "feature_detector" in config["model"]["extensions"]:
                if (
                    "feature_extractor_checkpoint" not in args
                    or (
                        "feature_extractor_checkpoint" in args
                        and args.feature_extractor_checkpoint is None
                    )
                ) and args.checkpoint is None:
                    print(
                        "Error! Transfer learning of Tacotron model with uninitialized prosody detector!"
                    )
                    exit()

                if (
                    "feature_extractor_checkpoint" in args
                    and args.feature_extractor_checkpoint is not None
                ):
                    prosody_predictor = prosody_detector.ProsodyPredictorLightning.load_from_checkpoint(
                        args.feature_extractor_checkpoint,
                        # output_dim=tacotron_args["speech_feature_dim"],
                    )
                    use_trainer_checkpoint = False
                    tacotron_args["feature_detector"] = prosody_predictor
                else:
                    prosody_predictor = prosody_detector.ProsodyPredictorLightning(
                        # output_dim=tacotron_args["speech_feature_dim"]
                    )
                    for x in prosody_predictor.prosody_predictor.parameters():
                        x.requires_grad = False
                    for x in prosody_predictor.parameters():
                        x.requires_grad = False

                    tacotron_args["feature_detector"] = prosody_predictor
                    use_trainer_checkpoint = False

    if args.mode == "say":
        tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            num_chars=len(config["preprocessing"]["valid_chars"]) + 1,
            encoder_kernel_size=config["encoder"]["encoder_kernel_size"],
            num_mels=config["audio"]["num_mels"],
            char_embedding_dim=config["encoder"]["char_embedding_dim"],
            prenet_dim=config["decoder"]["prenet_dim"],
            att_rnn_dim=config["attention"]["att_rnn_dim"],
            att_dim=config["attention"]["att_dim"],
            rnn_hidden_dim=config["decoder"]["rnn_hidden_dim"],
            postnet_dim=config["decoder"]["postnet_dim"],
            dropout=config["tacotron2"]["dropout"],
            teacher_forcing=False,
            **tacotron_args,
        )

        ALLOWED_CHARS = (
            "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        end_token = "^"
        encoder = OrdinalEncoder()
        encoder.fit([[x] for x in list(ALLOWED_CHARS) + [end_token]])

        encoded = (
            torch.LongTensor(
                encoder.transform([[x] for x in args.text.lower()] + [[end_token]])
            )
            .squeeze(1)
            .unsqueeze(0)
        ) + 1

        tts_data = {
            "chars_idx": torch.LongTensor(encoded),
            "mel_spectrogram": torch.zeros((1, 250, 80)),
        }
        tts_metadata = {
            "chars_idx_len": torch.IntTensor([encoded.shape[1]]),
            "mel_spectrogram_len": torch.IntTensor([800]),
            "features_log_norm": torch.Tensor([[-2, 0, 0, 0, 0, 0, 0]]),
        }

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=1.0,
        )

        with torch.no_grad():
            tacotron2.eval()
            mels, mels_post, gates, alignments = tacotron2.predict_step(
                (tts_data, tts_metadata), 0, 0
            )

        mels_post = torch.exp(mels_post[0])
        wav = librosa.feature.inverse.mel_to_audio(
            mels_post.numpy().T,
            sr=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            power=1,
        )
        soundfile.write("output.wav", wav, samplerate=16000)

    if args.mode == "test":
        if not path.exists(args.dir_out):
            os.mkdir(args.dir_out)

        dataset_dir = args.dataset_dir

        df_train = pd.read_csv(
            config["data"]["test"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )
        df_test = pd.read_csv(
            config["data"]["test"],
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )

        test_dataset = load_dataset(
            df_test,
            config,
            dataset_dir,
            df_train=df_train,
            feature_override=args.with_speech_features,
        )

        test_dataloader = TTSDataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
        )

        tacotron2 = Tacotron2.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            num_chars=len(config["preprocessing"]["valid_chars"]) + 1,
            encoder_kernel_size=config["encoder"]["encoder_kernel_size"],
            num_mels=config["audio"]["num_mels"],
            char_embedding_dim=config["encoder"]["char_embedding_dim"],
            prenet_dim=config["decoder"]["prenet_dim"],
            att_rnn_dim=config["attention"]["att_rnn_dim"],
            att_dim=config["attention"]["att_dim"],
            rnn_hidden_dim=config["decoder"]["rnn_hidden_dim"],
            postnet_dim=config["decoder"]["postnet_dim"],
            dropout=config["tacotron2"]["dropout"],
            teacher_forcing=False,
            **tacotron_args,
        )

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            gradient_clip_val=1.0,
        )

        out = trainer.predict(tacotron2, dataloaders=test_dataloader)

        i = 0
        for mel_spectrogram, mel_spectrogram_post, gate, alignment in out:
            for m in tqdm(range(len(mel_spectrogram_post))):
                stop_idx = 0
                g = torch.sigmoid(gate[m])

                for j in range(len(g)):
                    stop_idx = j
                    if g[j] <= 0.5:
                        break

                mel = torch.exp(mel_spectrogram_post[m, :stop_idx])
                wav = librosa.feature.inverse.mel_to_audio(
                    mel.numpy().T,
                    sr=16000,
                    n_fft=1024,
                    win_length=1024,
                    hop_length=256,
                    power=1,
                )

                soundfile.write(
                    path.join(args.dir_out, f"{i}.wav"), wav, samplerate=16000
                )
                i += 1

    elif args.mode == "train":
        if use_trainer_checkpoint:
            tacotron2 = Tacotron2(
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
                num_chars=len(config["preprocessing"]["valid_chars"]) + 1,
                encoder_kernel_size=config["encoder"]["encoder_kernel_size"],
                num_mels=config["audio"]["num_mels"],
                char_embedding_dim=config["encoder"]["char_embedding_dim"],
                prenet_dim=config["decoder"]["prenet_dim"],
                att_rnn_dim=config["attention"]["att_rnn_dim"],
                att_dim=config["attention"]["att_dim"],
                rnn_hidden_dim=config["decoder"]["rnn_hidden_dim"],
                postnet_dim=config["decoder"]["postnet_dim"],
                dropout=config["tacotron2"]["dropout"],
                teacher_forcing=config["training"]["teacher_forcing"],
                **tacotron_args,
            )
        else:
            if "feature_detector" in tacotron_args:
                feature_detector = tacotron_args["feature_detector"]
                del tacotron_args["feature_detector"]

            tacotron2 = Tacotron2.load_from_checkpoint(
                args.checkpoint,
                lr=config["training"]["lr"],
                weight_decay=config["training"]["weight_decay"],
                num_chars=len(config["preprocessing"]["valid_chars"]) + 1,
                encoder_kernel_size=config["encoder"]["encoder_kernel_size"],
                num_mels=config["audio"]["num_mels"],
                char_embedding_dim=config["encoder"]["char_embedding_dim"],
                prenet_dim=config["decoder"]["prenet_dim"],
                att_rnn_dim=config["attention"]["att_rnn_dim"],
                att_dim=config["attention"]["att_dim"],
                rnn_hidden_dim=config["decoder"]["rnn_hidden_dim"],
                postnet_dim=config["decoder"]["postnet_dim"],
                dropout=config["tacotron2"]["dropout"],
                teacher_forcing=config["training"]["teacher_forcing"],
                # gst=gst,
                # gst_dim=256,
                **tacotron_args,
            )

            tacotron2.prosody_predictor = feature_detector

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=1.0,
            max_epochs=config["training"]["max_epochs"],
            check_val_every_n_epoch=2,
        )

        trainer.fit(
            tacotron2,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.checkpoint if use_trainer_checkpoint else None,
        )
