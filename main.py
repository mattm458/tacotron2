import argparse
import csv
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

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.gst import GST
from model.speaker_embeddings import utils as speaker_embedding_utils
from model.tacotron2 import Tacotron2


def load_dataset(filepath, config, dataset_dir, base_dir="./"):
    df = pd.read_csv(
        path.join(base_dir, filepath),
        delimiter="|",
        quoting=csv.QUOTE_NONE,
    )
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
            features = config["extensions"]["features"]["allowed_features"]
            if config["extensions"]["features"]["normalize_by"] is None:
                args["features"] = df[features].values.tolist()
                args["features_log"] = np.log(df[features]).values.tolist()
                args["features_norm"] = zscore(df[features]).values.tolist()
                args["features_log_norm"] = zscore(np.log(df[features])).values.tolist()

    return TTSDataset(**args, base_dir=dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Tacotron2 instance")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="A YAML configuration file containing hyperparameters",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=False,
        default=None,
        help="The base dataset directory",
    )
    parser.add_argument(
        "--inference",
        type=bool,
        required=False,
        default=False,
        help="Set to True to conduct inference instead of training",
    )
    parser.add_argument(
        "--train",
        type=bool,
        required=False,
        default=True,
        help="Set to True to conduct training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        required=False,
        help="The path to a model checkpoint",
    )
    parser.add_argument(
        "--say", required=False, type=str, default=None, help="The text to say"
    )
    parser.add_argument(
        "--wav-out",
        type=str,
        default=None,
        required=False,
        help="Where to save a generated wav file",
    )
    parser.add_argument(
        "--with-speech-features",
        type=float,
        required=None,
        default=None,
        nargs="+",
        help="Speech features to include in inference",
    )

    args = parser.parse_args()

    with open(args.config, "r") as infile:
        config = yaml.load(infile, Loader=yaml.Loader)

    if args.train:
        dataset_dir = args.dataset_dir

        train_dataset = load_dataset(config["data"]["train"], config, dataset_dir)
        test_dataset = load_dataset(config["data"]["test"], config, dataset_dir)
        val_dataset = load_dataset(config["data"]["val"], config, dataset_dir)

        train_dataloader = TTSDataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            prefetch_factor=config["data"]["prefetch_factor"],
            pin_memory=True,
            shuffle=True,
        )
        test_dataloader = TTSDataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
        )
        val_dataloader = TTSDataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
            shuffle=False,
        )

    tacotron_args = {}

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

    if args.inference:
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
            **tacotron_args
        )

        ALLOWED_CHARS = (
            "!'(),.:;? \\-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        end_token = "^"
        encoder = OrdinalEncoder()
        encoder.fit([[x] for x in list(ALLOWED_CHARS) + [end_token]])

        encoded = (
            torch.LongTensor(
                encoder.transform([[x] for x in args.say.lower()] + [[end_token]])
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
            "features_log_norm": torch.Tensor([[3.0, 2.0, -0.1, 4]]),
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

    elif args.train:
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
            # gst=gst,
            # gst_dim=256,
            **tacotron_args
        )

        trainer = Trainer(
            devices=config["training"]["devices"],
            accelerator=config["training"]["accelerator"],
            precision=config["training"]["precision"],
            gradient_clip_val=1.0,
            max_epochs=150,
        )

        trainer.fit(
            tacotron2,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.checkpoint,
        )
