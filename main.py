import argparse
import csv

import pandas as pd
import yaml
from pytorch_lightning import Trainer
from torch import nn

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.gst import GST
from model.tacotron2 import Tacotron2


def load_dataset(filepath, config):
    df = pd.read_csv(filepath, delimiter="|", header=None, quoting=csv.QUOTE_NONE)
    return TTSDataset(list(df[0]), list(df[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Tacotron2 instance")
    parser.add_argument(
        "--config",
        type=str,
        default="example-hyperparameters/vanilla-tacotron.yaml",
        help="A YAML configuration file containing hyperparameters",
    )

    args = parser.parse_args()

    with open(args.config, "r") as infile:
        config = yaml.load(infile, Loader=yaml.Loader)

    train_dataset = load_dataset(config["data"]["train"], config)
    test_dataset = load_dataset(config["data"]["test"], config)
    val_dataset = load_dataset(config["data"]["val"], config)

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

    # gst = GST()
    speaker_embeddings = nn.Embedding(
        config["extensions"]["speaker_embeddings"]["num_speakers"],
        config["extensions"]["speaker_embeddings"]["embedding_dim"],
    )
    speaker_embeddings_dim = config["extensions"]["speaker_embeddings"]["embedding_dim"]

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
        speaker_embeddings=speaker_embeddings,
        speaker_embeddings_dim=speaker_embeddings_dim,
    )

    trainer = Trainer(
        devices=config["training"]["devices"],
        accelerator=config["training"]["accelerator"],
        precision=config["training"]["precision"],
        gradient_clip_val=1.0,
    )

    trainer.fit(
        tacotron2,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path="lightning_logs/version_55/checkpoints/epoch=81-step=16317.ckpt",
    )
