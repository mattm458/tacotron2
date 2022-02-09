import argparse
import csv
from os import path

import pandas as pd
import yaml
from pytorch_lightning import Trainer
from torch import nn

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.gst import GST
from model.speaker_embeddings import utils as speaker_embedding_utils
from model.tacotron2 import Tacotron2
from scipy.stats import zscore


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
        
        if "features" in config['model']['extensions']:
            features = config['extensions']['features']['allowed_features']
            if config['extensions']['features']['normalize_by'] is None:
                args['features'] = zscore(df[features]).values.tolist()

    return TTSDataset(**args, base_dir=dataset_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Tacotron2 instance")
    parser.add_argument(
        "--config",
        type=str,
        default="example-hyperparameters/vanilla-tacotron.yaml",
        help="A YAML configuration file containing hyperparameters",
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="The base dataset directory"
    )

    args = parser.parse_args()

    with open(args.config, "r") as infile:
        config = yaml.load(infile, Loader=yaml.Loader)

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

    # gst = GST()
    args = {}

    if "model" in config and "extensions" in config["model"]:
        if "speaker_embeddings" in config["model"]["extensions"]:
            args["speaker_embeddings"] = nn.Embedding(
                config["extensions"]["speaker_embeddings"]["num_speakers"],
                config["extensions"]["speaker_embeddings"]["embedding_dim"],
            )
            args["speaker_embeddings_dim"] = config["extensions"]["speaker_embeddings"][
                "embedding_dim"
            ]

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
        **args
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
