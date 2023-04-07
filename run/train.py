import csv
import datetime
import os
from os import path

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.tts_model import TTSModel


def do_train(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    device: int,
    speech_dir: str,
):
    train_df = pd.read_csv(
        dataset_config["train"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )
    val_df = pd.read_csv(
        dataset_config["val"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )

    train_dataset = TTSDataset(
        filenames=train_df.wav,
        texts=train_df.text,
        base_dir=speech_dir,
        **dataset_config["preprocessing"],
    )
    val_dataset = TTSDataset(
        filenames=val_df.wav,
        texts=val_df.text,
        base_dir=speech_dir,
        **dataset_config["preprocessing"],
    )

    train_dataloader = TTSDataLoader(
        dataset=train_dataset,
        batch_size=training_config["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_dataloader = TTSDataLoader(
        dataset=val_dataset,
        batch_size=training_config["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )

    results_dir = f"results_{training_config['name']} {datetime.datetime.now()}"
    os.mkdir(results_dir)

    torch.set_float32_matmul_precision(training_config["float32_matmul_precision"])

    logger = TensorBoardLogger(
        path.join(results_dir, "lightning_logs"), name=training_config["name"]
    )

    model = TTSModel(
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        **model_config["args"],
    )

    trainer = Trainer(
        logger=logger,
        devices=[device],
        accelerator="gpu",
        precision=training_config["precision"],
        gradient_clip_val=1.0,
        max_epochs=training_config["max_epochs"],
        check_val_every_n_epoch=1,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
