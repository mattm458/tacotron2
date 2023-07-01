import csv
import datetime
import os
from os import path
from typing import Optional

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.tts_model import TTSModel
from prosody_modeling.model.lightning import ProsodyModelLightning


def do_train(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    extensions_config: dict,
    device: int,
    speech_dir: str,
    results_dir: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
    prosody_model_checkpoint: Optional[str] = None,
):
    if results_dir is None:
       results_dir = f"results_{training_config['name']} {datetime.datetime.now()}"
       os.mkdir(results_dir)

    cache_dir = path.join(results_dir, "mel_cache")

    train_df = pd.read_csv(
        dataset_config["train"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )
    val_df = pd.read_csv(
        dataset_config["val"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )

    # If the config restricts the data to a single speaker ID, deal with this now
    if extensions_config["speaker_tokens"]["force_speaker"] is not None:
        # This is a single-speaker model - no need for speaker tokens
        if extensions_config["speaker_tokens"]["active"]:
            raise Exception("Cannot use speaker tokens with force_speaker parameter!")

        # If we're offering controls, ensure we're only offering them over speaker-normalized
        # controls. Otherwise we might get weird results!
        if extensions_config["controls"]["active"]:
            if not all(
                ["speaker_norm" in x for x in extensions_config["controls"]["features"]]
            ):
                raise Exception(
                    "If force_speaker, all controls must be for speaker-normalized values!"
                )

        force_speaker_id = extensions_config["speaker_tokens"]["force_speaker"]
        train_df = train_df[train_df.speaker_id == force_speaker_id].reset_index(drop=True)
        val_df = val_df[val_df.speaker_id == force_speaker_id].reset_index(drop=True)

    train_features = (
        train_df[extensions_config["controls"]["features"]].values.tolist()
        if extensions_config["controls"]["active"]
        else None
    )

    train_dataset = TTSDataset(
        filenames=train_df.wav,
        texts=train_df.text,
        speaker_ids=train_df.speaker_id
        if extensions_config["speaker_tokens"]["active"]
        else None,
        features=train_features,
        base_dir=speech_dir,
        cache_dir=cache_dir,
        **dataset_config["preprocessing"],
    )

    val_features = (
        val_df[extensions_config["controls"]["features"]].values.tolist()
        if extensions_config["controls"]["active"]
        else None
    )

    val_dataset = TTSDataset(
        filenames=val_df.wav,
        texts=val_df.text,
        speaker_ids=train_df.speaker_id
        if extensions_config["speaker_tokens"]["active"]
        else None,
        features=val_features,
        base_dir=speech_dir,
        cache_dir=cache_dir,
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

    torch.set_float32_matmul_precision(training_config["float32_matmul_precision"])

    logger = TensorBoardLogger(
        path.join(results_dir, "lightning_logs"), name=training_config["name"]
    )

    controls = False
    controls_dim = 0
    if extensions_config["controls"]["active"]:
        controls = True
        controls_dim = len(extensions_config["controls"]["features"])

    speaker_tokens = False
    num_speakers = 1
    if extensions_config["speaker_tokens"]["active"]:
        speaker_tokens = True
        num_speakers = extensions_config["speaker_tokens"]["num_speakers"]

    if extensions_config["prosody_model"]["active"]:
        if prosody_model_checkpoint is None:
            raise Exception(
                "Prosody model extension is active, but no prosody model checkpoint was given!"
            )

        prosody_model = ProsodyModelLightning.load_from_checkpoint(
            prosody_model_checkpoint
        ).prosody_predictor

        for param in prosody_model.parameters():
            param.requires_grad = False

        prosody_model_after = int(
            training_config["args"]["max_steps"]
            * extensions_config["prosody_model"]["active_after"]
        )

        model_config["args"]["prosody_model"] = prosody_model
        model_config["args"]["prosody_model_after"] = prosody_model_after

    scheduler_milestones = [
        int(x * training_config["args"]["max_steps"])
        for x in model_config["scheduler_milestones"]
    ]

    model = TTSModel(
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        controls=controls,
        controls_dim=controls_dim,
        speaker_tokens=speaker_tokens,
        num_speakers=num_speakers,
        scheduler_milestones=scheduler_milestones,
        **model_config["args"],
    )

    trainer = Trainer(
        logger=logger,
        devices=[device],
        accelerator="gpu",
        precision=training_config["precision"],
        gradient_clip_val=1.0,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        **training_config["args"],
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt,
    )
