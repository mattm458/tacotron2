import csv
import datetime
import os
from os import path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.tts_model import TTSModel


def do_train_mel_export(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    extensions_config: dict,
    device: int,
    speech_dir: str,
    checkpoint: str,
):
    dataset_config["preprocessing"]["cache"] = False

    train_df = pd.read_csv(
        dataset_config["train"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )
    val_df = pd.read_csv(
        dataset_config["val"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )

    train_features = (
        train_df[extensions_config["controls"]["features"]].values.tolist()
        if extensions_config["controls"]["active"]
        else None
    )
    val_features = (
        val_df[extensions_config["controls"]["features"]].values.tolist()
        if extensions_config["controls"]["active"]
        else None
    )

    train_dataset = TTSDataset(
        filenames=train_df.wav,
        texts=train_df.text,
        base_dir=speech_dir,
        speaker_ids=train_df.speaker_id
        if extensions_config["speaker_tokens"]["active"]
        else None,
        features=train_features,
        **dataset_config["preprocessing"],
        include_text=False,
        include_filename=True,
    )
    val_dataset = TTSDataset(
        filenames=val_df.wav,
        texts=val_df.text,
        base_dir=speech_dir,
        speaker_ids=val_df.speaker_id
        if extensions_config["speaker_tokens"]["active"]
        else None,
        features=val_features,
        **dataset_config["preprocessing"],
        include_text=False,
        include_filename=True,
    )

    train_dataloader = TTSDataLoader(
        dataset=train_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )
    val_dataloader = TTSDataLoader(
        dataset=val_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )

    results_dir = (
        f"results_{training_config['name']}_train_mel_export {datetime.datetime.now()}"
    )
    os.mkdir(results_dir)

    torch.set_float32_matmul_precision(training_config["float32_matmul_precision"])

    model = TTSModel.load_from_checkpoint(
        checkpoint,
        map_location=torch.device("cpu"),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        max_len_override=5000,
        **model_config["args"],
    ).cuda(device)

    model.eval()

    i = 0

    with torch.no_grad():
        for dataloader in [train_dataloader, val_dataloader]:
            for tts_data, tts_metadata, tts_extra in tqdm(dataloader):
                args = {}

                if model.speaker_tokens:
                    args["speaker_id"] = tts_metadata["speaker_id"].cuda(device)

                if model.controls:
                    args["controls"] = tts_metadata["features"].cuda(device)

                _, mel_spectrogram_post, gate, _ = model(
                    teacher_forcing=True,
                    chars_idx=tts_data["chars_idx"].cuda(device),
                    chars_idx_len=tts_metadata["chars_idx_len"].cuda(device),
                    mel_spectrogram=tts_data["mel_spectrogram"].cuda(device),
                    mel_spectrogram_len=tts_metadata["mel_spectrogram_len"].cuda(
                        device
                    ),
                    **args,
                )

                for mel_out, length, filename in zip(
                    mel_spectrogram_post.cpu(),
                    tts_metadata["mel_spectrogram_len"],
                    tts_extra["filename"],
                ):
                    out_path = path.join(
                        results_dir, f"{filename.replace('/', '_')}.np"
                    )
                    np.save(out_path, mel_out[:length].numpy())
