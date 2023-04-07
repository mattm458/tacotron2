import csv
import datetime
import logging
import os
from os import path
from typing import Optional
from torch.nn import functional as F

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from hifi_gan.model.generator import Generator
from lightning import Trainer
from tqdm import tqdm

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.tts_model import TTSModel


def do_test(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    device: int,
    speech_dir: str,
    checkpoint: str,
    hifi_gan_checkpoint: Optional[str] = None,
):
    use_hifi_gan = hifi_gan_checkpoint is not None
    generator = None

    if use_hifi_gan:
        logging.info(f"Loading HiFi-GAN checkpoint {hifi_gan_checkpoint}...")
        hifi_gan_states = torch.load(hifi_gan_checkpoint, map_location="cuda:0")[
            "state_dict"
        ]
        hifi_gan_states = dict(
            [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
        )

        generator = Generator()
        generator.weight_norm()
        generator.load_state_dict(hifi_gan_states)
        generator.remove_weight_norm()
        generator.eval()
        generator = generator.cuda()

    test_df = pd.read_csv(
        dataset_config["test"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )

    test_dataset = TTSDataset(
        filenames=test_df.wav,
        texts=test_df.text,
        base_dir=speech_dir,
        **dataset_config["preprocessing"],
    )

    test_dataloader = TTSDataLoader(
        dataset=test_dataset,
        batch_size=training_config["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )

    results_dir = f"results_{training_config['name']}_test {datetime.datetime.now()}"
    os.mkdir(results_dir)

    torch.set_float32_matmul_precision(training_config["float32_matmul_precision"])

    model = TTSModel.load_from_checkpoint(
        checkpoint,
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        **model_config["args"],
    )

    trainer = Trainer(
        logger=None,
        devices=[device],
        accelerator="gpu",
        gradient_clip_val=1.0,
        max_epochs=training_config["max_epochs"],
        check_val_every_n_epoch=1,
    )

    results = trainer.predict(
        model,
        dataloaders=test_dataloader,
    )

    i = 0

    if use_hifi_gan:
        with torch.no_grad():
            for _, mel_spectrogram_post, gate, _ in tqdm(results, desc="Saving WAVs"):
                if mel_spectrogram_post.shape[1] == 2000:
                    logging.warn(
                        "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                    )
                    continue

                mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
                wav_lengths = mel_lengths * 256

                wavs = generator(mel_spectrogram_post.cuda())
                wavs = wavs.cpu().numpy()

                for wav, wav_length in zip(wavs, wav_lengths):
                    i += 1
                    wav = wav[:wav_length]
                    if len(wav) == 0:
                        continue

                    sf.write(
                        path.join(results_dir, f"{i}.wav"), wav[:wav_length], 22050
                    )
    else:
        for _, mel_spectrogram_post, gate, _ in tqdm(results, desc="Saving WAVs"):
            if mel_spectrogram_post.shape[1] == 2000:
                logging.warn(
                    "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                )
                continue
            mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
            for this_mel_spectrogram, mel_length in zip(
                mel_spectrogram_post, mel_lengths
            ):
                i += 1

                this_mel_spectrogram = this_mel_spectrogram[:mel_length]
                mel_spectrogram_post = np.exp(this_mel_spectrogram.numpy())

                wav = librosa.feature.inverse.mel_to_audio(
                    mel_spectrogram_post.T,
                    sr=22050,
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024,
                    center=True,
                    power=1.0,
                    fmin=0,
                    fmax=8000,
                )

                sf.write(path.join(results_dir, f"{i}.wav"), wav, 22050)
