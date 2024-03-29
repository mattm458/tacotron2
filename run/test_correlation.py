import csv
import datetime
import logging
import os
from os import path
from typing import Optional

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


def do_test_correlation(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    extensions_config: dict,
    device: int,
    speech_dir: str,
    checkpoint: str,
    hifi_gan_checkpoint: Optional[str] = None,
):
    use_hifi_gan = hifi_gan_checkpoint is not None
    generator = None

    feature_override = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    feature_overrides = set()
    for i in range(7):
        for j in np.arange(-1, 1.1, 0.5):
            feature_override_temp = feature_override[:]
            feature_override_temp[i] = j
            feature_overrides.add(tuple(feature_override_temp))

    if use_hifi_gan:
        assert hifi_gan_checkpoint, "You must give a checkpoint if using HiFi-GAN"

        logging.info(f"Loading HiFi-GAN checkpoint {hifi_gan_checkpoint}...")
        hifi_gan_states = torch.load(hifi_gan_checkpoint, map_location=f"cuda:{device}")[
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
        generator = generator.cuda(device)

    test_df = pd.read_csv(
        dataset_config["test"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )

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

    base_results_dir = (
        f"results_{training_config['name']}_test {datetime.datetime.now()}"
    )

    for feature_override_i, feature_override in enumerate(feature_overrides):
        print(f"{feature_override_i} / {len(feature_overrides)-1}: {feature_override}")

        test_features = [list(feature_override)] * len(
            test_df[extensions_config["controls"]["features"]].values
        )

        dataset_config["preprocessing"]["cache"] = False

        test_dataset = TTSDataset(
            filenames=test_df.wav,
            texts=test_df.text,
            base_dir=speech_dir,
            speaker_ids=test_df.speaker_id if extensions_config["speaker_id"] else None,
            features=test_features,
            **dataset_config["preprocessing"],
            include_text=True,
        )

        test_dataloader = TTSDataLoader(
            dataset=test_dataset,
            batch_size=32,
            num_workers=3,
            pin_memory=True,
            shuffle=False,
            persistent_workers=False,
            drop_last=False,
        )

        results_dir = path.join(base_results_dir, str(feature_override))
        os.makedirs(results_dir)

        trainer = Trainer(
            logger=None,
            devices=[device],
            accelerator="gpu",
            gradient_clip_val=1.0,
            check_val_every_n_epoch=1,
        )

        results = trainer.predict(
            model,
            dataloaders=test_dataloader,
        )

        i = 0

        if generator:
            with torch.no_grad():
                for _, mel_spectrogram_post, gate, _, texts in tqdm(
                    results, desc="Saving WAVs"
                ):
                    if mel_spectrogram_post.shape[1] == 2000:
                        logging.warn(
                            "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                        )
                        exit()

                    mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
                    wav_lengths = mel_lengths * 256

                    wavs = generator(mel_spectrogram_post.cuda(device))
                    del mel_spectrogram_post
                    wavs = wavs.cpu().numpy()

                    for wav, wav_length, text in zip(wavs, wav_lengths, texts):
                        i += 1
                        wav = wav[:wav_length]

                        if len(wav) == 0:
                            print(f"Error: {i}: {text}")
                            with open(
                                path.join(results_dir, "failures.csv"), "a"
                            ) as fail_log:
                                fail_log.write(f"{i}|{text}\n")

                        sf.write(
                            path.join(results_dir, f"{i}.wav"), wav[:wav_length], 22050
                        )
        else:
            for _, mel_spectrogram_post, gate, _, texts in tqdm(
                results, desc="Saving WAVs"
            ):
                if mel_spectrogram_post.shape[1] == 2000:
                    logging.warn(
                        "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                    )
                    exit()
                mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
                for this_mel_spectrogram, mel_length, text in zip(
                    mel_spectrogram_post, mel_lengths, texts
                ):
                    i += 1

                    this_mel_spectrogram = this_mel_spectrogram[:mel_length]
                    mel_spectrogram_post = np.exp(this_mel_spectrogram.numpy())

                    try:
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
                    except ValueError:
                        print(f"Error: {i}: {text}")
                        with open(
                            path.join(results_dir, "failures.csv"), "a"
                        ) as fail_log:
                            fail_log.write(f"{i}|{text}\n")
