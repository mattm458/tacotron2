import csv
import datetime
import json
import logging
import os
from os import path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from lightning import Trainer
from tqdm import tqdm

from datasets.tts_dataloader import TTSDataLoader
from datasets.tts_dataset import TTSDataset
from model.hifi_gan import Generator
from model.tts_model import TTSModel


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def do_test(
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

    if use_hifi_gan:
        assert hifi_gan_checkpoint, "You must give a checkpoint if using HiFi-GAN"

        # logging.info(f"Loading HiFi-GAN checkpoint {hifi_gan_checkpoint}...")
        # hifi_gan_states = torch.load(
        #     hifi_gan_checkpoint, map_location=torch.device("cpu")
        # )["state_dict"]
        # hifi_gan_states = dict(
        #     [(k[10:], v) for k, v in hifi_gan_states.items() if "generator" in k]
        # )

        # generator = Generator()
        # generator.weight_norm()
        # generator.load_state_dict(hifi_gan_states)
        # generator.remove_weight_norm()
        # generator.eval()
        # generator = generator.cuda(device)

        with open("web_checkpoints/hifi-gan/UNIVERSAL_V1/config.json", "r") as infile:
            generator_config = AttrDict(json.load(infile))

        generator_state_dict = torch.load(
            "web_checkpoints/hifi-gan/UNIVERSAL_V1/g_02500000",
            map_location=f"cuda:{device}",
        )

        print(generator_config)
        generator = Generator(generator_config)
        generator.load_state_dict(generator_state_dict["generator"])

        generator.remove_weight_norm()
        generator.eval()
        generator = generator.cuda(device)

    test_df = pd.read_csv(
        dataset_config["test"], delimiter="|", quoting=csv.QUOTE_NONE, engine="c"
    )

    # If the config restricts the data to a single speaker ID, deal with this now
    if "force_speaker" in extensions_config["speaker_tokens"]:
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
        test_df = test_df[test_df.speaker_id == force_speaker_id].reset_index(
            drop=True
        )

    test_features = (
        test_df[extensions_config["controls"]["features"]].values.tolist()
        if extensions_config["controls"]["active"]
        else None
    )

    dataset_config["preprocessing"]["cache"] = False

    test_dataset = TTSDataset(
        filenames=test_df.wav,
        texts=test_df.text,
        base_dir=speech_dir,
        speaker_ids=test_df.speaker_id
        if extensions_config["speaker_tokens"]["active"]
        else None,
        features=test_features,
        **dataset_config["preprocessing"],
        include_text=True,
    )

    test_dataloader = TTSDataLoader(
        dataset=test_dataset,
        batch_size=8,
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
        map_location=torch.device("cpu"),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
        num_chars=len(dataset_config["preprocessing"]["allowed_chars"])
        + (dataset_config["preprocessing"]["end_token"] is not None),
        num_mels=dataset_config["preprocessing"]["num_mels"],
        max_len_override=5000,
        scheduler_milestones=[],
        **model_config["args"],
    )

    trainer = Trainer(logger=False, devices=[device], accelerator="gpu")
    results = trainer.predict(model, dataloaders=test_dataloader)

    del model

    i = 0

    if generator:
        with torch.no_grad():
            for _, mel_spectrogram_post, gate, _, texts in tqdm(
                results, desc="Saving WAVs"
            ):
                if mel_spectrogram_post.shape[1] == 5000:
                    logging.warn(
                        "Warning: Batch contains max-length Mel spectrogram! Skipping..."
                    )
                    exit()

                mel_lengths = (gate < 0).type(torch.long).squeeze(-1).argmax(dim=-1)
                wav_lengths = mel_lengths * 256

                wavs = generator(mel_spectrogram_post.cuda(device).swapaxes(1, 2))
                wavs = wavs.squeeze(1)
                del mel_spectrogram_post
                wavs = wavs.cpu().numpy()

                for wav, wav_length, text in zip(wavs, wav_lengths, texts):
                    i += 1
                    if wav_length == 0:
                        wav_length = -1

                    wav = wav[:wav_length]

                    if wav_length == -1:
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
                continue
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
                    with open(path.join(results_dir, "failures.csv"), "a") as fail_log:
                        fail_log.write(f"{i}|{text}\n")
