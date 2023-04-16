import csv
import json
import os
import re
from functools import partial
from os import path
from typing import Tuple

import numpy as np
import pandas as pd
import parselmouth
from pandas import Series
from pqdm.processes import pqdm
from sklearn.preprocessing import OrdinalEncoder
from speech_utils.preprocessing.feature_extraction import extract_features


def __load_set(base_dir: str, set: str):
    df = []
    for file in (x for x in os.listdir(base_dir) if "clean" in x and set in x):
        id = file.split("_")[0]

        with open(path.join(base_dir, f"{id}_manifest_clean_{set}.json")) as infile:
            for line in infile:
                json_data = json.loads(line)
                json_data["speaker_id"] = id
                df.append(json_data)

    return pd.DataFrame(df)


min_amplitude_regex = re.compile(r"Minimum: (-{0,1}\d+.\d+) Pascal")
max_amplitude_regex = re.compile(r"Maximum: (-{0,1}\d+.\d+) Pascal")


def __do_preprocess(speech_dir: str, iterrow: Tuple[int, Series]):
    _, row = iterrow
    row = row.copy()

    filepath = path.join(speech_dir, row.audio_filepath)
    sound = parselmouth.Sound(filepath)
    sound = sound.resample(22050)

    min_amplitude = float(min_amplitude_regex.search(str(sound)).group(1))
    max_amplitude = float(max_amplitude_regex.search(str(sound)).group(1))

    if min_amplitude < -0.99 or max_amplitude > 0.99:
        sound.scale_peak()

    extracted_features = extract_features(sound=sound, transcript=row.text_normalized)
    extracted_features["speaker_id"] = int(row.speaker_id)
    extracted_features["text"] = row.text_normalized

    wav_filepath = "audio_22050" + row.audio_filepath[5:].replace("flac", "wav")
    extracted_features["filename"] = wav_filepath

    save_filepath = path.join(speech_dir, wav_filepath)

    os.makedirs(path.dirname(save_filepath), exist_ok=True)
    sound.save(file_path=save_filepath, format=parselmouth.SoundFileFormat.WAV)

    return extracted_features


def __set_preprocess(speech_dir: str, set: str, n_jobs: int):
    df = __load_set(speech_dir, set)

    results = pqdm(
        df.iterrows(), partial(__do_preprocess, speech_dir), n_jobs=n_jobs, desc=set
    )

    results = [x for x in results if isinstance(x, dict)]

    return pd.DataFrame(results)


def do_preprocess(speech_dir: str, out_dir: str, out_postfix: str, n_jobs: int):
    # First preprocess the training data
    train_results = __set_preprocess(speech_dir, "train", n_jobs)
    dev_results = __set_preprocess(speech_dir, "dev", n_jobs)
    test_results = __set_preprocess(speech_dir, "test", n_jobs)

    # Create an ordinal encoder to transform the speaker IDs from large, arbitrary
    # numbers into 0-indexed numbers
    id_encoder = OrdinalEncoder(dtype=np.int64)
    train_results.speaker_id = id_encoder.fit_transform(
        train_results.speaker_id.values.reshape(-1, 1)
    ).squeeze()

    # Transform the dev and test speaker IDs
    dev_results.speaker_id = id_encoder.transform(
        dev_results.speaker_id.values.reshape(-1, 1)
    ).squeeze()
    test_results.speaker_id = id_encoder.transform(
        test_results.speaker_id.values.reshape(-1, 1)
    ).squeeze()

    # Save the results
    train_results.to_csv(
        path.join(out_dir, f"hifi-tts-train-{out_postfix}.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )

    dev_results.to_csv(
        path.join(out_dir, f"hifi-tts-val-{out_postfix}.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )

    test_results.to_csv(
        path.join(out_dir, f"hifi-tts-test-{out_postfix}.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
