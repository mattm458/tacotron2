import csv
import json
import os
import re
from functools import partial
from os import path
from typing import Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from pandas import Series
from pqdm.processes import pqdm
from sklearn.preprocessing import OrdinalEncoder
from speech_utils.preprocessing.feature_extraction import extract_features

from consts import (
    FEATURES_ALL,
    FEATURES_ALL_DATASET_NORM,
    FEATURES_ALL_DATASET_NORM_CLIP,
    FEATURES_ALL_SPEAKER_NORM,
    FEATURES_ALL_SPEAKER_NORM_CLIP,
)
from preprocessing.utils import normalize


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


def __no_clip(sound: parselmouth.Sound):
    # Check if the sound will clip after resampling
    # Do this by checking the minimum and maximum amplitude from the string
    # representation of the Praat sound object. If max is above 0.99 or
    # the min is below -0.99, use built in scale_peak() function to
    # scale the audio so the peaks are no higher than +/- 0.99

    min_amplitude = None
    max_amplitude = None

    match_min = min_amplitude_regex.search(str(sound))
    match_max = max_amplitude_regex.search(str(sound))

    if match_min and match_max:
        min_amplitude = float(match_min.group(1))
        max_amplitude = float(match_max.group(1))

    if not min_amplitude or not max_amplitude:
        return None

    if min_amplitude < -0.99 or max_amplitude > 0.99:
        sound.scale_peak()


def __do_preprocess(
    speech_dir: str,
    trim: bool,
    trim_top_db: float,
    iterrow: Tuple[int, Series],
):
    _, row = iterrow
    row = row.copy()

    filepath = row.audio_filepath

    # Resample to 22050
    sound = parselmouth.Sound(path.join(speech_dir, filepath))
    sound = sound.resample(22050)
    __no_clip(sound)  # Prevent clipping
    resampled_filepath = "audio_22050" + row.audio_filepath[5:].replace("flac", "wav")
    out_filepath = path.join(speech_dir, resampled_filepath)
    os.makedirs(path.dirname(out_filepath), exist_ok=True)
    sound.save(file_path=out_filepath, format=parselmouth.SoundFileFormat.WAV)
    filepath = resampled_filepath

    # Trimming
    if trim:
        # Generate trimmed variant paths
        trimmed_filepath = "audio_22050_trimmed" + row.audio_filepath[5:].replace(
            "flac", "wav"
        )
        trimmed_out_filepath = path.join(speech_dir, trimmed_filepath)

        # Load the resampled file, trim it, and save the result
        wav, sr = librosa.load(path.join(speech_dir, resampled_filepath))
        trimmed, _ = librosa.effects.trim(wav, top_db=trim_top_db)
        os.makedirs(path.dirname(trimmed_out_filepath), exist_ok=True)
        sf.write(file=trimmed_out_filepath, data=trimmed, samplerate=sr)
        filepath = trimmed_filepath

    extracted_features = extract_features(
        wavfile=path.join(speech_dir, filepath), transcript=row.text_normalized
    )

    extracted_features["speaker_id"] = int(row.speaker_id)
    extracted_features["text"] = row.text_normalized

    extracted_features["wav"] = resampled_filepath

    return extracted_features


def __set_preprocess(
    speech_dir: str,
    set_name: str,
    n_jobs: int,
    trim: bool,
    trim_top_df: float,
):
    df = __load_set(speech_dir, set_name)

    results = pqdm(
        df.iterrows(),
        partial(__do_preprocess, speech_dir, trim, trim_top_df),
        n_jobs=n_jobs,
        desc=set_name,
    )

    results = [x for x in results if isinstance(x, dict)]
    results_df = pd.DataFrame(results)

    results_df[FEATURES_ALL_DATASET_NORM] = normalize(
        results_df[FEATURES_ALL],
        results_df[FEATURES_ALL].median(),
        results_df[FEATURES_ALL].std(),
    ).values
    results_df[FEATURES_ALL_DATASET_NORM_CLIP] = results_df[
        FEATURES_ALL_DATASET_NORM
    ].clip(-1, 1)

    return results_df


def do_preprocess(
    speech_dir: str,
    out_dir: str,
    out_postfix: str,
    n_jobs: int,
    trim: bool,
    trim_top_db: bool,
    split: bool,
    val_size: Optional[int],
    test_size: Optional[int],
    random_state: int,
):
    assert not split, "Splitting not supported in HiFi-TTS"

    # First preprocess the training data
    train_df = __set_preprocess(speech_dir, "train", n_jobs, trim, trim_top_db)
    val_df = __set_preprocess(speech_dir, "dev", n_jobs, trim, trim_top_db)
    test_df = __set_preprocess(speech_dir, "test", n_jobs, trim, trim_top_db)

    # Create an ordinal encoder to transform the speaker IDs from large, arbitrary
    # numbers into 0-indexed numbers
    id_encoder = OrdinalEncoder(dtype=np.int64)
    train_df.speaker_id = id_encoder.fit_transform(
        train_df.speaker_id.values.reshape(-1, 1)
    ).squeeze()

    # Transform the dev and test speaker IDs
    val_df.speaker_id = id_encoder.transform(
        val_df.speaker_id.values.reshape(-1, 1)
    ).squeeze()
    test_df.speaker_id = id_encoder.transform(
        test_df.speaker_id.values.reshape(-1, 1)
    ).squeeze()

    for set_df, set_name in zip([train_df, val_df, test_df], ["train", "val", "test"]):
        set_df = set_df.copy()

        # Dataset-level normalization:
        # Normalize feature values by the median and standard deviation of all
        # features in the dataset.
        set_df[FEATURES_ALL_DATASET_NORM] = normalize(
            set_df[FEATURES_ALL],
            train_df[FEATURES_ALL].median(),
            train_df[FEATURES_ALL].std(),
        ).values
        set_df[FEATURES_ALL_DATASET_NORM_CLIP] = set_df[FEATURES_ALL_DATASET_NORM].clip(
            -1, 1
        )

        # Speaker-level normalization:
        # Normalize feature values by the median and standard deviation of each
        # speaker individually.
        speaker_normalized = []
        for speaker_id, group in set_df.groupby("speaker_id"):
            speaker_norm = normalize(
                group[FEATURES_ALL],
                train_df.loc[train_df.speaker_id == speaker_id, FEATURES_ALL].median(),
                train_df.loc[train_df.speaker_id == speaker_id, FEATURES_ALL].std(),
            )
            speaker_norm.columns = FEATURES_ALL_SPEAKER_NORM
            speaker_normalized.append(speaker_norm)

        speaker_normalized_df = pd.concat(speaker_normalized, axis=0)
        set_df = pd.concat([set_df, speaker_normalized_df], axis=1)

        set_df[FEATURES_ALL_SPEAKER_NORM_CLIP] = set_df[FEATURES_ALL_SPEAKER_NORM].clip(
            -1, 1
        )

        set_df.to_csv(
            path.join(out_dir, f"hifi-tts-{set_name}-{out_postfix}.csv"),
            sep="|",
            quoting=csv.QUOTE_NONE,
            index=None,
        )
