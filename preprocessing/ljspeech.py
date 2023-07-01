import csv
import os
from functools import partial
from os import path
from typing import Tuple

import librosa
import pandas as pd
import soundfile as sf
from pandas import Series
from pqdm.processes import pqdm
from speech_utils.preprocessing.feature_extraction import extract_features


def __do_preprocess(
    speech_dir: str, trim: bool, trim_top_db: float, iterrow: Tuple[int, Series]
):
    _, row = iterrow
    row = row.copy()

    filepath = path.join(speech_dir, "wavs", f"{row.id}.wav")

    if trim:
        trimmed_filepath = path.join(speech_dir, "wavs_trimmed", f"{row.id}.wav")

        # Load the original file, trim it, and save the result
        wav, sr = librosa.load(filepath)
        trimmed, _ = librosa.effects.trim(wav, top_db=trim_top_db)
        sf.write(file=trimmed_filepath, data=trimmed, samplerate=sr)

        filepath = trimmed_filepath

    extracted_features = extract_features(
        wavfile=filepath, transcript=row.text_normalized
    )

    if extracted_features is None:
        return None

    extracted_features["text"] = row.text_normalized

    root, filename = path.split(filepath)
    root, wav_dir = path.split(root)

    wav_filepath = path.join(wav_dir, filename)

    extracted_features["wav"] = wav_filepath

    return extracted_features


def do_preprocess(
    speech_dir: str,
    out_dir: str,
    out_postfix: str,
    n_jobs: int,
    trim: bool,
    trim_top_db: float,
):
    # Load the LJSpeech metadata
    df = pd.read_csv(
        path.join(speech_dir, "metadata.csv"),
        delimiter="|",
        quoting=csv.QUOTE_NONE,
        header=None,
    )
    df.columns = ["id", "text", "text_normalized"]

    if trim:
        try:
            os.makedirs(path.join(speech_dir, "wavs_trimmed"))
        except FileExistsError:
            print(
                "Warning: wavs_trimmed directory already exists, overwriting contents"
            )

    results = pqdm(
        df.iterrows(),
        partial(__do_preprocess, speech_dir, trim, trim_top_db),
        n_jobs=n_jobs,
        exception_behaviour="immediate",
    )
    results = [x for x in results if isinstance(x, dict)]

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        path.join(out_dir, f"ljspeech-{out_postfix}.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
