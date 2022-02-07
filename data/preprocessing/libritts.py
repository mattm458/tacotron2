import csv
import os
import sys
from os import path

import pandas as pd

if len(sys.argv) != 3:
    print()
    print("Usage: python libritts.py <LibriTTS directory> <CSV output directory>")
    exit()

_, DATASET_DIR, OUT_DIR = sys.argv
datasets = ["dev-clean", "test-clean", "train-clean-100"]

for dataset in datasets:
    wav_files = []
    speaker_ids = []
    normalized_txts = []

    for speaker_id in os.listdir(path.join(DATASET_DIR, dataset)):
        if not speaker_id.isnumeric():
            continue

        for chapter_id in os.listdir(path.join(DATASET_DIR, dataset, speaker_id)):
            if not chapter_id.isnumeric():
                continue

            for id in (
                x.replace(".wav", "")
                for x in os.listdir(
                    path.join(DATASET_DIR, dataset, speaker_id, chapter_id)
                )
                if ".wav" in x
            ):
                wav_file = path.join(dataset, speaker_id, chapter_id, f"{id}.wav")
                normalized_txt = open(
                    path.join(
                        DATASET_DIR,
                        dataset,
                        speaker_id,
                        chapter_id,
                        f"{id}.normalized.txt",
                    )
                ).read()

                wav_files.append(wav_file)
                speaker_ids.append(speaker_id)
                normalized_txts.append(normalized_txt)

    pd.DataFrame([wav_files, speaker_ids, normalized_txts]).T.to_csv(
        path.join(OUT_DIR, f"libritts-{dataset}.csv"),
        header=None,
        sep="|",
        quoting=csv.QUOTE_NONE,
    )
