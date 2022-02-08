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

# TODO: do train-clean-300 when you can get the data
datasets = ["dev-clean", "test-clean", "train-clean-100"]

all_speaker_ids = set()

durations = pd.read_csv(path.join(OUT_DIR, 'libritts-durations.csv'))
durations = durations[durations.duration <= 10]
durations = set(durations.wav)

for dataset in datasets:
    wav_files = []
    speaker_ids = []
    normalized_txts = []

    # LibriTTS datasets are organized into directories by speaker ID, then by chapter ID
    for speaker_id in os.listdir(path.join(DATASET_DIR, dataset)):
        all_speaker_ids.add(speaker_id)

        if not speaker_id.isnumeric():
            continue

        for chapter_id in os.listdir(path.join(DATASET_DIR, dataset, speaker_id)):
            if not chapter_id.isnumeric():
                continue

            # Flat file structure has wav and transcript files separately, with the same ID in the filename
            # Get all the recording IDs and iterate over them to load the WAV path and normalized transcript
            for id in (
                x.replace(".wav", "")
                for x in os.listdir(
                    path.join(DATASET_DIR, dataset, speaker_id, chapter_id)
                )
                if ".wav" in x
            ):
                wav_file = path.join(dataset, speaker_id, chapter_id, f"{id}.wav")

                if wav_file not in durations:
                    continue

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

    pd.DataFrame(
        zip(wav_files, speaker_ids, normalized_txts),
        columns=["wav", "speaker_id", "text_normalized"],
    ).to_csv(
        path.join(OUT_DIR, f"libritts-{dataset}.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=False,
    )

with open(path.join(OUT_DIR, "libritts-speaker-ids.csv"), "w") as outfile:
    outfile.writelines(f"{x}\n" for x in all_speaker_ids)
