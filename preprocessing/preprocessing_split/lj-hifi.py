import csv
from os import path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from normalize import (
    FEATURES_ALL,
    FEATURES_ALL_DATASET_GENDER_NORM,
    FEATURES_ALL_DATASET_GENDER_NORM_CLIP,
    FEATURES_ALL_DATASET_NORM,
    FEATURES_ALL_DATASET_NORM_CLIP,
    FEATURES_ALL_SPEAKER_NORM,
    FEATURES_ALL_SPEAKER_NORM_CLIP,
    normalize,
)


@click.command()
@click.option(
    "--hifi-train-in", type=str, required=True, help="Path to the HiFi TTS training CSV"
)
@click.option(
    "--hifi-val-in", type=str, required=True, help="Path to the HiFi TTS validation CSV"
)
@click.option(
    "--hifi-test-in", type=str, required=True, help="Path to the HiFi TTS test CSV"
)
@click.option(
    "--lj-train-in", type=str, required=True, help="Path to a LJSpeech training CSV"
)
@click.option(
    "--lj-val-in", type=str, required=True, help="Path to a LJSpeech validation CSV"
)
@click.option(
    "--lj-test-in", type=str, required=True, help="Path to a LJSpeech TTS test CSV"
)
@click.option(
    "--train-out", type=str, required=True, help="Path to save the output training CSV"
)
@click.option(
    "--val-out", type=str, required=True, help="Path to save the output validation CSV"
)
@click.option(
    "--test-out", type=str, required=True, help="Path to save the output test CSV"
)
@click.option(
    "--hifi-dir",
    type=str,
    required=False,
    default="hi_fi_tts_v0",
    help="The name of the HiFi-TTS parent directory",
)
@click.option(
    "--lj-dir",
    type=str,
    required=False,
    default="LJSpeech-1.1",
    help="The name of the LJSpeech parent directory",
)
def main(
    hifi_train_in: str,
    hifi_val_in: str,
    hifi_test_in: str,
    lj_train_in: str,
    lj_val_in: str,
    lj_test_in: str,
    train_out: str,
    val_out: str,
    test_out: str,
    hifi_dir: str,
    lj_dir: str,
):
    df_hifi_train = pd.read_csv(hifi_train_in, delimiter="|", quoting=csv.QUOTE_NONE)
    df_hifi_val = pd.read_csv(hifi_val_in, delimiter="|", quoting=csv.QUOTE_NONE)
    df_hifi_test = pd.read_csv(hifi_test_in, delimiter="|", quoting=csv.QUOTE_NONE)

    df_hifi_train.wav = [path.join(hifi_dir, x) for x in df_hifi_train.wav]
    df_hifi_val.wav = [path.join(hifi_dir, x) for x in df_hifi_val.wav]
    df_hifi_test.wav = [path.join(hifi_dir, x) for x in df_hifi_test.wav]

    df_lj_train = pd.read_csv(lj_train_in, delimiter="|", quoting=csv.QUOTE_NONE)
    df_lj_val = pd.read_csv(lj_val_in, delimiter="|", quoting=csv.QUOTE_NONE)
    df_lj_test = pd.read_csv(lj_test_in, delimiter="|", quoting=csv.QUOTE_NONE)

    df_lj_train.wav = [path.join(lj_dir, x) for x in df_lj_train.wav]
    df_lj_val.wav = [path.join(lj_dir, x) for x in df_lj_val.wav]
    df_lj_test.wav = [path.join(lj_dir, x) for x in df_lj_test.wav]

    # Adding gender and speaker ID annotation to LJSpeech
    df_lj_train["gender"] = "f"
    df_lj_val["gender"] = "f"
    df_lj_test["gender"] = "f"
    lj_speaker_id = df_hifi_train.speaker_id.max() + 1
    df_lj_train["speaker_id"] = lj_speaker_id
    df_lj_val["speaker_id"] = lj_speaker_id
    df_lj_test["speaker_id"] = lj_speaker_id

    # Sanity checking on the input files - do we have enough validation/test data in HiFi-TTS?
    for speaker_id, i in df_hifi_val.groupby("speaker_id"):
        if len(i) < len(df_lj_val):
            raise Exception(
                f"Speaker {speaker_id} in HiFi-TTS val has {len(i)} instances, which is fewer than LJSpeech's {len(df_lj_val)} instances"
            )
    for speaker_id, i in df_hifi_test.groupby("speaker_id"):
        if len(i) < len(df_lj_test):
            raise Exception(
                f"Speaker {speaker_id} in HiFi-TTS test has {len(i)} instances, which is fewer than LJSpeech's {len(df_lj_test)} instances"
            )

    # Combine the datasets
    df_train = pd.concat([df_hifi_train, df_lj_train], ignore_index=True)
    df_val = pd.concat([df_hifi_val, df_lj_val], ignore_index=True)
    df_test = pd.concat([df_hifi_test, df_lj_test], ignore_index=True)

    # Dataset normalization
    medians_all = df_train[FEATURES_ALL].median()
    stds_all = df_train[FEATURES_ALL].std()

    do_norm(
        df_train,
        feature_medians=medians_all,
        feature_stds=stds_all,
        F=FEATURES_ALL_DATASET_NORM,
        F_CLIP=FEATURES_ALL_DATASET_NORM_CLIP,
    )
    do_norm(
        df_val,
        feature_medians=medians_all,
        feature_stds=stds_all,
        F=FEATURES_ALL_DATASET_NORM,
        F_CLIP=FEATURES_ALL_DATASET_NORM_CLIP,
    )
    do_norm(
        df_test,
        feature_medians=medians_all,
        feature_stds=stds_all,
        F=FEATURES_ALL_DATASET_NORM,
        F_CLIP=FEATURES_ALL_DATASET_NORM_CLIP,
    )

    # Speaker normalization
    df_train = do_norm_by(
        df=df_train,
        df_train=df_train,
        F=FEATURES_ALL_SPEAKER_NORM,
        F_CLIP=FEATURES_ALL_SPEAKER_NORM_CLIP,
        by="speaker_id",
    )
    df_val = do_norm_by(
        df=df_val,
        df_train=df_train,
        F=FEATURES_ALL_SPEAKER_NORM,
        F_CLIP=FEATURES_ALL_SPEAKER_NORM_CLIP,
        by="speaker_id",
    )
    df_test = do_norm_by(
        df=df_test,
        df_train=df_train,
        F=FEATURES_ALL_SPEAKER_NORM,
        F_CLIP=FEATURES_ALL_SPEAKER_NORM_CLIP,
        by="speaker_id",
    )

    # Gender normalization
    df_train = do_norm_by(
        df=df_train,
        df_train=df_train,
        F=FEATURES_ALL_DATASET_GENDER_NORM,
        F_CLIP=FEATURES_ALL_DATASET_GENDER_NORM_CLIP,
        by="gender",
    )
    df_val = do_norm_by(
        df=df_val,
        df_train=df_train,
        F=FEATURES_ALL_DATASET_GENDER_NORM,
        F_CLIP=FEATURES_ALL_DATASET_GENDER_NORM_CLIP,
        by="gender",
    )
    df_test = do_norm_by(
        df=df_test,
        df_train=df_train,
        F=FEATURES_ALL_DATASET_GENDER_NORM,
        F_CLIP=FEATURES_ALL_DATASET_GENDER_NORM_CLIP,
        by="gender",
    )

    df_train.to_csv(train_out, sep="|", quoting=csv.QUOTE_NONE, index=None)
    df_val.to_csv(val_out, sep="|", quoting=csv.QUOTE_NONE, index=None)
    df_test.to_csv(test_out, sep="|", quoting=csv.QUOTE_NONE, index=None)


def do_norm_by(df, df_train, F, F_CLIP, by):
    medians = {}
    stds = {}

    # Get the medians and standard deviations from the training data
    for split_key, group in df_train.groupby(by):
        medians[split_key] = group[FEATURES_ALL].median()
        stds[split_key] = group[FEATURES_ALL].std()

    # Split the incoming DataFrame by the split key, then apply
    # normalization on each group
    df_new = []
    for split_key, group in df.groupby(by):
        group = group.copy()
        do_norm(
            df=group,
            feature_medians=medians[split_key],
            feature_stds=stds[split_key],
            F=F,
            F_CLIP=F_CLIP,
        )
        df_new.append(group)

    return pd.concat(df_new, ignore_index=True)


def do_norm(df, feature_medians, feature_stds, F, F_CLIP):
    df[F] = normalize(df[FEATURES_ALL], medians=feature_medians, stds=feature_stds)
    df[F_CLIP] = df[F].clip(-1, 1)


if __name__ == "__main__":
    main()
