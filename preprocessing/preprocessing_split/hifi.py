import csv

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from normalize import (
    FEATURES_ALL,
    FEATURES_ALL_SPEAKER_NORM,
    FEATURES_ALL_SPEAKER_NORM_CLIP,
    FEATURES_ALL_DATASET_NORM,
    FEATURES_ALL_DATASET_NORM_CLIP,
    FEATURES_ALL_DATASET_GENDER_NORM,
    FEATURES_ALL_DATASET_GENDER_NORM_CLIP,
    normalize,
)

hifi_gender = {92: "f", 6097: "m", 9017: "m"}


@click.command()
@click.option(
    "--train-in", type=str, required=True, help="Path to the HiFi TTS training CSV"
)
@click.option(
    "--val-in", type=str, required=True, help="Path to the HiFi TTS validation CSV"
)
@click.option(
    "--test-in", type=str, required=True, help="Path to the HiFi TTS test CSV"
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
    "--speaker-val-size",
    type=int,
    required=False,
    default=100,
    help="Size of the validation set, per speaker",
)
@click.option(
    "--speaker-test-size",
    type=int,
    required=False,
    default=2000,
    help="Size of the test set, per speaker",
)
@click.option(
    "--random_state",
    type=int,
    required=False,
    default=9001,
    help="Random state for splits",
)
def main(
    train_in: str,
    val_in: str,
    test_in: str,
    train_out: str,
    val_out: str,
    test_out: str,
    speaker_val_size: int,
    speaker_test_size: int,
    random_state: int,
):
    df_train = pd.read_csv(train_in, delimiter="|", quoting=csv.QUOTE_NONE)
    df_val = pd.read_csv(val_in, delimiter="|", quoting=csv.QUOTE_NONE)
    df_test = pd.read_csv(test_in, delimiter="|", quoting=csv.QUOTE_NONE)

    # Assign gender annotations
    df_train["gender"] = [hifi_gender[i] for i in df_train.speaker_id_dataset]
    df_val["gender"] = [hifi_gender[i] for i in df_val.speaker_id_dataset]
    df_test["gender"] = [hifi_gender[i] for i in df_test.speaker_id_dataset]

    # Convert the training DataFrame into a dictionary mapping speaker ID
    # to a DataFrame containing only that speaker's rows
    df_train_split = {}
    for speaker_id, df_group in df_train.groupby("speaker_id"):
        df_train_split[speaker_id] = df_group

    # Add more validation and test instances from the training data
    df_val = fix_sizes(
        df_train_split,
        df=df_val,
        expected_size=speaker_val_size,
        random_state=random_state,
    )
    df_test = fix_sizes(
        df_train_split,
        df=df_test,
        expected_size=speaker_test_size,
        random_state=random_state,
    )

    # Dataset normalization
    df_train = pd.concat(df_train_split.values(), ignore_index=True)
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


# Correct the size of a particular split by taking rows from the training data
def fix_sizes(df_train_split, df, expected_size, random_state):
    new_df = []

    for speaker_id, group in df.groupby("speaker_id"):
        new_df.append(group)

        group_len_diff = expected_size - len(group)
        if group_len_diff == 0:
            continue

        df_train_split_new, df_group_new = train_test_split(
            df_train_split[speaker_id],
            test_size=group_len_diff,
            random_state=random_state,
        )

        df_train_split[speaker_id] = df_train_split_new
        new_df.append(df_group_new)

    return pd.concat(new_df, ignore_index=True)


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
