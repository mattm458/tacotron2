import csv

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from normalize import (
    FEATURES_ALL,
    FEATURES_ALL_SPEAKER_NORM,
    FEATURES_ALL_SPEAKER_NORM_CLIP,
    normalize,
)


def do_norm(df, feature_medians, feature_stds):
    df[FEATURES_ALL_SPEAKER_NORM] = normalize(
        df[FEATURES_ALL], medians=feature_medians, stds=feature_stds
    )
    df[FEATURES_ALL_SPEAKER_NORM_CLIP] = df[FEATURES_ALL_SPEAKER_NORM].clip(-1, 1)


@click.command()
@click.option("--csv-in", type=str, required=True, help="Path to the LJSpeech CSV")
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
    "--val-size",
    type=int,
    required=False,
    default=100,
    help="Size of the validation set",
)
@click.option(
    "--test-size", type=int, required=False, default=2000, help="Size of the test set"
)
@click.option(
    "--random_state",
    type=int,
    required=False,
    default=9001,
    help="Random state for splits",
)
def main(
    csv_in: str,
    train_out: str,
    val_out: str,
    test_out: str,
    val_size: int,
    test_size: int,
    random_state: int,
):
    df = pd.read_csv(csv_in, delimiter="|", quoting=csv.QUOTE_NONE)

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    df_train, df_val = train_test_split(
        df_train, test_size=val_size, random_state=random_state
    )

    assert (len(df_train) + len(df_test) + len(df_val)) == len(
        df
    ), "Error: Size of dataset splits are not equal to the size of the original DataFrame!"

    feature_medians = df_train[FEATURES_ALL].median()
    feature_stds = df_train[FEATURES_ALL].std()

    do_norm(df_train, feature_medians, feature_stds)
    do_norm(df_val, feature_medians, feature_stds)
    do_norm(df_test, feature_medians, feature_stds)

    df_train.to_csv(train_out, sep="|", index=None, quoting=csv.QUOTE_NONE)
    df_val.to_csv(val_out, sep="|", index=None, quoting=csv.QUOTE_NONE)
    df_test.to_csv(test_out, sep="|", index=None, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    main()
