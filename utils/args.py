import argparse

parser = argparse.ArgumentParser(description="Train a Tacotron2 instance")

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="A YAML configuration file containing hyperparameters",
)

subparsers = parser.add_subparsers(required=True, dest="mode")

train_subparser = subparsers.add_parser("train", help="Train a Tacotron 2 model")

train_subparser.add_argument(
    "--dataset-dir",
    type=str,
    required=True,
    help="The base dataset directory",
)

train_subparser.add_argument(
    "--feature-extractor-checkpoint",
    type=str,
    required=False,
    default=None,
    help="Feature extractor checkpoint required for initial fine-tuning training",
)

train_subparser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    required=False,
    help="The path to a model checkpoint",
)

say_subparser = subparsers.add_parser(
    "say", help="Produce a WAV file from a given text string"
)

say_subparser.add_argument(
    "--text", required=True, type=str, default=None, help="The text to say"
)
say_subparser.add_argument(
    "--wav-out",
    type=str,
    required=False,
    default="out.wav",
    help="Where to save a generated wav file",
)
say_subparser.add_argument(
    "--with-speech-features",
    type=float,
    required=False,
    default=None,
    nargs="+",
    help="Speech features to include in inference",
)

say_subparser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to a model checkpoint",
)

args = parser.parse_args()
