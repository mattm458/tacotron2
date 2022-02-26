import argparse

parser = argparse.ArgumentParser(description="Train a Tacotron2 instance")

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="A YAML configuration file containing hyperparameters",
)
parser.add_argument(
    "--dataset-dir",
    type=str,
    required=False,
    default=None,
    help="The base dataset directory",
)
parser.add_argument(
    "--inference",
    type=bool,
    required=False,
    default=False,
    help="Set to True to conduct inference instead of training",
)
parser.add_argument(
    "--train",
    type=bool,
    required=False,
    default=True,
    help="Set to True to conduct training",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    required=False,
    help="The path to a model checkpoint",
)
parser.add_argument(
    "--say", required=False, type=str, default=None, help="The text to say"
)
parser.add_argument(
    "--wav-out",
    type=str,
    default=None,
    required=False,
    help="Where to save a generated wav file",
)
parser.add_argument(
    "--with-speech-features",
    type=float,
    required=None,
    default=None,
    nargs="+",
    help="Speech features to include in inference",
)

args = parser.parse_args()
