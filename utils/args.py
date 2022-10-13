import argparse

parser = argparse.ArgumentParser(description="Train a Tacotron2 instance")

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="A YAML configuration file containing hyperparameters",
)

subparsers = parser.add_subparsers(required=True, dest="mode")

torchscript_subparser = subparsers.add_parser(
    "torchscript", help="Export the model to TorchScript"
)

torchscript_subparser.add_argument(
    "--filename",
    type=str,
    required=False,
    help="The TorchScript model filename",
    default="tacotron.pt",
)

torchscript_subparser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to a model checkpoint",
)

torchscript_subparser.add_argument(
    "--hifi-gan-checkpoint",
    help="Load a HiFi-GAN checkpoint to use in generating audio",
    required=False,
    type=str,
    default=None,
)

train_subparser = subparsers.add_parser("train", help="Train a Tacotron 2 model")

train_subparser.add_argument(
    "--dataset-dir",
    type=str,
    required=True,
    help="The base dataset directory",
)

train_subparser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    required=False,
    help="Resume training from the given checkpoint",
)

finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a Tacotron 2 model")

finetune_parser.add_argument(
    "--dataset-dir",
    type=str,
    required=True,
    help="The base dataset directory",
)

finetune_parser.add_argument(
    "--tacotron-checkpoint",
    type=str,
    default=None,
    required=False,
    help="A trained Tacotron checkpoint to fine-tune",
)

finetune_parser.add_argument(
    "--prosody-model-checkpoint",
    type=str,
    required=False,
    default=None,
    help="Prosody model checkpoint required for fine-tuning",
)

finetune_parser.add_argument(
    "--finetune-prosody-model",
    type=bool,
    required=False,
    default=None,
    help="Fine-tune the supplied prosody model",
)

finetune_parser.add_argument(
    "--finetune-tacotron-style",
    type=bool,
    required=False,
    default=None,
    help="Fine-tune Tacotron's style according to the prosody model.",
)

finetune_parser.add_argument(
    "--finetune-tacotron-feature",
    type=bool,
    required=False,
    default=None,
    help="Fine-tune Tacotron's features according to the prosody model.",
)


say_subparser = subparsers.add_parser(
    "say", help="Produce a WAV file from a given text string"
)

say_subparser.add_argument(
    "--hifi-gan-checkpoint",
    help="Load a HiFi-GAN checkpoint to use in generating audio",
    required=False,
    type=str,
    default=None,
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


test_subparser = subparsers.add_parser(
    "test", help="Produce WAV files from the test set"
)

test_subparser.add_argument(
    "--dir-out",
    type=str,
    required=True,
    help="The directory to save generated wav files. If it does not exist, it will be created.",
)

test_subparser.add_argument(
    "--with-speech-features",
    type=float,
    required=False,
    default=None,
    nargs="+",
    help="Speech features to include in inference",
)

test_subparser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to a model checkpoint",
)

test_subparser.add_argument(
    "--dataset-dir",
    type=str,
    required=True,
    help="The base dataset directory",
)

args = parser.parse_args()
