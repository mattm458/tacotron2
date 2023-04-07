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
    required=True,
    help="A trained Tacotron checkpoint to fine-tune",
)

finetune_parser.add_argument(
    "--prosody-model-post-checkpoint",
    type=str,
    required=False,
    default=None,
    help="Post-HiFi-GAN model checkpoint required for fine-tuning",
)

finetune_parser.add_argument(
    "--prosody-model-checkpoint",
    type=str,
    required=False,
    default=None,
    help="Pre-vocoder model checkpoint required for fine-tuning",
)

finetune_parser.add_argument(
    "--hifi-gan-checkpoint",
    type=str,
    required=False,
    default=None,
    help="HiFi-GAN model checkpoint for finetuning",
)

finetune_parser.add_argument(
    "--resume-finetune",
    action=argparse.BooleanOptionalAction,
    help="Resume fine-tuning from a single checkpoint instead of loading new components separately",
)

finetune_parser.add_argument(
    "--fine-tune-freeze-tacotron",
    action=argparse.BooleanOptionalAction,
    help="During the fine-tuning process, freeze the Tacotron instance.",
)

finetune_parser.add_argument(
    "--fine-tune-tacotron",
    action=argparse.BooleanOptionalAction,
    help="Fine-tune Tacotron's style according to the prosody model.",
)

finetune_parser.add_argument(
    "--fine-tune-tacotron-limit",
    type=int,
    required=False,
    default=-1,
    help="Fine-tune Tacotron for a limited number of iterations.",
)

finetune_parser.add_argument(
    "--fine-tune-tacotron-style",
    action=argparse.BooleanOptionalAction,
    help="Fine-tune Tacotron's features according to the prosody model.",
)

finetune_parser.add_argument(
    "--fine-tune-tacotron-features",
    action=argparse.BooleanOptionalAction,
    help="Fine-tune Tacotron's features according to the prosody model.",
)

finetune_parser.add_argument(
    "--fine-tune-hifi-gan",
    action=argparse.BooleanOptionalAction,
    help="Fine-tune a pretrained HiFi-GAN model on Tacotron output.",
)

finetune_parser.add_argument(
    "--fine-tune-hifi-gan-style",
    action=argparse.BooleanOptionalAction,
    help="Fine-tune HiFi-GAN's style according to the prosody model.",
)

finetune_parser.add_argument(
    "--fine-tune-hifi-gan-features",
    action=argparse.BooleanOptionalAction,
    help="Fine-tune HiFi-GAN's features according to the prosody model.",
)

finetune_parser.add_argument(
    "--fine-tune-lr",
    type=float,
    required=False,
    default=None,
    help="The learning rate to use for fine-tuning",
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

say_subparser.add_argument(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The CUDA device to use for inference",
)

say_subparser.add_argument(
    "--checkpoint-has-hifi-gan",
    action=argparse.BooleanOptionalAction,
    help="Indicates whether the checkpoint contains a HiFi-GAN instance.",
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

test_subparser.add_argument(
    "--hifi-gan-checkpoint",
    type=str,
    required=False,
    help="A HiFi-GAN checkpoint",
)

args = parser.parse_args()
