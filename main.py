import json

import click

from run.say import do_say
from run.test import do_test
from run.train import do_train


@click.group()
@click.pass_context
@click.option(
    "--config", type=str, required=True, help="A Tacotron hyperparameter config file"
)
@click.option(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The GPU to use for training or inference. Default 0.",
)
def main(ctx, config, device):
    with open(config) as infile:
        config = json.load(infile)

    ctx.obj["config"] = config
    ctx.obj["device"] = device
    pass


@main.command()
@click.pass_context
@click.option(
    "--speech-dir",
    required=True,
    type=str,
    help="A directory containing audio files from the dataset.",
)
def train(ctx, speech_dir):
    do_train(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        device=ctx.obj["device"],
        speech_dir=speech_dir,
    )


@main.command()
@click.pass_context
@click.option(
    "--checkpoint", required=True, type=str, help="A trained Tacotron model checkpoint"
)
@click.option("--text", required=True, type=str, help="Text to speak")
@click.option(
    "--out",
    required=False,
    type=str,
    default="out.wav",
    help="Name of a .wav file to output. Default: out.wav",
)
def say(ctx, checkpoint, text, out):
    do_say(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        device=ctx.obj["device"],
        checkpoint=checkpoint,
        text=text,
        output=out,
    )


@main.command()
@click.pass_context
@click.option(
    "--speech-dir",
    required=True,
    type=str,
    help="A directory containing audio files from the dataset.",
)
@click.option(
    "--checkpoint", required=True, type=str, help="A trained Tacotron model checkpoint"
)
@click.option(
    "--hifi-gan-checkpoint",
    required=False,
    type=str,
    help="A trained HiFi-GAN model checkpoint",
)
def test(ctx, speech_dir, checkpoint, hifi_gan_checkpoint):
    do_test(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        checkpoint=checkpoint,
        hifi_gan_checkpoint=hifi_gan_checkpoint,
    )


if __name__ == "__main__":
    main(obj={})
