import json
import time

import click
from click import Context

from run.say import do_say
from run.test import do_test
from run.train import do_train


@click.group()
@click.pass_context
@click.option(
    "--config",
    type=str,
    required=False,
    default=None,
    help="A Tacotron hyperparameter config file",
)
@click.option(
    "--device",
    type=int,
    required=False,
    default=0,
    help="The GPU to use for training or inference. Default 0.",
)
def main(ctx: Context, config: str, device: int):
    ctx.obj["config"] = None
    ctx.obj["device"] = device

    if config is not None:
        with open(config) as infile:
            config = json.load(infile)

        ctx.obj["config"] = config


@main.command()
@click.pass_context
@click.option(
    "--speech-dir",
    required=True,
    type=str,
    help="A directory containing audio files from the dataset.",
)
def train(ctx: Context, speech_dir: str):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for training!")

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
def say(ctx: Context, checkpoint: str, text: str, out: str):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for speech!")

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
def test(ctx: Context, speech_dir: str, checkpoint: str, hifi_gan_checkpoint: str):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for testing!")

    do_test(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        checkpoint=checkpoint,
        hifi_gan_checkpoint=hifi_gan_checkpoint,
    )


@main.command()
@click.option(
    "--dataset", required=True, type=str, help="The name of a dataset to preprocess."
)
@click.option(
    "--speech-dir",
    required=True,
    type=str,
    help="A directory containing audio files from the dataset.",
)
@click.option(
    "--out-dir",
    required=False,
    type=str,
    default="",
    help="A directory to save output CSV files. Defaults to the current directory",
)
@click.option(
    "--out-postfix",
    required=False,
    type=str,
    default=None,
    help="A postfix to add to CSV filenames. Defaults to the current timestamp.",
)
@click.option(
    "--n-jobs",
    required=False,
    type=int,
    default=8,
    help="The number of multiprocessing jobs to use while processing.",
)
def preprocess(
    dataset: str, speech_dir: str, out_dir: str, out_postfix: str, n_jobs: int
):
    if out_postfix is None:
        out_postfix = str(int(time.time()))

    if dataset == "hifi-tts":
        from preprocessing.hifi_tts import do_preprocess

        do_preprocess(speech_dir, out_dir, out_postfix, n_jobs)
    else:
        raise NotImplementedError(f"Preprocessing for {dataset} not implemented!")


if __name__ == "__main__":
    main(obj={})
