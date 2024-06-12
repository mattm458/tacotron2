import json
import time
from typing import Optional

import click
from click import Context

from run.say import do_say
from run.test import do_test
from run.test_correlation import do_test_correlation
from run.train import do_train
from run.train_mel_export import do_train_mel_export


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
@click.option(
    "--results-dir",
    required=False,
    type=str,
    help="The directory to save results. Defaults to the model configuration name with a timestamp.",
)
@click.option(
    "--resume-ckpt",
    required=False,
    type=str,
    help="Resume training from the given checkpoint.",
)
@click.option(
    "--prosody-model-checkpoint",
    required=False,
    type=str,
    help="A prosody model checkpoint. If specified, the model will be used as an objective in the second half of training.",
)
def train(
    ctx: Context,
    speech_dir: str,
    results_dir: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
    prosody_model_checkpoint: Optional[str] = None,
):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for training!")

    do_train(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        extensions_config=ctx.obj["config"]["extensions"],
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        results_dir=results_dir,
        resume_ckpt=resume_ckpt,
        #prosody_model_checkpoint=prosody_model_checkpoint,
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
    "--checkpoint",
    required=True,
    type=str,
    help="Resume training from the given checkpoint.",
)
@click.option(
    "--results-dir",
    required=False,
    type=str,
    help="The directory to save results. Defaults to the model configuration name with a timestamp.",
)
def train_mel_export(
    ctx: Context,
    speech_dir: str,
    checkpoint: str,
    results_dir: Optional[str] = None,
):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required!")

    do_train_mel_export(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        extensions_config=ctx.obj["config"]["extensions"],
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        checkpoint=checkpoint,
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
@click.option(
    "--hifi-gan-checkpoint",
    required=False,
    type=str,
    default=None,
    help="A trained HiFi-GAN model checkpoint",
)
@click.option(
    "--random-seed",
    required=False,
    type=int,
    default=None,
    help="A random seed to use in generation. If not given, a seed will be randomly chosen.",
)
@click.option(
    "--speaker-id",
    required=False,
    type=int,
    default=None,
    help="A speaker ID to use in inference if using a multi-speaker model",
)
@click.option(
    "--controls",
    required=False,
    type=str,
    default=None,
    help="If controls are enabled, a comma-separated list of values to pass into the model. Defaults to all 0 values.",
)
@click.option(
    "--description",
    required=False,
    type=str,
    default=None,
    help="If descriptions are enabled, a textual description for how the text is to be spoken.",
)
def say(
    ctx: Context,
    checkpoint: str,
    text: str,
    out: str,
    speaker_id: Optional[int],
    hifi_gan_checkpoint: Optional[str],
    random_seed: Optional[int],
    controls: Optional[str],
    description: Optional[str],
):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for speech!")

    do_say(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        extensions_config=ctx.obj["config"]["extensions"],
        device=ctx.obj["device"],
        checkpoint=checkpoint,
        text=text,
        output=out,
        speaker_id=speaker_id,
        hifi_gan_checkpoint=hifi_gan_checkpoint,
        random_seed=random_seed,
        controls=controls,
        description=description,
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
    default=None,
    help="A trained HiFi-GAN model checkpoint",
)
def test_correlation(
    ctx: Context, speech_dir: str, checkpoint: str, hifi_gan_checkpoint: str
):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for testing!")

    do_test_correlation(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        extensions_config=ctx.obj["config"]["extensions"],
        device=ctx.obj["device"],
        speech_dir=speech_dir,
        checkpoint=checkpoint,
        hifi_gan_checkpoint=hifi_gan_checkpoint,
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
    default=None,
    help="A trained HiFi-GAN model checkpoint",
)
def test(ctx: Context, speech_dir: str, checkpoint: str, hifi_gan_checkpoint: str):
    if ctx.obj["config"] is None:
        raise Exception("Configuration required for testing!")

    do_test(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        extensions_config=ctx.obj["config"]["extensions"],
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
@click.option(
    "--trim",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to trim the audio files during preprocessing.",
)
@click.option(
    "--trim-top-db",
    required=False,
    show_default=True,
    type=float,
    default=60,
    help="If trimming, The threshold (in decibels) below reference to consider as silence.",
)
def preprocess(
    dataset: str,
    speech_dir: str,
    out_dir: str,
    out_postfix: str,
    n_jobs: int,
    trim: bool,
    trim_top_db: float,
):
    if out_postfix is None:
        out_postfix = str(int(time.time()))

    if dataset == "hifi-tts":
        from preprocessing.hifi_tts import do_preprocess

        do_preprocess(
            speech_dir,
            out_dir,
            out_postfix,
            n_jobs,
            trim,
            trim_top_db,
        )
    elif dataset == "ljspeech":
        from preprocessing.ljspeech import do_preprocess

        do_preprocess(
            speech_dir,
            out_dir,
            out_postfix,
            n_jobs,
            trim,
            trim_top_db,
        )
    else:
        raise NotImplementedError(f"Preprocessing for {dataset} not implemented!")


@main.command()
@click.pass_context
@click.option("--port", required=False, type=int, default=8080, help="The server port.")
def server(ctx: Context, port: int):
    if ctx.obj["config"] is None:
        raise Exception("A server configuration is required!")

    from run.server import do_server

    do_server(port)


if __name__ == "__main__":
    main(obj={})
