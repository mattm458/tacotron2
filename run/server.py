import asyncio
import os
import uuid
from typing import List

import aiohttp_jinja2
import jinja2
from aiohttp import web

_CONFIG = {
    "controls": [
        {"name": "Pitch", "val": "pitch"},
        {"name": "Pitch Range", "val": "pitch_range"},
        {"name": "Intensity", "val": "intensity"},
        {"name": "Noise-to-harmonics ratio", "val": "nhr"},
        {"name": "Syllable duration", "val": "rate"},
    ],
    "default_model": "vanilla-tacotron-3",
    "models": {
        "vanilla-tacotron-3": {
            "name": "Vanilla Tacotron (Speaker 3)",
            "multi_speaker": False,
            "controllable": False,
            "num_voices": 0,
            "config": "config/vanilla-ljspeech-stop.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/vanilla-tacotron-3.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-0": {
            "name": "Controllable Tacotron (Speaker 0)",
            "multi_speaker": False,
            "controllable": True,
            "num_voices": 0,
            "config": "config/controllable-lj-hifi-stop-speaker-0.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-0.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-1": {
            "name": "Controllable Tacotron (Speaker 1)",
            "multi_speaker": False,
            "controllable": True,
            "num_voices": 0,
            "config": "config/controllable-lj-hifi-stop-speaker-1.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-1.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-2": {
            "name": "Controllable Tacotron (Speaker 2)",
            "multi_speaker": False,
            "controllable": True,
            "num_voices": 0,
            "config": "config/controllable-lj-hifi-stop-speaker-2.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-2.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-3": {
            "name": "Controllable Tacotron (Speaker 3)",
            "multi_speaker": False,
            "controllable": True,
            "num_voices": 0,
            "config": "config/controllable-ljspeech-stop.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-3.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-multispeaker-speaker": {
            "name": "Controllable Multi-Speaker Tacotron, Speaker Normalized",
            "multi_speaker": True,
            "controllable": True,
            "num_voices": 4,
            "config": "config/controllable-lj-hifi-stop-speaker.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-multispeaker-speaker.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/controllable-lj-hifi-stop-speaker.ckpt",
        },
        "controllable-tacotron-multispeaker-gender": {
            "name": "Controllable Multi-Speaker Tacotron, Gender Normalized",
            "multi_speaker": True,
            "controllable": True,
            "num_voices": 4,
            "config": "config/controllable-lj-hifi-stop-gender.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-multispeaker-gender.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-multispeaker-gender-prosody": {
            "name": "Controllable Multi-Speaker Tacotron, Gender Normalized, Prosody Objective",
            "multi_speaker": True,
            "controllable": True,
            "num_voices": 4,
            "config": "config/controllable-lj-hifi-stop-gender-prosody-model.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-multispeaker-gender-prosody.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/controllable-lj-hifi-stop-gender-prosody-model_version_45.ckpt",
        },
        "controllable-tacotron-multispeaker-dataset": {
            "name": "Controllable Multi-Speaker Tacotron, Dataset Normalized",
            "multi_speaker": True,
            "controllable": True,
            "num_voices": 4,
            "config": "config/controllable-lj-hifi-stop-dataset.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-multispeaker-dataset.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
        "controllable-tacotron-multispeaker-dataset-prosody": {
            "name": "Controllable Multi-Speaker Tacotron, Dataset Normalized, Prosody Objective",
            "multi_speaker": True,
            "controllable": True,
            "num_voices": 4,
            "config": "config/controllable-lj-hifi-stop-dataset-prosody-model.json",
            "tacotron_checkpoint": "web_checkpoints/tacotron/controllable-tacotron-multispeaker-dataset-prosody.ckpt",
            "hifi_gan_checkpoint": "web_checkpoints/hifi-gan/hifi-gan-tamara-latest.ckpt",
        },
    },
}


_DEVICE = 0


async def config(request):
    return web.json_response(_CONFIG)


@aiohttp_jinja2.template("index.html")
async def index(request):
    return _CONFIG


async def generate(request):
    data = await request.json()
    config = _CONFIG["models"][data["model"]]

    filename = f"web_generated/{uuid.uuid4()}"

    filename_wav = filename + ".wav"
    filename_metadata = filename + ".json"

    text = data["text"].translate(
        str.maketrans(
            {
                "\\": r"\\",
                "!": r"\!",
                "'": r"\'",
            }
        )
    )

    random_seed = int(data["random_seed"])

    cmd_arr: List[str] = [
        "python main.py",
        f"--config {config['config']}",
        f"--device {_DEVICE}",
        "say",
        f"--text '{text}'",
        f"--checkpoint {config['tacotron_checkpoint']}",
        f"--out {filename_wav}",
        f"--random-seed {random_seed}",
    ]

    if data["vocoder"]:
        cmd_arr.append(f"--hifi-gan-checkpoint {config['hifi_gan_checkpoint']}")

    if config["controllable"]:
        print(data)
        controls: List[float] = [
            str(float(data[x["val"]])) for x in _CONFIG["controls"]
        ]
        cmd_arr.append(f"--controls {','.join(controls)}")

    if config["multi_speaker"]:
        speaker_id: int = int(data["speaker"])
        cmd_arr.append(f"--speaker-id {speaker_id}")

    cmd: str = " ".join(cmd_arr)

    print(cmd)

    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await proc.communicate()
    print(stdout)
    print(stderr)

    return web.json_response({"filename": "/" + filename_wav})


def do_server(port: int):
    os.makedirs("web_generated", exist_ok=True)

    app = web.Application()
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader("web"))
    app.add_routes(
        [
            web.get("/", index),
            web.get("/config", config),
            web.post("/generate", generate),
            web.static("/web_generated", "web_generated"),
        ]
    )

    web.run_app(app, port=port)
