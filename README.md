# Tacotron 2 with experimental textual descriptions

## Introduction

This repository contains a custom implementation of Tacotron for TTS research. We are experimenting with conditioning the model's output on BERT embeddings created from textual descriptions of output speech obtained from various sources. It was used as part of the research conducted for the master's thesis [Controlling Emotional Text to Speech Using Complex Adverbial Phrases](https://academicworks.cuny.edu/gc_etds/6027/).

This is a research repository and not for production use. Architecturally, it could be possible to train as prompted TTS, though we have not tried this, and no weights are available.

The Tacotron 2 backbone is heavily based on [Nvidia's implementation](https://github.com/NVIDIA/tacotron2).

Curently, the model contains references to prior versions, when it was being used to experiment with controllable prosodic features.

## Installation

This code works with Python 3.13. It is possible earlier versions could work, but it has not been tested.

Create a virtual environment, activate it, and install dependencies:

```console
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run:

```console
python main.py
```

## Commands

The model supports several commands. Most cases involve the following two commands:

| Command               | Description                          |
| --------------------- | ------------------------------------ |
| `python main.py say`  | Produce a WAV file of the given text |
| `pthon main.py train` | Train with a given configuration     |

All commands require some parameters to be given. Please see `python main.py --help` for more details about these parameters.

### Say

The `say` command allows you to generate a WAV file from pretrained model weights. The command supports several parameters. Those relevant to textual description research are given below:

| Parameter       | Description                                                                        |
| --------------- | ---------------------------------------------------------------------------------- |
| `--checkpoint`  | The path to a model checkpoint. These are automatically saved during training.     |
| `--text`        | The text to speak aloud.                                                           |
| `--out`         | The name of a WAV file to output. Defaults to `out.wav`                            |
| `--random-seed` | An optional integer value for reproducibility.                                     |
| `--speaker-id`  | The integer value of a speaker ID to use if the model is trained as multi-speaker. |
| `--description` | A textual description given as additional input to the model.                      |

Example:

```console
python main.py --config config/model-config.json --device 0 say --checkpoint checkpoint-1.ckpt --text "Hello, world." --out "hello.wav" --random-seed 99 --speaker-id 12 --description "happy and excited"
```
