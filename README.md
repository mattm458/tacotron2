# Tacotron 2

This repository contains an implementation of Tacotron 2. It is heavily based on [Nvidia's implementation](https://github.com/NVIDIA/tacotron2), but with the following changes:

* Written with an up-to-date version of PyTorch
* Uses native PyTorch AMP functionality instead of Nvidia Apex
* Partially simplified and restructured code
* Uses Lightning instead of custom training loop code
* YAML configuration system for experiments

Additionally, the repository supports global style tokens. It includes training metadata for the following datasets:

* [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), for training a standard text-to-speech model
* [M-AILABS](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) recordings from Judy Bieber, for training a text-to-speech model with global style tokens

While it is still under development, the ultimate goal of this repository is to have a simple, easy-to-read, well-documented, and easily configurable testbed for a number of different speech synthesis tasks.