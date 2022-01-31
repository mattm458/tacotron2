# Tacotron 2

This repository contains an implementation of Tacotron 2. It is heavily based on [Nvidia's implementation](https://github.com/NVIDIA/tacotron2), but with the following changes:

* Written with an up-to-date version of PyTorch
* Uses native PyTorch AMP functionality instead of Nvidia Apex
* Partially simplified and restructured code
* Uses Lightning instead of custom training loop code
* YAML configuration system for experiments

While it is still under development, the ultimate goal of this repository is to have a simple, easy-to-read, well-documented, and easily configurable testbed for a number of different speech synthesis tasks. It probably doesn't work yet and is missing some dataset files, so use at your own risk.