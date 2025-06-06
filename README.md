# OpenPi for Equinox

Inspired by: https://github.com/Physical-Intelligence/openpi.git

Motivation: Flax is hard to read and parse due to the stateful transforms hidden within code. Equinox offers a much more compelling abstraction that's python native and easy to understand.

Status:
- JAX/Equinox implementation of PaLI-Gemma 1/2, a multimodal language model from google combining PaLI and Gemma architectures: Tested and working
- Supports transfusion for efficient model adaptation: Implemented
- Vision Language Action modeling with OpenPI weights: Coming soon
- A library implementation to generate text: Coming soon
- A library implementation to generate actions: Coming soon
- Finetuning: not planned in this repo
- Running on actual robot: not planned in this repo

Usage:
See the main.py for example usage + decoding for the VLM
