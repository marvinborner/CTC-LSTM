#!/bin/env python3

import torchaudio

# download the datasets
train_dataset = torchaudio.datasets.LIBRISPEECH(
    "./data", url="train-clean-100", download=True
)
test_dataset = torchaudio.datasets.LIBRISPEECH(
    "./data", url="test-clean", download=True
)
