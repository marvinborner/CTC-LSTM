#!/bin/env python3

import sys

import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
EPOCHS = 25
FEATURES = 15  # idk


def usage():
    print(f"usage: {sys.argv[0]} <train|test|infer> [file]")
    sys.exit(1)


if __name__ != "__main__":
    sys.exit(1)  # this isn't a library bro

if len(sys.argv) < 2 or sys.argv[1] not in ["train", "test", "infer"]:
    usage()

MODE = sys.argv[1]

print("getting datasets...")

# download the datasets
if MODE == "train":
    train_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url="train-clean-100", download=True
    )
if MODE == "test":
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url="test-clean", download=True
    )

print("got datasets!")


def preprocess(data):
    transform = T.MFCC(sample_rate=16000, n_mfcc=FEATURES)

    inputs = []
    targets = []
    input_lengths = []
    target_lengths = []

    for wav, sr, label, _, _, _ in data:
        if sr != 16000:
            resample = T.Resample(orig_freq=sr, new_freq=16000)
            wav = resample(wav)
        input = transform(wav).squeeze(0).transpose(0, 1)
        inputs.append(input)
        target = torch.Tensor([CHARS.index(c) for c in label])
        targets.append(target)
        input_lengths.append(input.shape[0])
        target_lengths.append(len(target))

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return (inputs, targets, input_lengths, target_lengths)


print("preprocessing datasets...")

# load the datasets into batches
if MODE == "train":
    train_loader = data.DataLoader(
        train_dataset, batch_size=10, shuffle=True, collate_fn=preprocess
    )
if MODE == "test":
    test_loader = data.DataLoader(
        test_dataset, batch_size=10, shuffle=True, collate_fn=preprocess
    )

print("datasets ready!")


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


model = Model(input_size=FEATURES, hidden_size=128, output_size=len(CHARS) + 1)


def train():
    print("training model...")

    criterion = nn.CTCLoss(blank=len(CHARS))
    optimizer = optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(
            train_loader
        ):
            print(f"epoch {epoch}/{EPOCHS}, iteration {i}/{len(train_loader)}")
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = F.log_softmax(outputs, dim=2)
            outputs = outputs.transpose(0, 1)

            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"target/model-{epoch}.ckpt")

    torch.save(model.state_dict(), "target/model-final.ckpt")
    print("model trained!")


def ctc_decode(outputs):
    arg_maxes = torch.argmax(outputs, dim=2)
    inferred = ""
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, ind in enumerate(args):
            if ind != len(CHARS):
                if j != 0 and ind == args[j - 1]:
                    continue
                decode.append(ind.item())
        inferred += "".join([CHARS[c] for c in decode])
    return inferred


def ctc_decode_target(labels, label_lengths):
    target = ""
    for i, args in enumerate(labels):
        target += "".join(
            [CHARS[int(c)] for c in labels[i][: label_lengths[i]].tolist()]
        )
    return target


def test():
    model.load_state_dict(torch.load("target/model-final.ckpt"))

    # TODO: Calculate accuracy using string difference functions
    with torch.no_grad():
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(
            test_loader
        ):
            outputs = model(inputs)
            inferred = ctc_decode(outputs)
            target = ctc_decode_target(targets, target_lengths)
            print("\n=========================================\n")
            print("inferred: ", inferred)
            print("")
            print("target: ", target)


# TODO: Might be buggy?
def infer(file_):
    model.load_state_dict(torch.load("target/model-final.ckpt"))
    wav, sr = torchaudio.load(file_)
    inputs = preprocess([(wav, sr, "")])[0]
    with torch.no_grad():
        outputs = model(inputs)
        inferred = ctc_decode(outputs)
        print("inferred: ", inferred)


if MODE == "train":
    train()
elif MODE == "test":
    test()
elif MODE == "infer" and len(sys.argv) == 3:
    infer(sys.argv[2])
else:
    usage()
