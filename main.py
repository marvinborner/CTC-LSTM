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
FEATURES = 13  # idk

print("getting datasets...")

# download the datasets
train_dataset = torchaudio.datasets.LIBRISPEECH(
    "./data", url="train-clean-100", download=True
)
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

    for audio in data:
        input = transform(audio[0]).squeeze(0).transpose(0, 1)
        inputs.append(input)
        target = torch.Tensor([CHARS.index(c) for c in audio[2]])
        targets.append(target)
        input_lengths.append(input.shape[0])
        target_lengths.append(len(target))

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return (inputs, targets, input_lengths, target_lengths)


print("preprocessing datasets...")

# load the datasets into batches
train_loader = data.DataLoader(
    train_dataset, batch_size=10, shuffle=True, collate_fn=preprocess
)
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


def ctc_decode(outputs, labels, label_lengths):
    arg_maxes = torch.argmax(outputs, dim=2)
    inferred = ""
    target = ""
    for i, args in enumerate(arg_maxes):
        target += "".join(
            [CHARS[int(c)] for c in labels[i][: label_lengths[i]].tolist()]
        )

        decode = []
        for j, ind in enumerate(args):
            if ind != len(CHARS):
                if j != 0 and ind == args[j - 1]:
                    continue
                decode.append(ind.item())
        inferred += "".join([CHARS[c] for c in decode])
    return inferred, target


def test():
    model.load_state_dict(torch.load("target/model-final.ckpt"))

    # TODO: Calculate accuracy using string difference functions
    with torch.no_grad():
        for i, (inputs, targets, input_lengths, target_lengths) in enumerate(
            test_loader
        ):
            outputs = model(inputs)
            inferred, target = ctc_decode(outputs, targets, target_lengths)
            print("\n=========================================\n")
            print("inferred: ", inferred)
            print("")
            print("target: ", target)


def usage():
    print(f"usage: {sys.argv[0]} <train|test>")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)

    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        usage()
