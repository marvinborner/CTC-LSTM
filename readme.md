# CTC LSTM

> spoken word recognition using CTC LSTMs

## Instructions

- Create a virtual environment: `python -m venv venv`
- Install the required packages:
  `./venv/bin/pip install -r requirements.txt`
- Train the model: `./venv/bin/python main.py train` (takes a few hours
  and needs around 20GB disk and 5GB memory)
  - or download my pre-trained model (25 epochs, **not good**) from
    [here](https://marvinborner.de/model-final.ckpt) and move it to
    `target/model-final.ckpt`
- Test the final model: `./venv/bin/python main.py test`
- Infer text from flac: `./venv/bin/python main.py infer audio.flac`

## Note

- This is a proof-of-concept
- Does not use CUDA but should be easy to implement
