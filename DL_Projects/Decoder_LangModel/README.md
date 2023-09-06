# Landscape Recognition

Implemented transformer based language generation model from scratch with PyTorch using paper *Attention is all you need* to generate a sequence of **characters** given an input. The goal of this project was to generate English conversations in Shakespearean English after training on Shakespeare toy dataset. The project consists of two main scripts: `train.py` and `decoder_transformer.py` which contain the training procedure and model architecture.

## Training and Generation

To train the model and generate text, you can use the `train.py` script. This script allows you to specify the number of epochs and other hyperparameters, the dataset (which should be in .txt format) as well as the length of characters being generated.

```bash
python train.py
```

## Dataset

The dataset used for training is the Shakespear toy dataset [(Tiny Shakespeare)](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Model

The model architecture can be found in the `decoder_transformer.py` script which is a transformer decoder without cross attention.

---