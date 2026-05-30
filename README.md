# Object Recognition with All-CNN on CIFAR-10

A convolutional neural network for object recognition on the CIFAR-10 dataset,
implementing the **All-CNN-C** architecture from the 2015 ICLR paper
*"Striving for Simplicity: The All Convolutional Net"*
(Springenberg et al., https://arxiv.org/pdf/1412.6806.pdf).

The defining feature of this network is that it is built **entirely from
convolutional layers** — no max-pooling and no fully-connected layers.
Downsampling is handled by strided convolutions, and classification is done
with 1×1 convolutions followed by global average pooling.

## Dataset

CIFAR-10: 60,000 32×32 colour images across 10 classes
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck),
split into 50,000 training and 10,000 test images.

## Architecture

| Block | Layers | Output |
|-------|--------|--------|
| 1 | 3× conv (96 filters), last with stride 2 + dropout | 16×16×96 |
| 2 | 3× conv (192 filters), last with stride 2 + dropout | 8×8×192 |
| 3 | conv 3×3 (192), conv 1×1 (192), conv 1×1 (10) | 8×8×10 |
| Head | global average pooling + softmax | 10 probabilities |

~1.3 million trainable parameters.

## Results

Using pre-trained weights, the model reaches roughly **90% accuracy** on the
CIFAR-10 test set. The notebook also reports per-class precision, recall and
F1-score, plus a confusion matrix (the usual confusions being cat/dog and
automobile/truck).

## What the notebook covers

- Loading and visualising the CIFAR-10 dataset
- Normalising images and one-hot encoding labels
- Building the All-CNN-C model in Keras
- Training from scratch, or loading pre-trained weights to skip the ~10-hour train
- Making and visualising predictions
- Evaluating with a classification report and confusion matrix

## Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib, Pillow
- scikit-learn, seaborn (for evaluation metrics)

## Usage

Open `Object_Recognition_.ipynb` in Jupyter or Google Colab and run the cells
in order. To skip training, make sure the pre-trained weights file
(`all_cnn_weights_0.9088_0.4994.hdf5`) is in the same directory.

## Pre-trained weights

This project uses pre-trained All-CNN weights
(`all_cnn_weights_0.9088_0.4994.hdf5`) originally provided by
[PAN001's All-CNN repo](https://github.com/PAN001/All-CNN). Download the
file from there and place it in the project root before running the
evaluation cells. Alternatively, train the model from scratch by running
the training cell (takes several hours without a GPU).

## Authors

Group 10 — Kokkiripati Abhishek Kumar, Pushkar Patel, Aniket Laha , Ashish Kumar
