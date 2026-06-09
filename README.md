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
Using pre-trained weights, the model reaches roughly 90% accuracy on the
CIFAR-10 test set. The notebook also reports per-class precision, recall and
F1-score, plus a confusion matrix (the usual confusions being cat/dog and
automobile/truck).

## What the notebook covers

Loading and visualising the CIFAR-10 dataset
Normalising images and one-hot encoding labels
Building the All-CNN-C model in Keras
Defining a custom categorical cross-entropy loss function
Training from scratch, or loading pre-trained weights to skip the long train
Plotting loss and accuracy curves for the custom-loss training run
Making and visualising predictions
Evaluating with a classification report and confusion matrix

## Team Contributions

### Aniket Laha (MC25B1002) — Model Architecture

Designed and implemented the full All-CNN-C network in Keras as a reusable
allcnn() function.
Built the three convolutional blocks (filter progression 96 -> 192 -> 192) with
padding='same'.
Implemented the strided convolutions that replace max-pooling for
downsampling — the defining idea of All-CNN.
Implemented the 1×1 convolutions and the global average pooling + softmax
classification head that replace flatten-and-dense layers.
Added ReLU activations and Dropout (0.5) for regularisation.

### Kokkiripati Abhishek Kumar (MC25B1021) — Dataset Pipeline & Optimizer

Loaded the CIFAR-10 dataset and verified its characteristics
(50,000 train / 10,000 test, 32×32×3).
Implemented preprocessing: normalising pixels from 0–255 to 0–1 and
one-hot encoding the labels with to_categorical().
Set the random seed for reproducibility.
Configured the SGD optimizer with Nesterov momentum, weight decay, and the
learning rate of 0.01.

### Pushkar Patel (MC25B1018) — Loss Function & Model Training

Implemented the custom categorical cross-entropy loss function
(my_categorical_crossentropy), including the prediction clipping that prevents
log(0) producing NaN/-inf.
Wired the custom loss into the model compilation for both training paths.
Ran the 5-epoch training experiment with the custom loss and produced the
loss and accuracy curves.

### Ashish Kumar (MC25B1017) — Evaluation, Predictions & Documentation

Implemented the pre-trained weights path: loading the weights, evaluating on
the test set, and visualising the final metrics.
Built the prediction visualisation (3×3 grid of predicted vs actual labels).
Implemented the evaluation metrics: the classification report (per-class
precision, recall, F1) and the confusion-matrix heatmap.
Prepared the README and project documentation.

## Requirements

Python 3.x
TensorFlow / Keras
NumPy, Matplotlib, Pillow
scikit-learn, seaborn (for evaluation metrics)

See `requirements.txt` for the full list.

## Usage

Open `Object_Recognition_.ipynb` in Jupyter or Google Colab and run the cells
in order. To skip training, make sure the pre-trained weights file
(`all_cnn_weights_0.9088_0.4994.hdf5`) is in the same directory.

## Authors

Group 10 — Kokkiripati Abhishek Kumar, Pushkar Patel, Aniket Laha , Ashish Kumar
