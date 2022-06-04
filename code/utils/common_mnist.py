from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as datasets
from tensorflow.keras.utils import to_categorical
import pathlib

from typing import Tuple


keras_model_path = pathlib.Path('mnist/raw_keras_models/')
doryta_model_path = pathlib.Path('mnist/snn-models/')


def my_key() -> np.ndarray:  # type: ignore
    key = np.zeros((10, 100))
    for i in range(10):
        key[i, i*10: (i+1)*10] = 1
    return key


def plot_img(img: np.ndarray) -> None:  # type: ignore
    plt.imshow(img.reshape((28, 28)), cmap='Greys')
    plt.show()


def load_data(
    dataset: str
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:  # type: ignore
    # Loading and preprocessing data
    numClasses = 10
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    elif dataset == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    else:
        raise Exception(f"Unrecognized {dataset} to load")

    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    y_train = to_categorical(y_train, numClasses)
    y_test = to_categorical(y_test, numClasses)

    x_train = np.reshape(x_train, (60000, 28*28))
    x_test = np.reshape(x_test, (10000, 28*28))
    return (x_train, y_train), (x_test, y_test)
