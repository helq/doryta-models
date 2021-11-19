from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def my_key() -> np.ndarray:  # type: ignore
    key = np.zeros((10, 100))
    for i in range(10):
        key[i, i*10: (i+1)*10] = 1
    return key


def plot_img(img: np.ndarray) -> None:  # type: ignore
    plt.imshow(img.reshape((28, 28)))
    plt.show()
