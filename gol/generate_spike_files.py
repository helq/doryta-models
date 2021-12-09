from __future__ import annotations
from generic_code.spikes import save_spikes_for_doryta

import numpy as np
from numpy.typing import NDArray


def insert_pattern(
    pattern: NDArray[int],
    height: int = 20,
    width: int = 20,
) -> NDArray[int]:
    assert len(pattern.shape) == 2
    assert pattern.shape[0] <= height
    assert pattern.shape[1] <= width
    img = np.zeros((1, height * width), dtype=int)
    i_start = int(height / 2 - pattern.shape[0] / 2)
    j_start = int(width / 2 - pattern.shape[1] / 2)
    for i, row in enumerate(pattern):
        for j, val in enumerate(row):
            x = i_start + i
            y = j_start + j
            img[0, x + y * width] = 1 if val else 0
    return img


if __name__ == '__main__':
    times = np.array([0.6])

    save_spikes_for_doryta(insert_pattern(np.array(
        [[0, 0, 1],
         [1, 0, 1],
         [0, 1, 1]]
    )), times, "gol-glider")

    save_spikes_for_doryta(insert_pattern(np.array(
        [[1, 1],
         [1, 1]]
    )), times, "gol-block")

    save_spikes_for_doryta(insert_pattern(np.array(
        [[1, 1, 1]]
    )), times, "gol-blinker")

    save_spikes_for_doryta(insert_pattern(np.array(
        [[0, 0, 0, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 1, 1]]
    )), times, "gol-die-hard")
