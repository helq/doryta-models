from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
import os
import tarfile
import shutil
import pathlib

from .doryta_io.spikes import save_spikes_for_doryta


def insert_pattern(
    pattern: NDArray[int],
    height: int = 20,
    width: int = 20,
) -> NDArray[int]:
    assert len(pattern.shape) == 2
    assert pattern.shape[0] <= height
    assert pattern.shape[1] <= width
    img = np.zeros((1, height * width), dtype=int)  # type: ignore
    i_start = int(height / 2 - pattern.shape[0] / 2)
    j_start = int(width / 2 - pattern.shape[1] / 2)
    for i, row in enumerate(pattern):
        for j, val in enumerate(row):
            x = i_start + i
            y = j_start + j
            img[0, y + x * width] = val
    return img


if __name__ == '__main__':
    path = pathlib.Path('gol/spikes/')

if False and __name__ == '__main__':
    times = np.array([0.6])

    save_spikes_for_doryta(insert_pattern(np.array(
        [[0, 0, 1],
         [1, 0, 1],
         [0, 1, 1]]
    )), times, path / "20x20/gol-glider")

    save_spikes_for_doryta(insert_pattern(np.array(
        [[1, 1],
         [1, 1]]
    )), times, path / "20x20/gol-block")

    save_spikes_for_doryta(insert_pattern(np.array(
        [[1, 1, 1]]
    )), times, path / "20x20/gol-blinker")

    save_spikes_for_doryta(insert_pattern(np.array(
        [[0, 0, 0, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 1, 1]]
    )), times, path / "20x20/gol-die-hard")

    # Saving random GoL 20x20 to tar.zst
    os.makedirs(path / "20x20/gol-random")
    np.random.seed(3287592)
    for i in range(1000):
        save_spikes_for_doryta(
            (np.random.rand(1, 400) < .2).astype(int),
            times, path / f"20x20/gol-random/gol-random-{i:04}")

    with tarfile.open(path / "20x20/gol-random.tar.xz", "w:xz") as tar:
        tar.add(path / '20x20/gol-random/')

    shutil.rmtree(path / '20x20/gol-random/')


# 100 x 100 grid
if True and __name__ == '__main__':
    times = np.array([0.6])  # type: ignore

    seed = 3287590
    np.random.seed(seed)
    save_spikes_for_doryta(
        (np.random.rand(1, 100*100) < .2).astype(int),
        times, f"path / 100x100/gol-random-{seed}")


# 1000 x 1000 grid
if False and __name__ == '__main__':
    times = np.array([0.6])

    save_spikes_for_doryta(
        insert_pattern(np.array(
            [[0, 0, 1],
             [1, 0, 1],
             [0, 1, 1]]
        ), width=1000, height=1000),
        times, "path / 1000x1000/gol-glider")

    save_spikes_for_doryta(
        insert_pattern(np.array(
            [[0, 0, 0, 0, 0, 0, 1, 0],
             [1, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 1, 1]]
        ), width=1000, height=1000),
        times, "path / 1000x1000/gol-die-hard")

    seed = 3287591
    np.random.seed(seed)
    save_spikes_for_doryta(
        (np.random.rand(1, 1000*1000) < .2).astype(int),
        times, f"path / 1000x1000/gol-random-{seed}")
