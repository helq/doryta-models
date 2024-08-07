from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
import os
import tarfile
import shutil
import pathlib

from .doryta_io.spikes import save_spikes_for_doryta


def insert_pattern(
    pattern: NDArray[np.int64],
    height: int = 20,
    width: int = 20,
) -> NDArray[np.int64]:
    assert len(pattern.shape) == 2
    assert pattern.shape[0] <= height
    assert pattern.shape[1] <= width
    img = np.zeros((1, height * width), dtype=np.int64)
    i_start = int(height / 2 - pattern.shape[0] / 2)
    j_start = int(width / 2 - pattern.shape[1] / 2)
    for i, row in enumerate(pattern):
        for j, val in enumerate(row):
            x = i_start + i
            y = j_start + j
            img[0, y + x * width] = val
    return img


die_hard = np.array(
    [[0, 0, 0, 0, 0, 0, 1, 0],
     [1, 1, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 1, 1, 1]]
)


# https://www.conwaylife.com/patterns/diehard2500.cells
die_hard_2500 = (np.frombuffer(
      b"OOOO...OOO.OOO...O..OO.O..O...OO"
    + b"OO...OOO..OOO.O.O....O.OO.O..OOO"
    + b".O.O.O...OOO..O...OOO.OOOOO....O"
    + b".......O..OOO.......OO.OO.OO.O.O"
    + b"...O.O.....OO.OOO..O.O.OO.O....."
    + b"OOOOO..O...OOOOO..OOO......O.OO."
    + b"..O....O.O..O...OO.OOOO.......O."
    + b".OOOO..OOO.O.O..O.OOOOOOOOO.O.OO"
    + b"OOOOO..O..O.O.O......O..O......O"
    + b"..O...OO.OOOO..O...O..O.OO...O.O"
    + b"OO.OO.OOO..OO.OO..OOO.OO....OOO."
    + b"O.OO.OOO..O...O.O.OO.OO.O.OOOOOO"
    + b"..O.OOOO.OOO.O..O....OOO.OOOO.OO"
    + b"..O..O.O..OO...OO..O....O.O....O"
    + b"O.....O.............OO.O..OO.OO."
    + b".O..O.OO...OOO.OO.O..OO...OO...."
    + b"....OO...OO..O.OO.OOO...OO.O..O."
    + b".OO.OO..O.OO.............O.....O"
    + b"O....O.O....O..OO...OO..O.O..O.."
    + b"OO.OOOO.OOO....O..O.OOO.OOOO.O.."
    + b"OOOOOO.O.OO.OO.O.O...O..OOO.OO.O"
    + b".OOO....OO.OOO..OO.OO..OOO.OO.OO"
    + b"O.O...OO.O..O...O..OOOO.OO...O.."
    + b"O......O..O......O.O.O..O..OOOOO"
    + b"OO.O.OOOOOOOOO.O..O.O.OOO..OOOO."
    + b".O.......OOOO.OO...O..O.O....O.."
    + b".OO.O......OOO..OOOOO...O..OOOOO"
    + b".....O.OO.O.O..OOO.OO.....O.O..."
    + b"O.O.OO.OO.OO.......OOO..O......."
    + b"O....OOOOO.OOO...O..OOO...O.O.O."
    + b"OOO..O.OO.O....O.O.OOO..OOO...OO"
    + b"OO...O..O.OO..O...OOO.OOO...OOOO",
    dtype=np.uint8).reshape((32, 32)) == ord('O')).astype(np.int64)


if __name__ == '__main__':
    save_20x20 = False
    save_100x100 = False
    save_128x128 = False
    save_1000x1000 = False
    save_1024x1024 = True

    path = pathlib.Path('gol/spikes/')

    if save_20x20:
        times = np.array([0.6])

        save_spikes_for_doryta(
            path / "20x20/gol-glider",
            insert_pattern(np.array(
                [[0, 0, 1],
                 [1, 0, 1],
                 [0, 1, 1]])),
            times)

        save_spikes_for_doryta(
            path / "20x20/gol-block",
            insert_pattern(np.array(
                [[1, 1],
                 [1, 1]]
            )), times,)

        save_spikes_for_doryta(path / "20x20/gol-blinker",
                               insert_pattern(np.array([[1, 1, 1]])), times)

        save_spikes_for_doryta(path / "20x20/gol-die-hard", insert_pattern(die_hard), times)

        # Saving random GoL 20x20 to tar.zst
        os.makedirs(path / "20x20/gol-random")
        np.random.seed(3287592)
        for i in range(1000):
            save_spikes_for_doryta(
                path / f"20x20/gol-random/gol-random-{i:04}",
                (np.random.rand(1, 400) < .2).astype(int),
                times)

        with tarfile.open(path / "20x20/gol-random.tar.xz", "w:xz") as tar:
            tar.add(path / '20x20/gol-random/')

        shutil.rmtree(path / '20x20/gol-random/')

    # 100 x 100 grid
    if save_100x100:
        times = np.array([0.6])

        seed = 3287590
        np.random.seed(seed)
        save_spikes_for_doryta(
            path / f"100x100/gol-random-{seed}",
            (np.random.rand(1, 100*100) < .2).astype(int),
            times)

        save_spikes_for_doryta(
            path / "100x100/gol-die-hard",
            insert_pattern(die_hard, width=100, height=100),
            times)

        save_spikes_for_doryta(
            path / "100x100/gol-die-hard-2500",
            insert_pattern(die_hard_2500, width=100, height=100),
            times)

    # 120 x 120 grid
    if save_128x128:
        times = np.array([0.6])

        save_spikes_for_doryta(
            path / "128x128/gol-die-hard-2500",
            insert_pattern(die_hard_2500, width=128, height=128),
            times)

    # 1000 x 1000 grid
    if save_1000x1000:
        times = np.array([0.6])
    
        save_spikes_for_doryta(
            path / "1000x1000/gol-glider",
            insert_pattern(np.array(
                [[0, 0, 1],
                 [1, 0, 1],
                 [0, 1, 1]]
            ), width=1000, height=1000),
            times)
    
        save_spikes_for_doryta(path / "1000x1000/gol-die-hard",
                               insert_pattern(die_hard, width=1000, height=1000),
                               times)
    
        seed = 3287591
        np.random.seed(seed)
        save_spikes_for_doryta(
            path / f"1000x1000/gol-random-{seed}",
            (np.random.rand(1, 1000*1000) < .2).astype(int),
            times)

    if save_1024x1024:
        times = np.array([0.6])

        save_spikes_for_doryta(
            path / "1024x1024/gol-die-hard-2500",
            insert_pattern(die_hard_2500, width=1024, height=1024),
            times)
