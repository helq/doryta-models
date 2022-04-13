from .utils.doryta.spikes import save_spikes_for_doryta

import numpy as np

if __name__ == '__main__':
    img = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
    ], dtype=int)
    times = np.array([0.1, 0.2, 0.52, 0.521, 0.9])
    save_spikes_for_doryta(img, times, "various/spikes/five-neurons-8-spikes")

    img = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ], dtype=int)
    times = np.array([0.1, 0.2, 0.52])
    save_spikes_for_doryta(img, times, "various/spikes/five-neurons-4-spikes")

    save_spikes_for_doryta(
        np.array([[1, 0, 0, 0, 0]], dtype=int),
        np.array([0]),
        "various/spikes/five-neurons-1-spike")
