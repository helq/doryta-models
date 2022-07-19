from .doryta_io.spikes import save_spikes_for_doryta

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
    save_spikes_for_doryta("various/spikes/five-neurons-8-spikes", img, times)

    img = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ], dtype=int)
    times = np.array([0.1, 0.2, 0.52])
    save_spikes_for_doryta("various/spikes/five-neurons-4-spikes", img, times)

    save_spikes_for_doryta(
        "various/spikes/five-neurons-1-spike",
        np.array([[1, 0, 0, 0, 0]], dtype=int),
        np.array([0]))
