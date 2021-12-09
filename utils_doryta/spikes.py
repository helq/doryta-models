from __future__ import annotations

import numpy as np
import struct

from typing import Any


def save_spikes_for_doryta(
    img: np.ndarray[Any, Any],
    times: np.ndarray[Any, Any],
    path: str
) -> None:
    """Saves sequence of spikes for each "image" (2nd dimension) at their given
    timestamps (times). Assumes spikes to be encoded as ones and not spikes as zero."""
    assert(len(img.shape) == 2)
    assert(len(times.shape) == 1)
    assert(img.shape[0] == times.shape[0])
    with open(f"{path}.bin", 'wb') as fh:
        # Magic number
        fh.write(struct.pack('>I', 0x23432BC5))
        fh.write(struct.pack('>H', 0x2))
        n_spikes_per_neuron = img.sum(axis=0)
        active_neurons = np.flatnonzero(n_spikes_per_neuron)
        # Total neurons
        fh.write(struct.pack('>i', active_neurons.shape[0]))
        # Total spikes
        fh.write(struct.pack('>i', img.sum()))
        # For each instant in time
        for neuron_i in active_neurons:
            # - Neuron id
            fh.write(struct.pack('>i', neuron_i))
            # - Number of spikes for neuron
            fh.write(struct.pack('>i', n_spikes_per_neuron[neuron_i]))
            # - Neuron times
            times[np.flatnonzero(img[:, neuron_i])].astype('>f4').tofile(fh)


if __name__ == '__main__':
    img = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ], dtype=int)

    times = np.array([0.1, 0.2, 0.52])
    save_spikes_for_doryta(img, times, "four-spikes")

    save_spikes_for_doryta(
        np.zeros((0, 5), dtype=int),
        np.zeros((0,)),
        "no-spikes")
