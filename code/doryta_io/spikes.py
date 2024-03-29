from __future__ import annotations

import numpy as np
import struct
import pathlib

from typing import Any, Optional, Dict, Union
from numpy.typing import NDArray


def save_spikes_for_doryta(
    path: Union[str, pathlib.Path],
    img: Optional[np.ndarray[Any, Any]] = None,
    times: Optional[np.ndarray[Any, Any]] = None,
    individual_spikes: Optional[Dict[int, NDArray[Any]]] = None
) -> None:
    """Saves sequence of spikes for each "image" (2nd dimension) at their given
    timestamps (times, 1st dimension). Assumes spikes to be encoded as ones and not spikes
    as zero. Additional spikes must start after the last neuron in the images"""
    assert (img is None) == (times is None), "both, img and times, must be defined or None"
    if img is None:
        img = np.zeros((0, 0)).astype(int)
    if times is None:
        times = np.zeros((0,))
    assert len(img.shape) == 2
    assert len(times.shape) == 1
    assert img.shape[0] == times.shape[0]
    if individual_spikes is None:
        individual_spikes = {}
    assert all(neuron >= img.shape[1] for neuron in individual_spikes.keys())
    assert all(len(spikes.shape) == 1 for spikes in individual_spikes.values())

    with open(f"{path}.bin", 'wb') as fh:
        # Magic number
        fh.write(struct.pack('>I', 0x23432BC5))
        fh.write(struct.pack('>H', 0x2))
        n_spikes_per_neuron = img.sum(axis=0)
        active_neurons = np.flatnonzero(n_spikes_per_neuron)
        # Total neurons
        fh.write(struct.pack('>i', active_neurons.shape[0] + len(individual_spikes)))
        # Total spikes
        total_spikes_in_additional = sum(spikes.shape[0] for spikes in individual_spikes.values())
        fh.write(struct.pack('>i', img.sum() + total_spikes_in_additional))
        # For each instant in time
        for neuron_i in active_neurons:
            # - Neuron id
            fh.write(struct.pack('>i', neuron_i))
            # - Number of spikes for neuron
            fh.write(struct.pack('>i', n_spikes_per_neuron[neuron_i]))
            # - Neuron times
            times[np.flatnonzero(img[:, neuron_i])].astype('>f4').tofile(fh)
            # TODO: change `>f4` for `>f8`!! Double point precision is necessary for large
            # numbers
        for neuron_i, spikes in individual_spikes.items():
            fh.write(struct.pack('>i', neuron_i))
            fh.write(struct.pack('>i', spikes.shape[0]))
            spikes.astype('>f4').tofile(fh)


if __name__ == '__main__':
    img = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ], dtype=int)

    times = np.array([0.1, 0.2, 0.52])
    save_spikes_for_doryta("five-neurons-spikes", img, times)

    save_spikes_for_doryta(
        "no-spikes",
        np.zeros((0, 5), dtype=int),
        np.zeros((0,)))
