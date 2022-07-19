from __future__ import annotations

import numpy as np

from numpy.typing import NDArray
from typing import List, Any, Tuple, Dict


def img_to_tempencoding(
    imgs: NDArray[Any],
    cuts: List[float],
    position_trigger_neuron: int
) -> Tuple[NDArray[Any], NDArray[Any], Dict[int, NDArray[Any]]]:
    assert len(imgs.shape) == 2
    assert imgs.shape[1] == 28 * 28
    # img_size = imgs.shape[1]
    n_gray = len(cuts)
    total_images = imgs.shape[0]

    grayscales = []
    for j in range(len(cuts) - 1):
        img0 = np.bitwise_and(cuts[j] < imgs, imgs <= cuts[j+1])
        grayscales.append(img0)
    img = cuts[-1] < imgs
    grayscales.append(img)

    spikes = np.stack(grayscales, axis=1).astype(int)  # type: ignore
    # padding = position_trigger_neuron - img_size + 1
    # spikes = np.pad(spikes, ((0, 0), (0, 0), (0, padding)))
    # spikes[:, -1, -1] = 1  # this is the trigger neuron
    spikes = spikes.reshape((-1, spikes.shape[2]))

    times = (np.array(range(n_gray), dtype=float) + 1) / 256 - 1/512
    # times *= 10  # Separating each spike to allow them show the exponential behaviour
    times = np.concatenate([times + i for i in range(total_images)])

    trigger_times = times[np.arange(n_gray-1, spikes.shape[0], n_gray)]
    individual_spikes = {
        position_trigger_neuron: trigger_times
    }
    assert len(individual_spikes[position_trigger_neuron]) == total_images

    return spikes, times, individual_spikes
