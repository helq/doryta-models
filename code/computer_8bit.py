from __future__ import annotations

import pathlib

import numpy as np

from .doryta_io.model_saver import ModelSaverLayers
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits.prelude.base import byte_latch

if __name__ == '__main__':
    dump_folder = pathlib.Path('snn-circuits/')
    heartbeat = 1/8

    msaver = ModelSaverLayers(dt=heartbeat)
    # Layer 0: activate and reset
    msaver.add_neuron_group(thresholds=np.array([0.8, 0.8]), partitions=2)
    # Layer 1: set 0-7 bits
    msaver.add_neuron_group(thresholds=np.array(np.ones(8) * 0.8), partitions=8)

    input_layers: list[tuple[int, int] | int]
    input_layers = [(0, 0), (0, 1)] + [(1, i) for i in range(8)]  # type: ignore

    byte_latch_ = byte_latch(heartbeat)
    msaver.add_sncircuit_layer(byte_latch_, input_layers)
    msaver.save(dump_folder / 'snn-models' / 'byte_latch.doryta.bin')

    # Generating spikes
    spikes = {
        # activate
        0: np.array([2, 4]),
        # reset
        1: np.array([3]),
        # set 0-7 bits
        2: np.array([1]),
        3: np.array([]),
        4: np.array([1]),
        5: np.array([1]),
        6: np.array([]),
        7: np.array([]),
        8: np.array([]),
        9: np.array([]),
    }

    save_spikes_for_doryta(
        None, None,
        dump_folder / 'spikes' / 'byte_latch',
        additional_spikes=spikes
    )

    print("Output neurons are:", [10 + i for out in byte_latch_.outputs
                                  for i in out])
