from __future__ import annotations

import pathlib

import numpy as np

from .doryta_io.circuit_saver import save
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits.prelude.base import byte_latch

if __name__ == '__main__':
    dump_folder = pathlib.Path('snn-circuits/')
    heartbeat = 1/8

    byte_latch_ = byte_latch(heartbeat)
    save(byte_latch_, dump_folder / 'snn-models' / 'byte_latch.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    spikes = {
        # activate
        0: np.array([2, 4]),
        # reset
        1: np.array([3]),
        # set 0-7 bits
        2: np.array([1]),  # bit 0
        3: np.array([]),   # bit 1
        4: np.array([1]),  # bit 2
        5: np.array([1]),  # bit 3
        6: np.array([]),   # bit 4
        7: np.array([]),   # bit 5
        8: np.array([]),   # bit 6
        9: np.array([]),   # bit 7
    }

    save_spikes_for_doryta(
        None, None,
        dump_folder / 'spikes' / 'byte_latch',
        additional_spikes=spikes
    )
