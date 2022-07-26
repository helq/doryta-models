from __future__ import annotations

import pathlib

import numpy as np

from .doryta_io.circuit_saver import save
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits.prelude.base import byte_latch, two_bytes_RAM, RAM

dump_folder = pathlib.Path('snn-circuits/')

# One byte memory
if False and __name__ == '__main__':
    heartbeat = 1/8

    save(byte_latch(heartbeat),
         dump_folder / 'snn-models' / 'byte_latch.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    spikes = {
        # read
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
        dump_folder / 'spikes' / 'byte_latch',
        individual_spikes=spikes
    )

# Two bytes RAM
if False and __name__ == '__main__':
    heartbeat = 1/8

    # test with:
    # > src/doryta
    # > --load-model=../data/models/snn-circuits/snn-models/two_bytes_RAM.doryta.bin
    # > --load-spikes=../data/models/snn-circuits/spikes/two_bytes_RAM.bin --probe-firing
    # > --output-dir=testing-8-bit/two-byte-RAM --save-state --end=10

    save(two_bytes_RAM(heartbeat),
         dump_folder / 'snn-models' / 'two_bytes_RAM.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    spikes = {
        # read
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
        # addr
        10: np.array([]),
        # naddr
        11: np.array([1, 2, 3, 4]),
    }

    save_spikes_for_doryta(
        dump_folder / 'spikes' / 'two_bytes_RAM',
        individual_spikes=spikes
    )


# 16 bytes RAM
if True and __name__ == '__main__':
    heartbeat = 1/8

    # test with:
    # > src/doryta
    # > --load-model=../data/models/snn-circuits/snn-models/16_bytes_RAM.doryta.bin
    # > --load-spikes=../data/models/snn-circuits/spikes/16_bytes_RAM.bin --probe-firing
    # > --output-dir=testing-8-bit/16-byte-RAM --save-state --end=10

    save(RAM(heartbeat, 4),
         dump_folder / 'snn-models' / '16_bytes_RAM.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    # time   addr    instruction
    # 1      1011    WRITE   data: 10110000
    # 2      1011    READ
    # 3      1011    RESET
    # 4      1011    READ
    # 5      1010    WRITE   data: 10001100
    # 6      1010    READ
    spikes = {
        # read
        0: np.array([2, 4, 6]),
        # reset
        1: np.array([3]),
        # set 0-7 bits
        2: np.array([1, 5]),  # bit 0
        3: np.array([]),   # bit 1
        4: np.array([1]),  # bit 2
        5: np.array([1]),  # bit 3
        6: np.array([5]),   # bit 4
        7: np.array([5]),   # bit 5
        8: np.array([]),   # bit 6
        9: np.array([]),   # bit 7
        # addr 0-3 bits
        10: np.array([1, 2, 3, 4, 5, 6]),
        11: np.array([]),
        12: np.array([1, 2, 3, 4, 5, 6]),
        13: np.array([1, 2, 3, 4]),
        # naddr 0-3 bits
        14: np.array([]),
        15: np.array([1, 2, 3, 4, 5, 6]),
        16: np.array([]),
        17: np.array([5, 6]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / '16_bytes_RAM',
                           individual_spikes=spikes)
