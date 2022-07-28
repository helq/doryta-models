from __future__ import annotations

import pathlib

import numpy as np

from .doryta_io.circuit_saver import save
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits.prelude.base import byte_latch, two_bytes_RAM, RAM, half_adder, \
    full_adder, multi_bit_adder

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
if False and __name__ == '__main__':
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


# Testing half adder
if False and __name__ == '__main__':
    heartbeat = 1/8

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/half_adder.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/half_adder.bin --probe-firing \
    # > --output-dir=testing-8-bit/half_adder --save-state --end=10

    save(half_adder(heartbeat),
         dump_folder / 'snn-models' / 'half_adder.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    # time     bits    output (+ and carry)
    # 1        11      0 1
    # 2        01      1 0
    # 3        10      1 0
    spikes = {
        # input bits
        0: np.array([1, 3]),
        1: np.array([1, 2]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'half_adder',
                           individual_spikes=spikes)


# Testing full adder
if False and __name__ == '__main__':
    heartbeat = 1/8

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/full_adder.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/full_adder.bin --probe-firing \
    # > --output-dir=testing-8-bit/full_adder --save-state --end=10

    save(full_adder(heartbeat),
         dump_folder / 'snn-models' / 'full_adder.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    # time     bits    output (+ and carry)
    # 1        111     1 1
    # 2        011     0 1
    # 3        001     1 0
    spikes = {
        # input bits
        0: np.array([1]),
        1: np.array([1, 2]),
        2: np.array([1, 2, 3]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'full_adder',
                           individual_spikes=spikes)


# Testing two bit adder
if False and __name__ == '__main__':
    heartbeat = 1/100

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/two_bit_adder.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/two_bit_adder.bin --probe-firing \
    # > --output-dir=testing-8-bit/two_bit_adder --save-state --end=10

    save(multi_bit_adder(heartbeat, 2),
         dump_folder / 'snn-models' / 'two_bit_adder.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    # time     bits      output (carry and addition)
    # 1        01 01     010
    # 2        10 01     011
    # 3        10 11     101
    # 4        01 11     100
    # 5        11 11     110
    spikes = {
        # input bits (first number)
        0: np.array([1, 4, 5]),
        1: np.array([2, 3, 5]),
        # input bits (second number)
        2: np.array([1, 2, 3, 4, 5]),
        3: np.array([3, 4, 5]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'two_bit_adder',
                           individual_spikes=spikes)


# Testing byte adder
if True and __name__ == '__main__':
    heartbeat = 1/100

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/byte_adder.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/byte_adder.bin --probe-firing \
    # > --output-dir=testing-8-bit/byte_adder --save-state --end=10

    save(multi_bit_adder(heartbeat, 8),
         dump_folder / 'snn-models' / 'byte_adder.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    # time     bits                  output (carry and addition)
    # 1        00000001 00000001     000000010
    # 2        00000010 00000001     000000011
    # 3        00000010 00000011     000000101
    # 4        00000001 00000011     000000100
    # 5        00000011 00000011     000000110
    # 6        00000111 00000001     000001000
    # 7        01111111 00000001     010000000
    # 8        11111111 00000001     100000000
    spikes = {
        # input bits (first number)
        0: np.array([1, 4, 5, 6, 7, 8]),
        1: np.array([2, 3, 5, 6, 7, 8]),
        2: np.array([6, 7, 8]),
        3: np.array([7, 8]),
        4: np.array([7, 8]),
        5: np.array([7, 8]),
        6: np.array([7, 8]),
        7: np.array([8]),
        # input bits (second number)
        8: np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        9: np.array([3, 4, 5]),
        10: np.array([]),
        11: np.array([]),
        12: np.array([]),
        13: np.array([]),
        14: np.array([]),
        15: np.array([]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'byte_adder',
                           individual_spikes=spikes)
