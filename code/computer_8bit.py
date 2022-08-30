from __future__ import annotations

import pathlib

import numpy as np

from .doryta_io.circuit_saver import save
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits.prelude import base
from .circuits.visualize.svg import save_svg
from .circuits.visualize import positioning
from .circuits.sncircuit import SNCreateVisual

dump_folder = pathlib.Path('snn-circuits/')


# One byte memory
if False and __name__ == '__main__':
    heartbeat = 1/8

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/byte_latch.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/byte_latch.bin --probe-firing \
    # > --output-dir=testing-8-bit/byte_latch --save-state --end=20

    save(base.byte_latch(heartbeat).circuit,
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

    save(base.two_bytes_RAM(heartbeat).circuit,
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

    save(base.RAM(heartbeat, 4).circuit,
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

    save(base.half_adder(heartbeat).circuit,
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

    save(base.full_adder(heartbeat).circuit,
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

    save(base.multi_bit_adder(heartbeat, 2).circuit,
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
if False and __name__ == '__main__':
    heartbeat = 1/100

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/byte_adder.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/byte_adder.bin --probe-firing \
    # > --output-dir=testing-8-bit/byte_adder --save-state --end=10

    save(base.multi_bit_adder(heartbeat, 8).circuit,
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
    # 9        11111111 11111111     111111110
    spikes = {
        # input bits (first number)
        0: np.array([1, 4, 5, 6, 7, 8, 9]),
        1: np.array([2, 3, 5, 6, 7, 8, 9]),
        2: np.array([6, 7, 8, 9]),
        3: np.array([7, 8, 9]),
        4: np.array([7, 8, 9]),
        5: np.array([7, 8, 9]),
        6: np.array([7, 8, 9]),
        7: np.array([8, 9]),
        # input bits (second number)
        8: np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        9: np.array([3, 4, 5, 9]),
        10: np.array([9]),
        11: np.array([9]),
        12: np.array([9]),
        13: np.array([9]),
        14: np.array([9]),
        15: np.array([9]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'byte_adder',
                           individual_spikes=spikes)


# Testing one byte register and counter up
if False and __name__ == '__main__':
    heartbeat = 1/100

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/counter_register.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/counter_register.bin --probe-firing \
    # > --output-dir=testing-8-bit/counter_register --save-state --end=20

    save(base.counter_register(heartbeat).circuit,
         dump_folder / 'snn-models' / 'counter_register.doryta.bin',
         heartbeat=heartbeat, verbose=True)

    # Generating spikes
    # time   action              expected output
    # 1      SET      00001010
    # 2      COUNT_UP
    # 3      COUNT_UP
    # 4      READ                00001100
    # 5      READ                00001100
    # 6      RESET
    # 7      COUNT_UP
    # 8      COUNT_UP
    # 9      COUNT_UP
    # 10     COUNT_UP
    # 11     READ                00000100
    spikes = {
        # read
        0: np.array([4, 5, 11]),
        # reset
        1: np.array([6]),
        # set 0-7 bits
        2: np.array([]),  # bit 0
        3: np.array([1]),   # bit 1
        4: np.array([]),  # bit 2
        5: np.array([1]),  # bit 3
        6: np.array([]),   # bit 4
        7: np.array([]),   # bit 5
        8: np.array([]),   # bit 6
        9: np.array([]),   # bit 7
        # count up
        10: np.array([2, 3, 7, 8, 9, 10]),
    }

    save_spikes_for_doryta(
        dump_folder / 'spikes' / 'counter_register',
        individual_spikes=spikes
    )


if False and __name__ == '__main__':
    ht = 1/8
    byte_visual2 = base.multi_latch_visual(base.multi_latch(ht, 8), depth=0).generate()
    save_svg(byte_visual2, dump_folder / 'svgs' / "byte_latch.svg", 30)

    byte_visual = base.multi_latch_visual(base.multi_latch(ht, 8), depth=1).generate()
    save_svg(byte_visual, dump_folder / 'svgs' / "byte_latch-detailed.svg", 30)

    asr_visual = base.asr_latch_visual(base.asr_latch(ht)).generate()
    save_svg(asr_visual, dump_folder / 'svgs' / "asr_latch.svg", 30)

    save_svg(base.RAM_visual(base.RAM(ht, 1), byte_visual2).generate(),
             dump_folder / 'svgs' / "RAM.svg", 30, print_dummy=True)

    register_visual = base.counter_register_visual(base.counter_register(ht, 8), depth=1).generate()
    save_svg(register_visual, dump_folder / 'svgs' / "counter_register.svg", 30)

if False and __name__ == '__main__':
    ht = 1/8
    # Autogenerated testing
    fulladder_visual = SNCreateVisual(base.full_adder(ht)).generate(mode='auto')
    save_svg(fulladder_visual, dump_folder / 'svgs' / "fulladder-auto.svg", 30)

    graph_drawing = positioning.SugiyamaGraphDrawing(
        remove_cycles=positioning.RemoveCycleDFS(reverse=False),
        layer_assignment=positioning.LayerAssignmentCoffmanGraham(
            w=3, crossings_in_layer=1),
        reuse_dummy_nodes=True
    )
    fulladder_visual = SNCreateVisual(base.full_adder(ht), graph_drawing).generate(mode='auto')
    save_svg(fulladder_visual, dump_folder / 'svgs' / "fulladder-auto2.svg", 30,
             print_dummy=True)

    halfadder_visual = SNCreateVisual(base.half_adder(ht), graph_drawing).generate(mode='auto')
    save_svg(halfadder_visual, dump_folder / 'svgs' / "halfadder-auto.svg", 30)

    asr_visual = SNCreateVisual(base.asr_latch(ht), graph_drawing).generate(mode='auto')
    save_svg(asr_visual, dump_folder / 'svgs' / "asr_latch-auto.svg", 30)

if False and __name__ == '__main__':
    ht = 1/100
    size = 8
    bus, _ = base.bus(ht, size, output_pieces={'outs': 'both'})
    ram = base.RAM(ht, 4)

    graph_drawing = positioning.SugiyamaGraphDrawing(
        remove_cycles=positioning.RemoveCycleDFS(reverse=False),
        layer_assignment=positioning.LayerAssignmentCoffmanGraham(
            w=32, crossings_in_layer=0),
        reuse_dummy_nodes=True,
        bias_nodes=True,
        vertex_reordering=True
    )

    bus_snvisual = SNCreateVisual(bus, graph_drawing)
    bus_snvisual.set_graph_drawing_params(x_axis_zoom=1.5, input_output_sep=3.5)
    bus_visual = bus_snvisual.generate(mode='auto')
    save_svg(bus_visual, dump_folder / 'svgs' / "bus-auto.svg", 30, print_dummy=True)

    # This takes some time as it is huuuge! (like half a minute)
    ram_snvisual = SNCreateVisual(ram, graph_drawing)
    ram_snvisual.set_graph_drawing_params(input_output_sep=5.0)
    ram_visual = ram_snvisual.generate(mode='auto')
    save_svg(ram_visual, dump_folder / 'svgs' / "RAM-16-auto.svg", 30, print_dummy=True)


# Testing bus
if True and __name__ == '__main__':
    ht = 1/256
    size = 8

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/bus.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/bus.bin --probe-firing \
    # > --output-dir=testing-8-bit/bus --save-state --end=20

    # Notice that a real bus should be reset once it sends data forward on one of its
    # outputs. This is not a problem that I should care about right now
    bus, outputs = base.bus(ht, size, output_pieces={'ram': 'both', 'reg-a': 'q'})
    save(bus.circuit,
         dump_folder / 'snn-models' / 'bus.doryta.bin',
         heartbeat=ht, verbose=True)
    n_inputs = len(bus.circuit.inputs)
    print({k: [i + n_inputs for i in outs] for k, outs in outputs.items()})

    # Spikes to bus
    # Time   Inputs         ouputs
    # 1      SET 00100111
    # 2      READ RAM       11011000 00100111 (outputs 10-25)
    # 3      SET 00000000
    # 4      READ RAM       11111111 00000000 (outputs 10-25)
    # 5      SET 11001111
    # 6      READ REG_A     11001111          (outputs 26-33)
    spikes = {
        # read ram
        0: np.array([2, 4]),
        # read reg-a
        1: np.array([6]),
        # set byte
        2: np.array([1, 5]),
        3: np.array([1, 5]),
        4: np.array([1, 5]),
        5: np.array([5]),
        6: np.array([]),
        7: np.array([1]),
        8: np.array([5]),
        9: np.array([5]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'bus', individual_spikes=spikes)
