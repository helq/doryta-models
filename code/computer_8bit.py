from __future__ import annotations

import pathlib
from typing import Any
import sys

import numpy as np

from .doryta_io.circuit_saver import save
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits.prelude import base
from .circuits.visualize.svg import save_svg
from .circuits.visualize import positioning
from .circuits.snvisualize import SNCreateVisual
from .circuits import sncircuit

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
    heartbeat = 1/256

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
if False and __name__ == '__main__':
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


# Testing ALU, register A and B
if False and __name__ == '__main__':
    ht = 1/256
    n_bits = 8

    # test with:
    # > src/doryta \
    # > --load-model=../data/models/snn-circuits/snn-models/gluing_ALU.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/gluing_ALU.bin --probe-firing \
    # > --output-dir=testing-8-bit/gluing_ALU --save-state --end=20

    glued_ALU = base.glued_ALU(ht, n_bits)

    # print(snc.circuit)
    save(glued_ALU.circuit,
         dump_folder / 'snn-models' / 'gluing_ALU.doryta.bin',
         heartbeat=ht, verbose=True)

    # Spike sequence
    # time    instruction           output
    # 1       LOAD A  00100111
    # 2       READ A                00100111 (outputs 21-28)
    # 3       LOAD B  00100010
    # 4       ACTIVATE ALU
    # 5       READ FLAGBIT          0 (output 29)
    # 6       READ A                01001001 (outputs 21-28)
    # 7       LOAD B  10110110
    # 8       ACTIVATE ALU
    # 9       READ FLAGBIT          0 (output 29)
    # 10      READ A                11111111 (outputs 21-28)
    # 11      LOAD B  00000001
    # 12      ACTIVATE ALU
    # 13      READ FLAGBIT          1 (output 29)
    # 14      READ A                00000000 (outputs 21-28)
    spikes = {
        # activate ALU
        0: np.array([4, 8, 12]),
        # Register A
        #   read/send BUS
        1: np.array([2, 6, 10, 14]),
        #   reset
        2: np.array([]),
        #   set byte
        3: np.array([1]),
        4: np.array([1]),
        5: np.array([1]),
        6: np.array([]),
        7: np.array([]),
        8: np.array([1]),
        9: np.array([]),
        10: np.array([]),
        # Register B
        #   reset
        11: np.array([7, 11]),
        #   set byte
        12: np.array([11]) + ht,
        13: np.array([3, 7]) + ht,
        14: np.array([7]) + ht,
        15: np.array([]),
        16: np.array([7]) + ht,
        17: np.array([3, 7]) + ht,
        18: np.array([]),
        19: np.array([7]) + ht,
        # read flag bit
        20: np.array([5, 9, 13])
    }

    # save_svg(gluing_visual.generate(), dump_folder / 'svgs' / "gluing_ALU.svg", 20,
    #          print_dummy=True)
    save_spikes_for_doryta(dump_folder / 'spikes' / 'gluing_ALU', individual_spikes=spikes)


# Testing all components (except for CPU) by hand
if False and __name__ == '__main__':
    # test with:
    # > src/doryta --spike-driven \
    # > --load-model=../data/models/snn-circuits/snn-models/all-glued-but-cpu.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/all-glued-but-cpu.bin --probe-firing \
    # > --output-dir=testing-8-bit/all-glued-but-cpu --save-state --end=20

    ht = 1/256
    n_bits = 8
    ram_addr_bits = 4

    all_glued_but_CPU = base.all_glued_but_CPU(ht, n_bits, ram_addr_bits)

    save(all_glued_but_CPU.circuit.remove_unneded_neurons(),
         dump_folder / 'snn-models' / 'all-glued-but-cpu.doryta.bin',
         heartbeat=ht, verbose=True)

    # creating spikes
    # time  operation+data                    output
    #            1. Load value to counter (from BUS)
    #            2. Increase counter
    #            3. Output counter value (using BUS)
    # 1     WRITE 000000 00001011 (in bus)
    # 1.5   ACTIVATE   bus-reg-counter
    # 2     ACTIVATE   counter-count-up
    # 3     ACTIVATE   counter-read
    # 3.5   ACTIVATE   bus-output-circuit     0000 00001100
    # 3.8   RESET      counter
    #            4. load data to RAM (two bytes, addr 04 and 07)
    # 4     WRITE 000100 01000001 (in bus)
    # 4.2   ACTIVATE   bus-ram
    # 4.4   WRITE 000111 11101001 (in bus)
    # 4.6   ACTIVATE   bus-ram
    #            5. move data addr 04 to register A
    # 5     WRITE 010100 00000000 (in bus)
    # 5.2   ACTIVATE   bus-ram
    # 5.4   ACTIVATE   bus-reg-a
    #            6. move data from addr 07 to register B
    # 6     WRITE 010111 00000000 (in bus)
    # 6.2   ACTIVATE   bus-ram
    # 6.4   ACTIVATE   bus-reg-b
    #            7. activate glued_ALU
    # 7     ACTIVATE   glued_alu-alu
    #            8. copy register A to output
    # 8     ACTIVATE   glued_alu-regA-bus
    # 8.5   ACTIVATE   bus-output-circuit  (expected output 000000 00101010)
    spikes = {
        # BUS
        # bus-ram
        0: np.array([4.2, 4.6, 5.2, 6.2]),
        # bus-reg-a
        1: np.array([5.4]),
        # bus-reg-b
        2: np.array([6.4]),
        # bus-reg-counter
        3: np.array([1.5]),
        # bus-cpu
        4: np.array([]),
        # bus-output-circuit
        5: np.array([3.5, 8.5]),
        # bus-set 0 to 7
        6: np.array([1, 4, 4.4]),
        7: np.array([1]),
        8: np.array([]),
        9: np.array([1, 4.4]),
        10: np.array([]),
        11: np.array([4.4]),
        12: np.array([4, 4.4]),
        13: np.array([4.4]),
        # bus-set 8 to 11
        14: np.array([4.4, 6]),
        15: np.array([4.4, 6]),
        16: np.array([1, 4, 4.4, 5, 6]),
        17: np.array([]),
        18: np.array([5, 6]),
        19: np.array([]),
        # Counter
        # counter-read
        20: np.array([3]),
        # counter-reset
        21: np.array([3.8]),
        # counter-count-up
        22: np.array([2]),
        # Glued_ALU
        # glued_alu-alu
        23: np.array([7]),
        # glued_alu-regA-bus
        24: np.array([8]),
        # glued_alu-regA-reset
        25: np.array([]),
        # glued_alu-regB-reset
        26: np.array([]),
        # glued_alu-flag-read
        27: np.array([19]),
    }

    save_spikes_for_doryta(dump_folder / 'spikes' / 'all-glued-but-cpu',
                           individual_spikes=spikes)


if False and __name__ == '__main__':
    # test with:
    # > src/doryta --spike-driven \
    # > --load-model=../data/models/snn-circuits/snn-models/all-glued-but-cpu.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/all-glued-but-cpu.bin --probe-firing \
    # > --output-dir=testing-8-bit/all-glued-but-cpu --save-state --end=20

    ht = 1/256
    n_bits = 8
    ram_addr_bits = 4

    all_glued_but_CPU = base.all_glued_but_CPU(ht, n_bits, ram_addr_bits)

    graph_drawing = positioning.SugiyamaGraphDrawing(
        remove_cycles=positioning.RemoveCycleDFS(reverse=False),
        layer_assignment=positioning.LayerAssignmentCoffmanGraham(
            w=100, crossings_in_layer=0),
        reuse_dummy_nodes=True,
        bias_nodes=True,
        vertex_reordering=True
    )

    with sncircuit.SNCreate(neuron_type=sncircuit.LIF, neuron_params={},
                            synapse_params={}) as snc:
        for inp in all_glued_but_CPU.circuit.inputs_id:
            snc.input(inp, inputs=[f"Glued.{inp}"])
        for out_i in range(len(all_glued_but_CPU.circuit.outputs)):
            snc.output(f"Glued.out_{out_i}")
        snc.include("Glued", all_glued_but_CPU.circuit)

    glued_snvisual = SNCreateVisual(snc, graph_drawing)
    # glued_snvisual.set_graph_drawing_params(x_axis_zoom=1.5, input_output_sep=3.5)
    glued_visual = glued_snvisual.generate(mode='auto')
    save_svg(glued_visual, dump_folder / 'svgs' / "all_glued_but_CPU-auto.svg", 30,
             print_dummy=True)


# Testing control unit
if False and __name__ == '__main__':
    ht = 1/256
    control_unit = base.control_unit(ht)


def content(mnemonic: str, data: int = 0) -> int:
    mn_table = {
        'LDA': 0b0000,  # loads into register A
        'ADD': 0b0001,  # adds number to register A
        # 'SUB': 0b0010,
        'CLR': 0b0011,  # clears address
        'OUT': 0b1110,  # outputs content of register A
        'HLT': 0b1111,  # stops execution
        'STA': 0b1010,  # copy register A to address
        'JMP': 0b1011,  # change counter to given address
        'NOP': 0b0111,  # no operation
        'JO':  0b0100,  # change counter to address if overflow flag active
        'JNO': 0b1100,  # change counter to address if overflow flag inactive
        # 'JZ':  0b0110,  # change counter to address if zero flag active
    }
    if mnemonic == 'DATA':
        if data != data & 0xFF:
            print(f"Warning: input data is not one byte {mnemonic, data}", file=sys.stderr)
        return data & 0xFF
    else:
        if data != data & 0x0F:
            print(f"Warning: input data is not one nib {mnemonic, data}", file=sys.stderr)
        return (mn_table[mnemonic] << 4) + (data & 0x0F)


def spike_data_from_instructions(
    instructions: list[str | tuple[str, int]],
    sep: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    spikes_array = np.zeros((len(instructions), 14))
    times = np.arange(1, len(instructions) + 1) * sep

    addresses = np.arange(len(instructions)).reshape(-1, 1)
    bin_addresses = (((2**np.arange(4)) & addresses) != 0).astype(int)

    data = np.array([
        content(*inst) if isinstance(inst, tuple) else content(inst)
        for inst in instructions])
    bin_data = (((2**np.arange(8)) & data.reshape(-1, 1)) != 0).astype(int)

    spikes_array[:, 2:6] = bin_addresses[:, -1::-1]
    spikes_array[:, 6:] = bin_data[:, -1::-1]

    return spikes_array.astype(int), times


# Testing everything
if True and __name__ == '__main__':
    # test with:
    # > src/doryta --spike-driven \
    # > --load-model=../data/models/snn-circuits/snn-models/computer_8bit_v1.doryta.bin \
    # > --load-spikes=../data/models/snn-circuits/spikes/computer_8bit_v1-hello-world.bin \
    # > --probe-firing{,-output-only} --output-dir=testing-8-bit/computer_8bit_v1 \
    # > --save-state --end=20
    # check output with:
    # > python tools/sncircuits/process_output.py \
    # > --path build/testing-8-bit/computer_8bit_v1 --complement

    ht = 1/256
    computer_8bit = base.computer_8bit(ht)

    save(computer_8bit.circuit.remove_unneded_neurons(),
         dump_folder / 'snn-models' / 'computer_8bit_v1.doryta.bin',
         heartbeat=ht, verbose=True)

    programs: dict[str, list[str | tuple[str, int]]] = {
        'add-to-42': [
            ('LDA', 0x8),            # 0x0
            ('ADD', 0x9),            # 0x1
            ('STA', 0x7),            # 0x2
            ('LDA', 0xA),            # 0x3
            ('ADD', 0x7),            # 0x4
            ('OUT'),                 # 0x5
            ('HLT'),                 # 0x6
            ('DATA', 0b00100010),    # 0x7
            ('DATA', 0b01000001),    # 0x8
            ('DATA', 0b10101000),    # 0x9
            ('DATA', 0b01000001),    # 0xA
        ],
        'hello-world': [  # will print "Hi world!" followed by many extraneous bytes
            ('LDA', 0x3),          # 0x0
            ('ADD', 0x6),          # 0x1
            ('STA', 0x3),          # 0x2
            ('LDA', 0x6),          # 0x3
            ('OUT'),               # 0x4
            ('JMP', 0x0),          # 0x5
            ('DATA', 0x1),         # 0x6
            ('DATA', ord('H')),    # 0x7
            ('DATA', ord('i')),    # 0x8
            ('DATA', ord(' ')),    # 0x9
            ('DATA', ord('W')),    # 0xA
            ('DATA', ord('o')),    # 0xB
            ('DATA', ord('r')),    # 0xC
            ('DATA', ord('l')),    # 0xD
            ('DATA', ord('d')),    # 0xE
            ('DATA', ord('!')),    # 0xF
        ],
    }

    for name, program in programs.items():
        # creating spikes
        spikes_array, times = spike_data_from_instructions(program, ht * 6)
        # print(spikes_array)
        # print(times)

        spikes_array[16:] = 0
        spikes_array[16:, 5] = 1

        spikes = {
            # bus to ram
            14: times[:16] + 2 * ht,
            # bus to counter
            # 15: np.array([]),
            15: times[16:] + 2 * ht,
            # bus to cpu
            16: np.array([]),
            # start
            # 17: np.array([10]),
            17: times[-1:] + 10 * ht
        }

        save_spikes_for_doryta(dump_folder / 'spikes' / f'computer_8bit_v1-{name}',
                               img=spikes_array[:, -1::-1], times=times,
                               individual_spikes=spikes)
