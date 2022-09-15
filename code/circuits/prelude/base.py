from __future__ import annotations

from math import log
from typing import Literal

from ..sncircuit import SNCreate, LIF
from ..snvisualize import SNCreateVisual
from ..visualize.base import CircuitDisplay


def asr_latch(heartbeat: float, noQ: bool = False) -> SNCreate:
    inf = 2e20
    with SNCreate(
        neuron_type=LIF,
        neuron_params={
            "resistance": 1.0,
            "capacitance": heartbeat,
            "threshold": 0.8
        },
        synapse_params={
            "weight": 0.5,
            "delay": 1
        }
    ) as snc:
        snc.output("q")
        snc.input("activate", synapses={"a": {"weight": 1.0}, "memory": {}})
        snc.input("set",      synapses={"memory": {}})
        snc.input("reset",    synapses={"memory": {"weight": 1.0}})
        snc.neuron("a", synapses={"memory", "q"})
        snc.neuron("memory", params={"resistance": inf}, synapses={"q"})
        snc.neuron("q")

        if noQ:
            snc.output("no-q")
            snc.neuron("no-q")
            snc.connection("a", "no-q", synapse_params={"delay": 2})
            snc.connection("memory", "no-q", synapse_params={})

    return snc


def asr_latch_visual(snc: SNCreate) -> SNCreateVisual:
    visual = SNCreateVisual(snc)

    visual.def_size(4, 3)
    visual.def_node_pos('a', (1.5, .8))
    visual.def_node_pos('memory', (1.5, 2.2))
    visual.def_node_pos('q', (3, .8))
    # visual.def_path('activate', 'a', [(.5, .5), (.5, .8)])
    # visual.def_path('memory', 'q', [(3, 2.2)])
    # visual.def_path('q', 'out_0', [(3.6, .8), (3.6, .5)])
    return visual


def multi_latch(heartbeat: float, n_bits: int = 8) -> SNCreate:
    assert n_bits > 1
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        asr_latch_val = asr_latch(heartbeat)
        for i in range(n_bits):
            snc.output(f"Latch_{i}.out_0")

        snc.input("read", inputs=[f"Latch_{i}.activate" for i in range(n_bits)])
        snc.input("reset", inputs=[f"Latch_{i}.reset" for i in range(n_bits)])
        for i in range(n_bits):
            snc.input(f"set_{i}", inputs=[f"Latch_{i}.set"])

        for i in range(n_bits):
            snc.include(f"Latch_{i}", asr_latch_val.circuit)

    return snc


def multi_latch_visual(snc: SNCreate, depth: int = 1) -> SNCreateVisual:
    n_bits = len(snc._include_circuit_names)
    if depth == 1:
        asr_visual = asr_latch_visual(asr_latch(1/2))
    else:
        assert depth == 0
        asr_visual = SNCreateVisual(asr_latch(1/2))

    visual = SNCreateVisual(snc)

    for i in range(n_bits):
        visual.include_visual(f'Latch_{i}', asr_visual.generate())

    inc_width = asr_visual.generate().size.x + 1
    width = 0.6 + inc_width * n_bits
    height = 3.9 + .6 * n_bits
    visual.def_size(width, height)
    activate_height = 1.3 + .3*n_bits

    for i in range(n_bits):
        visual.def_include_pos(f'Latch_{i}', (1 + inc_width*i, .8 + .3*n_bits))
        visual.def_path('read', f'Latch_{i}.activate',
                        [(0.7 + inc_width*i, 0.2),
                         (0.7 + inc_width*i, activate_height)])
        visual.def_path(f'set_{i}', f'Latch_{i}.set',
                        [(0.3 + inc_width*i, 0.8 + .3*i),
                         (0.3 + inc_width*i, activate_height + 1)])
        visual.def_path('reset', f'Latch_{i}.reset',
                        [(0.5 + inc_width*i, 0.5),
                         (0.5 + inc_width*i, activate_height + 2)])
        visual.def_path(f'Latch_{i}.out_0', f'out_{i}',
                        [(0.2 + inc_width*(i+1), activate_height),
                         (0.2 + inc_width*(i+1), activate_height + 2.7 + .3*i)])

    visual.def_inputs([(0.0, 0.2), (0, 0.5)] + [(0, 0.8 + 0.3*i) for i in range(n_bits)])
    visual.def_outputs([(width, 4 + 0.3 * n_bits + 0.3*i) for i in range(n_bits)])

    return visual


byte_latch = multi_latch


def two_bytes_RAM(heartbeat: float) -> SNCreate:
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        for i in range(8):
            snc.output({f"Byte_0.out_{i}", f"Byte_1.out_{i}"})

        snc.input("read", synapses={"and_addr_read", "and_naddr_read"})
        snc.input("reset", synapses={"and_addr_reset", "and_naddr_reset"})
        for i in range(8):
            snc.input(f"set_{i}", synapses={f"and_addr_set_{i}", f"and_naddr_set_{i}"})
        new_neuron_pos = ['read', 'reset'] + [f"set_{i}" for i in range(8)]
        snc.input("addr", synapses={f"and_addr_{s}" for s in new_neuron_pos})
        snc.input("naddr", synapses={f"and_naddr_{s}" for s in new_neuron_pos})

        byte_latch_snc = byte_latch(heartbeat)
        snc.include("Byte_0", byte_latch_snc.circuit)
        snc.include("Byte_1", byte_latch_snc.circuit)

        snc.neuron("and_addr_read", to_inputs={"Byte_0.read"})
        snc.neuron("and_addr_reset", to_inputs={"Byte_0.reset"})
        snc.neuron("and_naddr_read", to_inputs={"Byte_1.read"})
        snc.neuron("and_naddr_reset", to_inputs={"Byte_1.reset"})
        for i in range(8):
            snc.neuron(f"and_addr_set_{i}", to_inputs={f"Byte_0.set_{i}"})
            snc.neuron(f"and_naddr_set_{i}", to_inputs={f"Byte_1.set_{i}"})

    return snc


def RAM(heartbeat: float, depth: int = 1) -> SNCreate:  # noqa: C901
    assert depth > 0

    def _finding_address_helper(current_bit: int, suffix: str) -> None:
        assert 0 < current_bit <= depth

        if current_bit == depth:  # base case
            for data in data_names:
                snc.neuron(f"{data}-{suffix}", to_inputs={f"Byte_{suffix}.{data}"})

        else:  # recursive case
            # Defining neurons and their connections for data at this level
            for data in data_names:
                snc.neuron(f"{data}-{suffix}", synapses={f"{data}-{suffix}0", f"{data}-{suffix}1"})

            # These neurons decide where to route the data for the next layer
            data_names_curr = data_names + [f'addr_{i}' for i in range(current_bit + 1, depth)] \
                + [f'naddr_{i}' for i in range(current_bit + 1, depth)]
            snc.neuron(f"addr_{current_bit}-{suffix}",
                       synapses={f"{s}-{suffix}1" for s in data_names_curr})
            snc.neuron(f"naddr_{current_bit}-{suffix}",
                       synapses={f"{s}-{suffix}0" for s in data_names_curr})

            # These neurons are treated as data neurons at this level
            for i in range(current_bit + 1, depth):
                snc.neuron(f"addr_{i}-{suffix}",
                           synapses={f"addr_{i}-{suffix}0", f"addr_{i}-{suffix}1"})
                snc.neuron(f"naddr_{i}-{suffix}",
                           synapses={f"naddr_{i}-{suffix}0", f"naddr_{i}-{suffix}1"})

            _finding_address_helper(current_bit + 1, suffix + "0")
            _finding_address_helper(current_bit + 1, suffix + "1")

    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        # ### Defining the output. Each output bit aggregates `2**depth` bits output
        #     (connections)
        binary_repr = f'{{0:0{depth}b}}'  # for depth=4 this is '{0:04b}'
        for i in range(8):
            snc.output({f"Byte_{binary_repr.format(neu)}.out_{i}" for neu in range(2**depth)})

        data_names = ['read', 'reset'] + [f"set_{i}" for i in range(8)]

        # ### Defining data input to ram (read, reset and set_X/write inputs)
        for data in data_names:
            snc.input(data, synapses={f"{data}-0", f"{data}-1"})

        # ### Address inputs
        data_names_curr = data_names + [f'addr_{i}' for i in range(1, depth)] \
            + [f'naddr_{i}' for i in range(1, depth)]
        # - address
        snc.input("addr_0", synapses={f"{s}-1" for s in data_names_curr})
        for i in range(1, depth):
            snc.input(f"addr_{i}", synapses={f"addr_{i}-0", f"addr_{i}-1"})
        # - neq address
        snc.input("naddr_0", synapses={f"{s}-0" for s in data_names_curr})
        for i in range(1, depth):
            snc.input(f"naddr_{i}", synapses={f"naddr_{i}-0", f"naddr_{i}-1"})

        # ### Defining connections
        byte_latch_snc = byte_latch(heartbeat)
        _finding_address_helper(current_bit=1, suffix="0")
        _finding_address_helper(current_bit=1, suffix="1")

        # ### Including all the necessary bytes for `depth` levels
        for neu in range(2**depth):
            suffix = binary_repr.format(neu)
            snc.include(f"Byte_{suffix}", byte_latch_snc.circuit)

    return snc


def RAM_visual(
    snc: SNCreate,
    byte_visual: None | CircuitDisplay | dict[str, CircuitDisplay]
) -> SNCreateVisual:
    ram_depth = int(log(len(snc._include_circuit_names), 2))
    if ram_depth != 1:
        raise NotImplementedError("Sorry, RAM visual only implemented for depth of 1")

    if byte_visual is None:
        byte_visual = SNCreateVisual(byte_latch(1/2)).generate()

    visual = SNCreateVisual(snc)

    bin_repr = f'{{0:0{ram_depth}b}}'
    if isinstance(byte_visual, CircuitDisplay):
        for i in range(2 ** ram_depth):
            visual.include_visual(f'Byte_{bin_repr.format(i)}', byte_visual)
        inc_width = byte_visual.size.x + 1
        inc_heights = [byte_visual.size.y + 1 for i in range(2 ** ram_depth)]
    else:
        assert isinstance(byte_visual, dict)
        inc_width = 0
        inc_heights = []
        for inc_name, inc_visual in byte_visual.items():
            visual.include_visual(inc_name, inc_visual)
            inc_width = max(inc_width, inc_visual.size.x + 1)
            inc_heights.append(inc_visual.size.y)

    width = 3 + inc_width
    height = sum(ih + 1 for ih in inc_heights) + 1
    visual.def_size(width, height)

    acc_height = 1.0
    for i in range(2 ** ram_depth):
        visual.def_include_pos(f'Byte_{bin_repr.format(i)}', (3, acc_height))
        acc_height += 1 + inc_heights[i]

    visual.def_node_pos('read-0', (2, 1))
    visual.def_node_pos('reset-0', (2, 2))
    visual.def_node_pos('read-1', (2, 12))
    visual.def_node_pos('reset-1', (2, 13))
    for i in range(8):
        visual.def_node_pos(f'set_{i}-0', (2, 3 + i))
        visual.def_node_pos(f'set_{i}-1', (2, 14 + i))

    # visual.def_inputs([(0.0, 0.2), (0, 0.5)] + [(0, 0.8 + 0.3*i) for i in range(n_bits)])
    # visual.def_outputs([(width, 4 + 0.3 * n_bits + 0.3*i) for i in range(n_bits)])

    return visual


def half_adder(heartbeat: float) -> SNCreate:
    """
    SNCircuit with two inputs (two bits), and two outputs (addition and carry bits).
    """
    inf = 2e20
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        snc.output("xor")
        snc.output("and")
        snc.input("bit-0", synapses={"or": {"weight": 1.0}, "mem": {}})
        snc.input("bit-1", synapses={"or": {"weight": 1.0}, "mem": {}})
        snc.neuron("or", synapses={"mem": {}, "and": {}, "xor": {"delay": 2}})
        snc.neuron("mem", params={"resistance": inf}, synapses={"and", "xor"})
        snc.neuron("and", synapses={"mem"})
        snc.neuron("xor")
    return snc


def full_adder(heartbeat: float) -> SNCreate:
    """
    SNCircuit with three inputs (three bits), and two outputs (addition and carry bits).
    """
    inf = 2e20
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        snc.output({"xor-1", "xor-2"})
        snc.output("and")
        input_bit_synapses: dict[str, dict[str, int | float]] \
            = {"or": {"weight": 1.0}, "mem": {}, "xor-2": {"weight": 0.3, "delay": 3}}
        snc.input("bit-0", synapses=input_bit_synapses)
        snc.input("bit-1", synapses=input_bit_synapses)
        snc.input("bit-2", synapses=input_bit_synapses)
        snc.neuron("or", synapses={"mem": {}, "and": {}, "xor-1": {"delay": 2}})
        snc.neuron("mem", params={"resistance": inf}, synapses={"and", "xor-1"})
        snc.neuron("and", synapses={"mem"})
        snc.neuron("xor-1")
        snc.neuron("xor-2")
    return snc


def two_bit_adder(heartbeat: float) -> SNCreate:
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 1.0, "delay": 1}
    ) as snc:
        half = half_adder(heartbeat)
        full = full_adder(heartbeat)
        snc.output("Half.out_0")
        snc.output("Full_1.out_0")
        snc.output("Full_1.out_1")

        snc.input("in0-0", inputs=['Half.bit-0'])
        snc.input("in0-1", synapses={'to-full_1-1': {'delay': 2}})
        snc.input("in1-0", inputs=['Half.bit-1'])
        snc.input("in1-1", synapses={'to-full_1-2': {'delay': 2}})
        snc.connection('Half.out_1', 'Full_1.bit-0')
        # TODO: get rid of these extra neurons, it should be possible to modify the inputs
        snc.neuron('to-full_1-1', to_inputs={'Full_1.bit-1'})
        snc.neuron('to-full_1-2', to_inputs={'Full_1.bit-2'})

        snc.include("Half", half.circuit)
        snc.include("Full_1", full.circuit)
    return snc


def multi_bit_adder(heartbeat: float, n_bits: int) -> SNCreate:
    assert n_bits > 0
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 1.0, "delay": 1}
    ) as snc:
        half = half_adder(heartbeat)
        full = full_adder(heartbeat)
        snc.output("Half.out_0")
        for b in range(1, n_bits):
            snc.output(f"Full_{b}.out_0")
        snc.output(f"Full_{n_bits-1}.out_1")

        snc.input("in0-0", inputs=['Half.bit-0'])
        for b in range(1, n_bits):
            snc.input(f"in0-{b}", synapses={f'to-full_{b}-1': {'delay': b * 2}})
        snc.input("in1-0", inputs=['Half.bit-1'])
        for b in range(1, n_bits):
            snc.input(f"in1-{b}", synapses={f'to-full_{b}-2': {'delay': b * 2}})

        if n_bits > 1:
            snc.connection('Half.out_1', 'Full_1.bit-0')
        for b in range(2, n_bits):
            snc.connection(f'Full_{b-1}.out_1', f'Full_{b}.bit-0')

        # TODO: get rid of these extra neurons, it should be possible to modify the inputs
        for b in range(1, n_bits):
            snc.neuron(f'to-full_{b}-1', to_inputs={f'Full_{b}.bit-1'})
            snc.neuron(f'to-full_{b}-2', to_inputs={f'Full_{b}.bit-2'})

        snc.include("Half", half.circuit)
        for b in range(1, n_bits):
            snc.include(f"Full_{b}", full.circuit)
    return snc


def counter_register(heartbeat: float, n_bits: int = 8) -> SNCreate:
    assert n_bits > 1
    """
    This is a byte latch with extra functionality, triggering `counter-up` will make the
    internal value of the register go up one bit.
    """
    asr_latch_val = asr_latch(heartbeat)
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        for i in range(n_bits):
            snc.output(f"Latch_{i}.out_0")

        snc.input("read", inputs=[f"Latch_{i}.activate" for i in range(n_bits)])
        snc.input("reset", inputs=[f"Latch_{i}.reset" for i in range(n_bits)])
        for i in range(n_bits):
            snc.input(f"set_{i}", inputs=[f"Latch_{i}.set"])
        snc.input("count-up", synapses={'Latch_0.memory': {},
                                        'count-up-1': {'delay': 2}})

        for i in range(1, n_bits-1):
            snc.neuron(f'count-up-{i}', synapses={
                f'Latch_{i}.memory': {}, f'count-up-{i+1}': {'delay': 2}})
            snc.connection(f'Latch_{i}.memory', f'count-up-{i+1}', synapse_params={})
        if n_bits > 1:
            snc.connection('Latch_0.memory', 'count-up-1', synapse_params={})
            snc.neuron(f'count-up-{n_bits-1}', synapses={f'Latch_{n_bits-1}.memory'})

        for i in range(n_bits):
            snc.include(f"Latch_{i}", asr_latch_val.circuit)
    return snc


def counter_register_visual(snc: SNCreate, depth: int = 0) -> SNCreateVisual:
    n_bits = len(snc._include_circuit_names)
    if depth == 1:
        asr_visual = asr_latch_visual(asr_latch(1/2)).generate()
    else:
        assert depth == 0
        asr_visual = SNCreateVisual(asr_latch(1/2)).generate()

    visual = SNCreateVisual(snc)

    for i in range(n_bits):
        visual.include_visual(f'Latch_{i}', asr_visual)

    inc_width = asr_visual.size.x + 2.5
    width = 0.6 + inc_width * n_bits
    height = 3.9 + .6 * n_bits
    visual.def_size(width, height)
    activate_height = 1.3 + .3*n_bits

    for i in range(1, n_bits):
        visual.def_node_pos(f'count-up-{i}', (-0.65 + inc_width*i, 3.0 + .3*n_bits))

    for i in range(n_bits):
        visual.def_include_pos(f'Latch_{i}', (1 + inc_width*i, .8 + .3*n_bits))
        visual.def_path('read', f'Latch_{i}.activate',
                        [(0.7 + inc_width*i, 0.2),
                         (0.7 + inc_width*i, activate_height)])
        visual.def_path(f'set_{i}', f'Latch_{i}.set',
                        [(0.3 + inc_width*i, 0.8 + .3*i),
                         (0.3 + inc_width*i, activate_height + 1)])
        visual.def_path('reset', f'Latch_{i}.reset',
                        [(0.5 + inc_width*i, 0.5),
                         (0.5 + inc_width*i, activate_height + 2)])
        visual.def_path(f'Latch_{i}.out_0', f'out_{i}',
                        [(0.2 + inc_width*(i+1), activate_height),
                         (0.2 + inc_width*(i+1), activate_height + 2.7 + .3*i)])

    for i in range(1, n_bits-1):
        visual.def_path(f'count-up-{i}', f'count-up-{i+1}',
                        [(-0.75 + inc_width*(i+.75), activate_height + 3.4)])

    visual.def_inputs(
        [(0.0, 0.2), (0, 0.5)] +
        [(0, 0.8 + 0.3*i) for i in range(n_bits)] +
        [(0, 0.8 + 0.3*n_bits)]
    )
    visual.def_outputs([(width, 4 + 0.3 * n_bits + 0.3*i) for i in range(n_bits)])

    return visual


ASRType = Literal['q', 'no-q', 'both']


# This function uses way too many loops, I know, everything could be done in a single loop
# (or two), but order does matter! unless you don't care much about it
def bus(  # noqa: C901
    heartbeat: float,
    n_bits: int = 8,
    output_pieces: dict[str, ASRType] | None = None,
    clear_after_sending: bool = True
) -> tuple[SNCreate, dict[str, list[int]]]:
    """
    This function extends a multi latch with the ability to send its content to one of
    multiple destination instead of just reading from it (it has multiple outputs).

    Params:
    - `output_pieces`: indicates the number of outputs and their types (q: contents of the
      register, no-q: the inverse of the contents of the multilatch, or, both)
    - `clear_after_sending`: if true, it will clear the value contained in the multilatch
      after it is accessed
    """
    assert n_bits > 1
    if output_pieces is None:
        output_pieces = {}

    is_noQ_present = any(t != 'q' for t in output_pieces.values())
    asr_latch_val = asr_latch(heartbeat, noQ=is_noQ_present)

    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 0.5, "delay": 1}
    ) as snc:
        # Defining output of circuit
        for j, (name, type) in enumerate(output_pieces.items()):
            if type in {'q', 'both'}:
                for i in range(n_bits):
                    snc.output(f"{name}-{i}-q")
            if type in {'no-q', 'both'}:
                for i in range(n_bits):
                    snc.output(f"{name}-{i}-no-q")

        # Defining input of circuit
        activate_inputs = [f"Latch_{i}.activate" for i in range(n_bits)]
        reset_inputs = [f"Latch_{i}.reset" for i in range(n_bits)]
        synapses_base = {} if not clear_after_sending else \
            {"reset-neuron": {"weight": 1.0, "delay": 2}}
        delay_q = 4 if is_noQ_present else 3
        for name, type in output_pieces.items():
            synapses = synapses_base.copy()
            if type in {'q', 'both'}:
                synapses |= {f'{name}-{i}-q': {'delay': delay_q} for i in range(n_bits)}
            if type in {'no-q', 'both'}:
                synapses |= {f'{name}-{i}-no-q': {'delay': 4} for i in range(n_bits)}
            snc.input(name, synapses=synapses, inputs=activate_inputs)
        if not clear_after_sending:
            snc.input("reset", inputs=reset_inputs)

        for i in range(n_bits):
            snc.input(f"set_{i}", inputs=[f"Latch_{i}.set"])

        # Defining new neurons and connections
        if clear_after_sending:
            snc.neuron("reset-neuron", to_inputs=reset_inputs)
        delay_q = 2 if is_noQ_present else 1
        for name, type in output_pieces.items():
            for i in range(n_bits):
                if type in {'both', 'q'}:
                    snc.neuron(f"{name}-{i}-q")
                    snc.connection(f"Latch_{i}.q", f"{name}-{i}-q",
                                   synapse_params={'delay': delay_q})
                if type in {'both', 'no-q'}:
                    snc.neuron(f"{name}-{i}-no-q")
                    snc.connection(f"Latch_{i}.no-q", f"{name}-{i}-no-q", synapse_params={})

        # including ASR latches
        for i in range(n_bits):
            snc.include(f"Latch_{i}", asr_latch_val.circuit)

    outputs = {}
    i = 0
    for name, type in output_pieces.items():
        outputs[name] = list(range(i, i + n_bits))
        i += n_bits
        if type == 'both':
            outputs[f'{name}-no-q'] = list(range(i, i + n_bits))
            i += n_bits

    return snc, outputs


def glued_ALU(heartbeat: float, n_bits: int = 8) -> SNCreate:
    """
    Glued ALU consists of: two registers (A and B; each with own set and reset inputs), an
    ALU activation input, and one output (the value contained in A).
    """
    # Define register A (a variation of register with additional connections, similar to BUS)
    regA, regA_outs = bus(heartbeat, n_bits=n_bits,
                          output_pieces={'alu': 'q', 'bus': 'q'},
                          clear_after_sending=False)
    # Define register B, a simple byte of memory that feeds to ALU
    regB = byte_latch(heartbeat, n_bits)
    # Load ALU
    alu = multi_bit_adder(heartbeat, n_bits)

    # connecting all pieces
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 1.0, "delay": 1}
    ) as snc:
        # Define inputs
        snc.input("alu", inputs=["Register-A.alu"], synapses={'delay-ALU-conn'})
        snc.input("regA-bus", inputs=["Register-A.bus"])
        snc.input("regA-reset", inputs=["Register-A.reset"])
        for i in range(n_bits):
            snc.input(f"regA-set_{i}", inputs=[f"Register-A.set_{i}"])
        # snc.input("regB-alu", inputs=["Register-B.read"])
        snc.input("regB-reset", inputs=["Register-B.reset"])
        for i in range(n_bits):
            snc.input(f"regB-set_{i}", inputs=[f"Register-B.set_{i}"])

        # Define outputs (only register A)
        for i, out_i in enumerate(regA_outs['bus']):
            snc.output(f"Register-A.out_{out_i}")
            # snc.output({f"Register-A.out_{out_i}", f"ALU.out_{i}"})

        # Define connections between circuits
        for i, out_i in enumerate(regA_outs['alu']):
            # - Register A to ALU
            snc.connection(f'Register-A.out_{out_i}', f'ALU.in0-{i}')
        for i in range(n_bits):
            # - Register B to ALU
            snc.connection(f'Register-B.out_{i}', f'ALU.in1-{i}')
            # - ALU to Register A
            snc.connection(f'ALU.out_{i}', f'Register-A.set_{i}')

        # Additional neurons to coordinate components interaction
        snc.neuron('delay-ALU-conn', to_inputs=['Register-B.read'],
                   synapses={'delay2-ALU-conn'})
        snc.neuron('delay2-ALU-conn', to_inputs=['Register-A.reset'])

        # Including circuits
        snc.include("Register-A", regA.circuit)
        snc.include("Register-B", regB.circuit)
        snc.include("ALU", alu.circuit)

    # regB_visual = SNCreateVisual(regB).generate()
    # regA_visual = SNCreateVisual(regA).generate()
    # alu_visual = SNCreateVisual(alu).generate()
    # gluing_visual = SNCreateVisual(snc)
    # gluing_visual.include_visual('Register-A', regA_visual)
    # gluing_visual.include_visual('Register-B', regB_visual)
    # gluing_visual.include_visual('ALU', alu_visual)
    # gluing_visual.def_include_pos('Register-A', (4, 1))
    # gluing_visual.def_include_pos('Register-B', (4, 18))
    # gluing_visual.def_include_pos('ALU', (20, 1))
    # gluing_visual.def_size(30, 30)

    return snc


def all_glued_but_CPU(  # noqa: C901
    heartbeat: float, n_bits: int = 8, ram_addr_bits: int = 4
) -> SNCreate:
    # Load glued-ALU
    glued_ALU_ = glued_ALU(heartbeat, n_bits)
    # Load BUS (five output connections, to: RAM, register A, register B, counter up, and output)
    #   The BUS is not of size `n_bits` because it also contains the address (when asking
    #   for an address to RAM) and extra bits to be used for each unit as they please.
    #   For the RAM, the first extra bit is for read and the second is for reset
    bus_, bus_outputs = bus(
        heartbeat,
        n_bits + ram_addr_bits + 2,
        output_pieces={
            'ram': 'both',
            'reg-a': 'q',
            'reg-b': 'q',
            'reg-counter': 'q',
            'cpu': 'q',
            'output-circuit': 'q',
        })
    # Load RAM and counter up register
    ram = RAM(heartbeat, depth=ram_addr_bits)
    # Load counter register
    counter = counter_register(heartbeat, ram_addr_bits)

    # connecting all pieces
    with SNCreate(
        neuron_type=LIF,
        neuron_params={"resistance": 1.0, "capacitance": heartbeat, "threshold": 0.8},
        synapse_params={"weight": 1.5, "delay": 1}
    ) as snc:
        # glue together RAM, BUS, ALU, Register A and B, and Counter Register
        # Inputs:
        #   - BUS set, activation and redirection bits
        #   - counter count-up, read and reset
        #   - read, reset and activate (alu) for glued-ALU
        for circuit_name, input_list in [
                ('BUS', bus_.circuit.inputs_id_list),
                ('Counter', ['read', 'reset', 'count-up']),
                ('Glued_ALU', ['alu', 'regA-bus', 'regA-reset', 'regB-reset']),
        ]:
            for input_name in input_list:
                snc.input(f'{circuit_name.lower()}-{input_name}',
                          inputs=[f'{circuit_name}.{input_name}'])
        # Outputs:
        #   - BUS to output connection
        for out_i in bus_outputs['output-circuit'][:n_bits]:
            snc.output(f'BUS.out_{out_i}')

        # Connect BUS to all four of its components and assign output as output of the whole circuit
        #   - Connecting BUS to Counter and back
        for i, out_i in enumerate(bus_outputs['reg-counter'][:ram_addr_bits]):
            snc.connection(f'BUS.out_{out_i}', f'Counter.set_{i}')
            snc.connection(f'Counter.out_{i}', f'BUS.set_{i}')
        #   - Connecting BUS to RAM and back
        for i, out_i in enumerate(bus_outputs['ram'][:n_bits]):
            snc.connection(f'BUS.out_{out_i}', f'RAM.set_{i}')
            snc.connection(f'RAM.out_{i}', f'BUS.set_{i}')
        for i, (add_i, nadd_i) in enumerate(
            zip(bus_outputs['ram'][n_bits:n_bits+ram_addr_bits],
                bus_outputs['ram-no-q'][n_bits:])):
            snc.connection(f'BUS.out_{add_i}', f'RAM.addr_{i}')
            snc.connection(f'BUS.out_{nadd_i}', f'RAM.naddr_{i}')
        for out_i, name in zip(bus_outputs['ram'][-2:], ['read', 'reset']):
            snc.connection(f'BUS.out_{out_i}', f'RAM.{name}')
        #   - Connecting BUS to Reg-A and back
        for i, out_i in enumerate(bus_outputs['reg-a'][:n_bits]):
            snc.connection(f'BUS.out_{out_i}', f'Glued_ALU.regA-set_{i}')
            snc.connection(f'Glued_ALU.out_{i}', f'BUS.set_{i}')
        #   - Connecting BUS to Reg-B
        for i, out_i in enumerate(bus_outputs['reg-b'][:n_bits]):
            snc.connection(f'BUS.out_{out_i}', f'Glued_ALU.regB-set_{i}')

        # Including parts
        for circuit_name, circuit in [
                ('BUS', bus_.circuit),
                ('RAM', ram.circuit),
                ('Counter', counter.circuit),
                ('Glued_ALU', glued_ALU_.circuit),
        ]:
            snc.include(circuit_name, circuit)

    return snc


if __name__ == '__main__':
    asr_latch_test = asr_latch(1/256)
    byte_latch_test = byte_latch(1/256)
    two_bytes_RAM_test = two_bytes_RAM(1/256)
    RAM_16_bytes_test = RAM(1/256, 4)
    two_bit_adder_test = multi_bit_adder(1/256, 2)
