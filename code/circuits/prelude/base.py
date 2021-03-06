from __future__ import annotations

from ..sncircuit import SNCircuit, SNCreate, LIF


def asr_latch(heartbeat: float) -> SNCircuit:
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
        snc.neuron("a", synapses={"memory": {}, "q": {}})
        snc.neuron("memory", params={"resistance": inf}, synapses={"q": {}})
        snc.neuron("q")
    return snc.circuit


def multi_latch(heartbeat: float, n_bits: int = 8) -> SNCircuit:
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
            snc.include(f"Latch_{i}", asr_latch_val)
    return snc.circuit


byte_latch = multi_latch


def two_bytes_RAM(heartbeat: float) -> SNCircuit:
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
        snc.include("Byte_0", byte_latch_snc)
        snc.include("Byte_1", byte_latch_snc)

        snc.neuron("and_addr_read", to_inputs={"Byte_0.read"})
        snc.neuron("and_addr_reset", to_inputs={"Byte_0.reset"})
        snc.neuron("and_naddr_read", to_inputs={"Byte_1.read"})
        snc.neuron("and_naddr_reset", to_inputs={"Byte_1.reset"})
        for i in range(8):
            snc.neuron(f"and_addr_set_{i}", to_inputs={f"Byte_0.set_{i}"})
            snc.neuron(f"and_naddr_set_{i}", to_inputs={f"Byte_1.set_{i}"})

    return snc.circuit


def RAM(heartbeat: float, depth: int = 1) -> SNCircuit:  # noqa: C901
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
            snc.include(f"Byte_{suffix}", byte_latch_snc)

    return snc.circuit


def half_adder(heartbeat: float) -> SNCircuit:
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
    return snc.circuit


def full_adder(heartbeat: float) -> SNCircuit:
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
    return snc.circuit


def two_bit_adder(heartbeat: float) -> SNCircuit:
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

        snc.include("Half", half)
        snc.include("Full_1", full)
    return snc.circuit


def multi_bit_adder(heartbeat: float, n_bits: int) -> SNCircuit:
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

        snc.include("Half", half)
        for b in range(1, n_bits):
            snc.include(f"Full_{b}", full)
    return snc.circuit


def counter_register(heartbeat: float, n_bits: int = 8) -> SNCircuit:
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

        if n_bits > 1:
            snc.connection('Latch_0.memory', 'count-up-1', synapse_params={})
            snc.neuron(f'count-up-{n_bits-1}', synapses={f'Latch_{n_bits-1}.memory'})
        for i in range(1, n_bits-1):
            snc.neuron(f'count-up-{i}', synapses={
                f'Latch_{i}.memory': {}, f'count-up-{i}': {'delay': 2}})
            snc.connection(f'Latch_{i}.memory', f'count-up-{i+1}', synapse_params={})

        for i in range(n_bits):
            snc.include(f"Latch_{i}", asr_latch_val)
    return snc.circuit


if __name__ == '__main__':
    asr_latch_test = asr_latch(1/256)
    byte_latch_test = byte_latch(1/256)
    two_bytes_RAM_test = two_bytes_RAM(1/256)
    RAM_16_bytes_test = RAM(1/256, 4)
    two_bit_adder_test = multi_bit_adder(1/256, 2)
