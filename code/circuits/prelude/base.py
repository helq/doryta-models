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


def byte_latch(heartbeat: float) -> SNCircuit:
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
        for i in range(8):
            snc.output(f"Latch_{i}.out_0")

        snc.input("read", inputs=[f"Latch_{i}.activate" for i in range(8)])
        snc.input("reset", inputs=[f"Latch_{i}.reset" for i in range(8)])
        for i in range(8):
            snc.input(f"set_{i}", inputs=[f"Latch_{i}.set"])

        for i in range(8):
            snc.include(f"Latch_{i}", asr_latch(heartbeat))
    return snc.circuit


def two_bytes_RAM(heartbeat: float) -> SNCircuit:
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


if __name__ == '__main__':
    asr_latch_test = asr_latch(1/256)
    byte_latch_test = byte_latch(1/256)
    two_bytes_RAM_test = two_bytes_RAM(1/256)
    RAM_16_bytes_test = RAM(1/256, 4)
