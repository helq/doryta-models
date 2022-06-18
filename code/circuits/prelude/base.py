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

        snc.input("activate", inputs=[f"Latch_{i}.activate" for i in range(8)])
        snc.input("reset", inputs=[f"Latch_{i}.reset" for i in range(8)])
        for i in range(8):
            snc.input(f"set_{i}", inputs=[f"Latch_{i}.set"])

        for i in range(8):
            snc.include(f"Latch_{i}", asr_latch(heartbeat))
    return snc.circuit


if __name__ == '__main__':
    asr_latch_test = asr_latch(1/256)
    byte_latch_test = byte_latch(1/256)
