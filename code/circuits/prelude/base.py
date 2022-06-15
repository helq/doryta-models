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
        snc.neuron("a",                         synapses={"memory": {}, "q": {}})
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
        snc.output("Latch_0.out_0")
        snc.output("Latch_1.out_0")
        snc.output("Latch_2.out_0")
        snc.output("Latch_3.out_0")
        snc.output("Latch_4.out_0")
        snc.output("Latch_5.out_0")
        snc.output("Latch_6.out_0")
        snc.output("Latch_7.out_0")

        snc.input("activate", inputs=["Latch_0.activate", "Latch_1.activate", "Latch_2.activate",
                                      "Latch_3.activate", "Latch_4.activate", "Latch_5.activate",
                                      "Latch_6.activate", "Latch_7.activate"])
        snc.input("set_0", inputs=["Latch_0.set"])
        snc.input("set_1", inputs=["Latch_1.set"])
        snc.input("set_2", inputs=["Latch_2.set"])
        snc.input("set_3", inputs=["Latch_3.set"])
        snc.input("set_4", inputs=["Latch_4.set"])
        snc.input("set_5", inputs=["Latch_5.set"])
        snc.input("set_6", inputs=["Latch_6.set"])
        snc.input("set_7", inputs=["Latch_7.set"])
        snc.input("reset", inputs=["Latch_0.reset", "Latch_1.reset", "Latch_2.reset",
                                   "Latch_3.reset", "Latch_4.reset", "Latch_5.reset",
                                   "Latch_6.reset", "Latch_7.reset"])

        snc.include("Latch_0", asr_latch(heartbeat))
        snc.include("Latch_1", asr_latch(heartbeat))
        snc.include("Latch_2", asr_latch(heartbeat))
        snc.include("Latch_3", asr_latch(heartbeat))
        snc.include("Latch_4", asr_latch(heartbeat))
        snc.include("Latch_5", asr_latch(heartbeat))
        snc.include("Latch_6", asr_latch(heartbeat))
        snc.include("Latch_7", asr_latch(heartbeat))
    return snc.circuit


def byte_latch_v1a(heartbeat: float) -> SNCircuit:
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
        for i in range(8):
            snc.input(f"set_{i}", inputs=[f"Latch_{i}.set"])
        snc.input("reset", inputs=[f"Latch_{i}.reset" for i in range(8)])

        for i in range(8):
            snc.include(f"Latch_{i}", asr_latch(heartbeat))
    return snc.circuit


if __name__ == '__main__':
    asr_latch_test = asr_latch(1/256)
    byte_latch_test = byte_latch_v1a(1/256)
