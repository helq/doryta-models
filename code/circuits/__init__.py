from .sncircuit import Neuron, SynapParams, SNCircuit, LIF
from .load import from_json_path, from_json_str, from_json_obj

__all__ = ['from_json_path', 'from_json_str', 'from_json_obj', 'Neuron', 'SNCircuit',
           'LIF']


if __name__ == '__main__':
    ex_circuit = from_json_path('snn-circuits/json/clock-smallest.json', {'heartbeat': 1/256})
    ex_circuit2 = from_json_path('snn-circuits/json/clock.json', {'heartbeat': 1/256})

    assert ex_circuit == ex_circuit2, \
        "Both files implement the same network, thus they should be the same"

    stop_circuit = from_json_path('gol/json/gol-nonnegative-v3.json', {'heartbeat': 1/256})
    cycle3_circuit = from_json_path('snn-circuits/json/turnable-cycle-3.json', {'heartbeat': 1/256})

    cycle_from_stop_circuit = stop_circuit \
        .connect_with_itself([(0, 0)], remove_outputs=False) \
        .insert_input(pos=0, params={'e': SynapParams(weight=1.0, delay=1)})

    assert cycle_from_stop_circuit.same_as(cycle3_circuit)

    ex_circuit = from_json_path('snn-circuits/json/clock-smallest.json', {'heartbeat': 1/256})
