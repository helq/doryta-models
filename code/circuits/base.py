from __future__ import annotations

import json

import numpy as np
from numpy.typing import NDArray

from typing import NamedTuple, List, Dict, Optional, Union, Any


class NeuronParams(NamedTuple):
    resistance: float
    capacitance: float
    threshold: float
    potential: float = 0
    current: float = 0
    resting_potential: float = 0
    reset_potential: float = 0


class SynapParams(NamedTuple):
    weight: float
    delay: float


class Neuron(NamedTuple):
    params: NeuronParams
    synapses: Dict[int, SynapParams]


class GenerateParams:
    def __init__(self, args: Dict[str, float]):
        self.args = args

    def determine_global(
        self,
        params: Dict[str, Union[float, str]],
        key: str,
        default: Optional[float] = None
    ) -> Optional[float]:
        # TODO: Actually, delay is an `int`
        # assert all(isinstance(p, (float, str)) for p in params.values())
        if key not in params:
            return default
        val = params[key]
        if isinstance(val, str):
            assert val in self.args
            return self.args[val]
        return val

    def determine_param(
        self,
        params: Dict[str, Union[float, str]],
        key: str,
        default: Optional[float] = None
    ) -> float:
        # assert all(isinstance(p, (float, str)) for p in params.values())
        if key not in params:
            if default is None:
                raise Exception(f"The parameter `{key}` has no default.")
            return default
        val = params[key]
        if isinstance(val, str):
            assert val in self.args
            return self.args[val]
        return val


class SNCircuit(NamedTuple):
    # args: List[Var]
    # neuron_params: NeuronParams
    # synap_params: SynapParams
    outputs: List[int]
    inputs: List[Dict[int, SynapParams]]
    neurons: Dict[int, Neuron]
    # inject: Something[SNCircuit]

    # Yes, littering this with `asserts` is not the proper way to construct the function.
    # Ideally, it should raise Exceptions for every single thing it finds it's weird.
    # Assertions are simpler to write and thus easier to develop code with than Exceptions
    # which require a custom message.
    # TODO: change `assert`s for raising Exceptions
    @classmethod
    def load_json(
        cls,
        s: str,
        args_: Dict[str, float]
    ) -> SNCircuit:
        args = GenerateParams(args_)
        obj = json.loads(s)
        assert isinstance(obj, dict)
        assert obj['version'] == '0.0.1'
        assert isinstance(obj['args'], list)
        assert sorted(obj['args']) == sorted(args_.keys())
        assert isinstance(obj['params'], dict)

        # neuron parameters
        g_resistance = args.determine_global(obj['params'], 'R')
        g_capacitance = args.determine_global(obj['params'], 'C')
        g_threshold = args.determine_global(obj['params'], 'threshold')
        g_potential = args.determine_global(obj['params'], 'potential', 0.0)
        g_current = args.determine_global(obj['params'], 'current', 0.0)
        g_resting_potential = args.determine_global(obj['params'], 'resting_potential', 0.0)
        g_reset_potential = args.determine_global(obj['params'], 'rest_potential', 0.0)
        # synapses parameters
        g_weight = args.determine_global(obj['params'], 'weight')
        g_delay = args.determine_global(obj['params'], 'delay')

        obj_neurons = obj['neurons']
        assert isinstance(obj_neurons, dict)
        ids_to_int = {k: i for i, k in enumerate(obj_neurons.keys())}
        neurons = {}
        for neuron_id, n_dict in obj_neurons.items():
            assert isinstance(n_dict, dict)

            n_params = n_dict["params"] if "params" in n_dict else {}
            assert isinstance(n_params, dict)
            # assert all(isinstance(p, str) and isinstance(v, (str, float))
            #            for p, v in n_params.items())

            n_synapses = n_dict["synapses"] if "synapses" in n_dict else {}
            assert isinstance(n_synapses, dict)
            # assert all(isinstance(p, str) and isinstance(v, (str, float))
            #            for p, v in n_synapses.items())

            neurons[ids_to_int[neuron_id]] = Neuron(
                params=NeuronParams(
                    resistance=args.determine_param(n_params, 'R', g_resistance),
                    capacitance=args.determine_param(n_params, 'C', g_capacitance),
                    threshold=args.determine_param(n_params, 'threshold', g_threshold),
                    potential=args.determine_param(n_params, 'potential', g_potential),
                    current=args.determine_param(n_params, 'current', g_current),
                    resting_potential=args.determine_param(
                        n_params, 'resting_potential', g_resting_potential),
                    reset_potential=args.determine_param(
                        n_params, 'reset_potential', g_reset_potential)
                ),
                synapses={
                    ids_to_int[n]: SynapParams(
                        weight=args.determine_param(s_params, 'weight', g_weight),
                        delay=args.determine_param(s_params, 'delay', g_delay)
                    )
                    for n, s_params in n_synapses.items()
                }
            )

        obj_outputs = obj['outputs']
        assert isinstance(obj_outputs, list)
        assert all(o in obj_neurons for o in obj_outputs)
        outputs = [ids_to_int[o] for o in obj_outputs]

        obj_inputs = obj['inputs']
        assert isinstance(obj_inputs, list)
        assert all(isinstance(i, dict) and all(n in obj_neurons for n in i) for i in obj_inputs)
        inputs = [{
            ids_to_int[n]: SynapParams(
                weight=args.determine_param(s_params, 'weight', g_weight),
                delay=args.determine_param(s_params, 'delay', g_delay)
            )
            for n, s_params in input.items()
        } for input in obj_inputs]

        return SNCircuit(outputs, inputs, neurons)

    def get_params_in_bulk(self) -> Dict[str, NDArray[Any]]:
        neuron_args = {  # type: ignore
            param: np.array([getattr(self.neurons[id].params, param)
                             for id in range(self.num_neurons)])
            for param in ['resistance', 'capacitance', 'threshold', 'potential', 'current',
                          'resting_potential', 'reset_potential']
        }
        return neuron_args

    @property
    def num_neurons(self) -> int:
        return len(self.neurons)


if __name__ == '__main__':
    with open('snn-circuits/json/clock-smallest.json') as f:
        s = f.read()
        ex_circuit = SNCircuit.load_json(s, {'heartbeat': 1/256})

    with open('snn-circuits/json/clock.json') as f:
        s = f.read()
        ex_circuit2 = SNCircuit.load_json(s, {'heartbeat': 1/256})

    assert ex_circuit == ex_circuit2, \
        "Both files implement the same network, thus they should be the same"
