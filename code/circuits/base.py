from __future__ import annotations

import json
import pathlib

import numpy as np
from numpy.typing import NDArray

from typing import NamedTuple, List, Dict, Optional, Union, Any, Tuple


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
    """
    Contains the description of a Spiking Neural Circuit. To be used anywhere else.

    Treat SNCircuits as immutable objects.
    Careful: all operations between `SNCircuit`s return new SNCircuit objects which are
    NOT deep copies of the originals. Any alteration to the objects might be reflected on
    their parent objects or children.
    """
    # TODO: change the type of all variables from mutable objects to immutable
    # args: List[Var]
    outputs: List[int]
    inputs: List[Dict[int, SynapParams]]
    neurons: Dict[int, Neuron]
    ids_to_int: Dict[str, int]
    # inject: Something[SNCircuit]

    def same_as(self, other: SNCircuit) -> bool:
        """
        Less strict "equal" operation. This ignores neuron ids.
        """
        return self.outputs == other.outputs \
            and self.inputs == other.inputs \
            and self.neurons == other.neurons

    @classmethod
    def load_json(
        cls,
        path: Union[str, pathlib.Path],
        args: Dict[str, float]
    ) -> SNCircuit:
        with open(path) as f:
            data = f.read()
        return SNCircuit.loads_json(data, args)

    # Yes, littering this with `asserts` is not the proper way to construct the function.
    # Ideally, it should raise Exceptions for every single thing it finds it's weird.
    # Assertions are simpler to write and thus easier to develop code with than Exceptions
    # which require a custom message.
    # TODO: change `assert`s for raising Exceptions
    @classmethod
    def loads_json(
        cls,
        data: str,
        args: Dict[str, float]
    ) -> SNCircuit:
        args_ = GenerateParams(args)

        obj = json.loads(data)
        assert isinstance(obj, dict)
        assert obj['version'] == '0.0.1'
        assert isinstance(obj['args'], list)
        assert sorted(obj['args']) == sorted(args.keys())
        assert isinstance(obj['params'], dict)

        # neuron parameters
        g_resistance = args_.determine_global(obj['params'], 'R')
        g_capacitance = args_.determine_global(obj['params'], 'C')
        g_threshold = args_.determine_global(obj['params'], 'threshold')
        g_potential = args_.determine_global(obj['params'], 'potential', 0.0)
        g_current = args_.determine_global(obj['params'], 'current', 0.0)
        g_resting_potential = args_.determine_global(obj['params'], 'resting_potential', 0.0)
        g_reset_potential = args_.determine_global(obj['params'], 'rest_potential', 0.0)
        # synapses parameters
        g_weight = args_.determine_global(obj['params'], 'weight')
        g_delay = args_.determine_global(obj['params'], 'delay')

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
                    resistance=args_.determine_param(n_params, 'R', g_resistance),
                    capacitance=args_.determine_param(n_params, 'C', g_capacitance),
                    threshold=args_.determine_param(n_params, 'threshold', g_threshold),
                    potential=args_.determine_param(n_params, 'potential', g_potential),
                    current=args_.determine_param(n_params, 'current', g_current),
                    resting_potential=args_.determine_param(
                        n_params, 'resting_potential', g_resting_potential),
                    reset_potential=args_.determine_param(
                        n_params, 'reset_potential', g_reset_potential)
                ),
                synapses={
                    ids_to_int[n]: SynapParams(
                        weight=args_.determine_param(s_params, 'weight', g_weight),
                        delay=args_.determine_param(s_params, 'delay', g_delay)
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
                weight=args_.determine_param(s_params, 'weight', g_weight),
                delay=args_.determine_param(s_params, 'delay', g_delay)
            )
            for n, s_params in input.items()
        } for input in obj_inputs]

        return SNCircuit(outputs=outputs, inputs=inputs, neurons=neurons, ids_to_int=ids_to_int)

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

    def check_correct_input_id(self, input: int) -> None:
        if not (0 <= input < len(self.inputs)):
            raise Exception(
                f"There are {len(self.inputs)} possible inputs, but index {input} "
                "was given as input")

    def check_correct_output_id(self, output: int) -> None:
        if not (0 <= output < len(self.outputs)):
            raise Exception(
                f"There are {len(self.outputs)} possible outputs, but index {output} "
                "was given as output")

    def connect_with_itself(
        self,
        out2in: List[Tuple[int, int]],
        remove_outputs: bool = True
    ) -> SNCircuit:
        """
        Creates a new circuit by connecting inputs to outputs as described by `out2in`.
        """
        neurons = self.neurons.copy()

        # this loop could be replaced with:
        # > self.__helper_wire_input_to_output(other, neurons, out2in)
        # but it has been left here as documentation for the function `connect`
        for output, input in out2in:
            self.check_correct_input_id(input)
            self.check_correct_output_id(output)

            # connecting neurons[outputs[output]] to inputs[input]
            old_neuron = neurons[self.outputs[output]]
            synapses = old_neuron.synapses.copy()
            synapses.update(self.inputs[input])
            neurons[self.outputs[output]] = Neuron(
                params=old_neuron.params,
                synapses=synapses
            )

        # removing connected outputs and inputs
        conn_outputs = set(out for out, in_ in out2in)
        conn_inputs = set(in_ for out, in_ in out2in)

        if remove_outputs:
            outputs = [out for i, out in enumerate(self.outputs) if i not in conn_outputs]
        else:
            outputs = self.outputs.copy()
        inputs = [in_ for i, in_ in enumerate(self.inputs) if i not in conn_inputs]

        return SNCircuit(neurons=neurons, outputs=outputs, inputs=inputs,
                         ids_to_int=self.ids_to_int)

    def connect(
        self,
        other: SNCircuit,
        outgoing: Optional[List[Tuple[int, int]]] = None,
        incoming: Optional[List[Tuple[int, int]]] = None,
        self_id: str = 'self',
        other_id: str = 'other'
    ) -> SNCircuit:
        """
        This function creates a new circuit combining the neurons of `self` and `other`.
        `outgoing` determines the output neurons in `self` to connect as inputs in
        `other`. `incoming` determines the opposite, which output neurons from `other` to
        connect as inputs of `self`.
        """
        if outgoing is None:
            outgoing = []
        if incoming is None:
            incoming = []

        # neurons = self.neurons + other.neurons
        neurons = {i: Neuron(params=neu.params, synapses=neu.synapses)
                   for i, neu in self.neurons.items()}
        shift_id_other = self.num_neurons
        neurons.update({  # the neuron numerical id's have to be shifted
            i + shift_id_other:
            Neuron(
                params=neu.params,
                synapses={i + shift_id_other:
                          syn for i, syn in neu.synapses.items()}
            )
            for i, neu in other.neurons.items()
        })

        # Each of these calls traverses thru the connections defined in `outgoing` and
        # `incoming`, modifying the `neurons` dictionary to reflect the connections
        # Yes, this could be done with two loops, but having a helper function reduces the
        # lines of code and makes everything more general
        self.__helper_wire_input_to_output(
            other, neurons, outgoing, 0, shift_id_other)
        other.__helper_wire_input_to_output(
            self, neurons, incoming, shift_id_other, 0)

        # construct outputs and inputs as anything that remains unconnected
        conn_outputs_self = set(out for out, in_ in outgoing)
        conn_inputs_other = set(in_ for out, in_ in outgoing)
        conn_outputs_other = set(out for out, in_ in incoming)
        conn_inputs_self = set(in_ for out, in_ in incoming)
        outputs = [
            out for i, out in enumerate(self.outputs)
            if i not in conn_outputs_self
        ]
        outputs.extend(
            out + shift_id_other
            for i, out in enumerate(other.outputs)
            if i not in conn_outputs_other)
        inputs = [
            in_ for i, in_ in enumerate(self.inputs)
            if i not in conn_inputs_self
        ]
        inputs.extend(
            {i + shift_id_other: syn for i, syn in in_.items()}
            for i, in_ in enumerate(other.inputs)
            if i not in conn_inputs_other)

        ids_to_int = {
            f"{self_id}.{id}": i
            for id, i in self.ids_to_int.items()}
        ids_to_int.update({
            f"{other_id}.{id}": i + shift_id_other
            for id, i in other.ids_to_int.items()})

        return SNCircuit(neurons=neurons, outputs=outputs, inputs=inputs, ids_to_int=ids_to_int)

    def __helper_wire_input_to_output(
        this, that: SNCircuit,
        neurons: Dict[int, Neuron],  # mutable!!
        out2in: List[Tuple[int, int]],
        shift_id_this: int = 0,
        shift_id_that: int = 0
    ) -> None:
        for output, input in out2in:
            this.check_correct_output_id(output)
            that.check_correct_input_id(input)

            # connecting self.outputs[self.outputs[output]] to other.inputs[input]
            output_ = this.outputs[output] + shift_id_this
            old_neuron = neurons[output_]
            synapses = old_neuron.synapses.copy()
            synapses.update({
                i + shift_id_that: syn
                for i, syn in that.inputs[input].items()
            })
            neurons[output_] = Neuron(
                params=old_neuron.params,
                synapses=synapses
            )

    def insert_input(self, pos: int, params: Dict[str, SynapParams]) -> SNCircuit:
        inputs = self.inputs.copy()
        inputs.insert(pos, {
            self.ids_to_int[id]: syn for id, syn in params.items()
        })

        return SNCircuit(outputs=self.outputs, inputs=inputs, neurons=self.neurons,
                         ids_to_int=self.ids_to_int)

    def remove_inputs(self, inputs_to_del: List[int]) -> SNCircuit:
        new_inputs = [in_ for i, in_ in enumerate(self.inputs)
                      if i not in inputs_to_del]
        return SNCircuit(outputs=self.outputs, inputs=new_inputs, neurons=self.neurons,
                         ids_to_int=self.ids_to_int)


if __name__ == '__main__':
    with open('snn-circuits/json/clock-smallest.json') as f:
        ex_circuit = SNCircuit.loads_json(f.read(), {'heartbeat': 1/256})

    with open('snn-circuits/json/clock.json') as f:
        ex_circuit2 = SNCircuit.loads_json(f.read(), {'heartbeat': 1/256})

    assert ex_circuit == ex_circuit2, \
        "Both files implement the same network, thus they should be the same"

    with open('gol/json/gol-nonnegative-v3.json') as f:
        stop_circuit = SNCircuit.loads_json(f.read(), {'heartbeat': 1/256})

    with open('snn-circuits/json/turnable-cycle-3.json') as f:
        cycle3_circuit = SNCircuit.loads_json(f.read(), {'heartbeat': 1/256})

    cycle_from_stop_circuit = stop_circuit \
        .connect_with_itself([(0, 0)], remove_outputs=False) \
        .insert_input(pos=0, params={'e': SynapParams(weight=1.0, delay=1)})

    assert cycle_from_stop_circuit.same_as(cycle3_circuit)
