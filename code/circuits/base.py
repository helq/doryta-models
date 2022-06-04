from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray

from .ast import SNFile, Version, ParamName, ParamValue, Token, SynapseParams

from typing import NamedTuple, List, Dict, Optional, Union, Any, Tuple
from collections.abc import Iterable


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
    delay: int


class Neuron(NamedTuple):
    params: NeuronParams
    synapses: Dict[int, SynapParams]


def _raise_if_false(question: bool, msg: str) -> None:
    if not question:
        raise ValueError(msg)


def _params_from_ast_params(  # noqa: C901
    args:        Dict[str, Union[float, int]],
    params_dict: Dict[ParamName, ParamValue],
    defaults:    Optional[Dict[str, Union[float, int]]] = None,
    keys:        Optional[Iterable[str]] = None,
    alias_keys:  Optional[Dict[str, str]] = None
) -> Dict[str, Union[float, int]]:
    """
    This function takes a dictionary of AST parameters and returns a dictionary with no
    "gaps" (variable/argument names). All gaps are filled with the data given by `args`.
    The output dictionary can be initialized with a `defaults` input dict.
    The keys/parameters can be restricted to `keys`.
    `alias_keys` determines the aliases for all parameters
    """
    if defaults is None:
        params = {}
    elif keys is None:
        params = defaults.copy()
    else:
        if alias_keys is not None:
            keys = set(keys).union(alias_keys)
            # checking that final aliases should be contained within keys
            assert len(set(alias_keys.values()) - keys) == 0
        params = {k: defaults[k] for k in keys if k in defaults}

    # Removing converting all aliases into the same key
    if alias_keys is not None:
        for old_key, new_key in alias_keys.items():
            if old_key in params:
                assert new_key not in params
                params[new_key] = params.pop(old_key)

    for name, p_val in params_dict.items():
        # Parameter name
        p_name = name.value
        if alias_keys is not None and p_name in alias_keys:
            p_name = alias_keys[p_name]  # replacing key for alias

        if isinstance(p_val.value, Token):
            token_name = p_val.value.value
            if token_name not in args:
                raise ValueError(f"No argument `{p_val.value}` given")
            # The parameter's value is the one from args
            params[p_name] = args[token_name]
        else:
            # The parameter's value is the one from params
            params[p_name] = p_val.value

    return params


def _synapses_from_ast_synapses(
    ast_synapses: Dict[Token, SynapseParams],
    args: Dict[str, Union[float, int]],
    g_params: Dict[str, Union[float, int]],
    ids_to_int: Dict[str, int],
) -> Dict[int, SynapParams]:
    synapses = {}
    for n, s in ast_synapses.items():
        # n: neuron that synapse points to (Token)
        # s: synapse parameters (SynapseParams)
        s_params = _params_from_ast_params(
            args, s.params, g_params, keys=['weight', 'delay'])
        assert type(s_params['delay']) is int
        synapses[ids_to_int[n.value]] = SynapParams(**s_params)  # type: ignore
    return synapses


class SNCircuit(NamedTuple):
    """
    Contains the description of a Spiking Neural Circuit. To be used anywhere else.

    Treat SNCircuits as immutable objects.
    Careful: all operations between `SNCircuit`s return new SNCircuit objects which are
    NOT deep copies of the originals. Any alteration to the objects might be reflected on
    their parent objects or children.
    """
    # TODO: change the type of all variables from mutable objects to immutable
    outputs: List[int]
    inputs: List[Dict[int, SynapParams]]
    neurons: Dict[int, Neuron]
    ids_to_int: Dict[str, int]

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

    @classmethod
    def loads_json(
        cls,
        data: str,
        args: Dict[str, float]
    ) -> SNCircuit:
        ast_circuit = SNFile.from_json(data)
        return SNCircuit.from_ast(ast_circuit, args)

    @classmethod
    def from_ast(
        cls,
        ast: SNFile,
        args: Dict[str, float]
    ) -> SNCircuit:
        _raise_if_false(ast.version == Version(0, 0, 1), f"Version {ast.version} unsupported")
        _raise_if_false(sorted(arg.value for arg in ast.args) == sorted(args.keys()),
                        "Arguments given do not correspond to arguments needed")

        defaults_params = {
            'potential': 0.0,
            'current': 0.0,
            'resting_potential': 0.0,
            'reset_potential': 0.0,
        }
        g_params = _params_from_ast_params(args, ast.params, defaults_params)
        ids_to_int = {k.value: i for i, k in enumerate(ast.neurons.keys())}

        neurons = {}
        for neuron_tok, neuron_def in ast.neurons.items():
            n_params = _params_from_ast_params(
                args, neuron_def.params, g_params,
                keys=['resistance', 'capacitance', 'threshold', 'potential', 'current',
                      'resting_potential', 'reset_potential'],
                alias_keys={'R': 'resistance', 'C': 'capacitance'})

            neurons[ids_to_int[neuron_tok.value]] = Neuron(
                params=NeuronParams(**n_params),
                synapses=_synapses_from_ast_synapses(
                    neuron_def.synapses, args, g_params, ids_to_int)
            )

        return SNCircuit(
            outputs=[ids_to_int[o.value] for o in ast.outputs],
            inputs=[_synapses_from_ast_synapses(input, args, g_params, ids_to_int)
                    for input in ast.inputs],
            neurons=neurons,
            ids_to_int=ids_to_int)

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
