from __future__ import annotations

# Note: this code requires Python 3.7+ because it depends on ordered dicts
# Note: this code requires Python 3.9+ because of typing from generics

import re

import numpy as np
from numpy.typing import NDArray

from types import TracebackType
from typing import NamedTuple, Optional, Any, Union
from collections.abc import Iterable, Container

token_re = re.compile(r'[a-z][a-zA-Z0-9/_-]*')
circuit_re = re.compile(r'[A-Z][a-zA-Z0-9/_-]*')


class LIF(NamedTuple):
    resistance: float
    capacitance: float
    threshold: float
    potential: float = 0
    current: float = 0
    resting_potential: float = 0
    reset_potential: float = 0


NeuronType = Union[LIF]


class SynapParams(NamedTuple):
    weight: float
    delay: int


class Neuron(NamedTuple):
    params: NeuronType
    synapses: dict[int, SynapParams]


class SNCircuit(NamedTuple):
    """
    Contains the description of a Spiking Neural Circuit. To be used anywhere else.

    Treat SNCircuits as immutable objects.
    Careful: all operations between `SNCircuit`s return new SNCircuit objects which are
    NOT deep copies of the originals. Any alteration to the objects might be reflected on
    their parent objects or children.
    """
    # TODO: change the type of all variables from mutable objects to immutable
    outputs: list[int]
    inputs: list[dict[int, SynapParams]]
    inputs_id: dict[str, int]
    neurons: dict[int, Neuron]
    ids_to_int: dict[str, int]

    def same_as(self, other: SNCircuit) -> bool:
        """
        Less strict "equal" operation. This ignores neuron ids.
        """
        return self.outputs == other.outputs \
            and self.inputs == other.inputs \
            and self.inputs_id == self.inputs_id \
            and self.neurons == other.neurons

    def get_params_in_bulk(self) -> dict[str, NDArray[Any]]:
        neuron_args = {
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
        out2in: list[tuple[int, int]],
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
        inputs_id = {id: in_ for id, in_ in self.inputs_id.items()
                     if in_ not in conn_inputs}

        return SNCircuit(neurons=neurons, outputs=outputs, inputs=inputs,
                         inputs_id=inputs_id, ids_to_int=self.ids_to_int)

    def connect(
        self,
        other: SNCircuit,
        outgoing: Optional[list[tuple[int, int]]] = None,
        incoming: Optional[list[tuple[int, int]]] = None,
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

        return SNCircuit(neurons=neurons, outputs=outputs, inputs_id={},
                         inputs=inputs, ids_to_int=ids_to_int)

    def __helper_wire_input_to_output(
        this, that: SNCircuit,
        neurons: dict[int, Neuron],  # mutable!!
        out2in: list[tuple[int, int]],
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

    def insert_input(self, pos: int, params: dict[str, SynapParams]) -> SNCircuit:
        inputs = self.inputs.copy()
        inputs.insert(pos, {
            self.ids_to_int[id]: syn for id, syn in params.items()
        })

        # shifting input `ids`
        inputs_id = {id: (int_ if int_ < pos else int_ + 1)
                     for id, int_ in self.inputs_id.items()}

        return SNCircuit(outputs=self.outputs, inputs_id=inputs_id, inputs=inputs,
                         neurons=self.neurons, ids_to_int=self.ids_to_int)

    def remove_inputs(self, inputs_to_del: Container[int]) -> SNCircuit:
        new_inputs = [in_ for i, in_ in enumerate(self.inputs)
                      if i not in inputs_to_del]

        inputs_id = {id: int_ for id, int_ in self.inputs_id.items()
                     if int_ not in inputs_to_del}

        return SNCircuit(outputs=self.outputs, inputs=new_inputs, inputs_id=inputs_id,
                         neurons=self.neurons, ids_to_int=self.ids_to_int)


class SNCreate:
    def __init__(
        self,
        neuron_type: type[NeuronType] = LIF,
        neuron_params: Optional[dict[str, Union[int, float]]] = None,
        synapse_params: Optional[dict[str, Union[int, float]]] = None
    ) -> None:
        self.neuron_type = neuron_type
        self.neuron_params = neuron_params if neuron_params is not None else {}
        self.synapse_params = synapse_params if synapse_params is not None else {}
        self._outputs: list[str] = []
        self._inputs: dict[str, tuple[dict[str, SynapParams], list[str]]] = {}
        self._neurons: dict[str, tuple[NeuronType, dict[str, SynapParams]]] = {}
        # TODO: replace these _include variables for the circuits themselves. The current
        # strategy is simple to implement but wastes a lot of memory
        self._include_circuit_names: set[str] = set()
        self._include_inputs: dict[str, dict[str, SynapParams]] = {}
        self._include_output_aliases: dict[str, str] = {}
        self.__circuit: Optional[SNCircuit] = None

    def __enter__(self) -> SNCreate:
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        if exc_type is None:
            self.generate()

    @property
    def circuit(self) -> SNCircuit:
        if self.__circuit is None:
            raise AttributeError("SNCreate has to be closed to generate a `circuit`")
        return self.__circuit

    def output(self, output: str) -> None:
        self._outputs.append(output)

    def input(self, name: str,
              synapses: Optional[dict[str, dict[str, Union[int, float]]]] = None,
              inputs: Optional[Iterable[str]] = None) -> None:

        # TODO: check that input does not follow the convention `out_X` or `in_X`, those
        # are reserved tokens

        if inputs is None and synapses is None:
            raise ValueError("Both `synapses` and `inputs` cannot be None at the same time")
        match = token_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid input name")

        synapses_for_input = {}
        if synapses is not None:
            for neu_id, params in synapses.items():
                params = self.synapse_params | params
                assert 'delay' in params and isinstance(params['delay'], int)
                synapses_for_input[neu_id] = SynapParams(**params)  # type: ignore

        inputs_for_input = [] if inputs is None else list(inputs)

        self._inputs[name] = (synapses_for_input, inputs_for_input)

    def neuron(self, name: str,
               params: Optional[dict[str, Union[int, float]]] = None,
               synapses: Optional[dict[str, dict[str, Union[int, float]]]] = None) -> None:

        # TODO: check that input does not follow the convention `out_X` or `in_X`, those
        # are reserved tokens

        match = token_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid neuron name")

        neuron_params = self.neuron_params if params is None \
            else self.neuron_params | params
        neuron_ = self.neuron_type(**neuron_params)

        neuron_synapses = {}
        if synapses is not None:
            for neu_id, syn_params in synapses.items():
                syn_params = self.synapse_params | syn_params
                assert 'delay' in syn_params and isinstance(syn_params['delay'], int)
                neuron_synapses[neu_id] = SynapParams(**syn_params)  # type: ignore

        self._neurons[name] = (neuron_, neuron_synapses)

    def include(self, name: str, circuit: SNCircuit) -> None:
        match = circuit_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid circuit name")
        if name in self._include_circuit_names:
            raise ValueError(f"Circuit with `{name}` has already been included")
        self._include_circuit_names.add(name)

        ints_to_id = {i: id for id, i in circuit.ids_to_int.items()}

        # for each output, an alias is generated (out_X)
        for i, out in enumerate(circuit.outputs):
            self._include_output_aliases[f"{name}.out_{i}"] = f"{name}.{ints_to_id[out]}"

        # inputs from the included circuit. Does not polute variable `_inputs`
        inputs_int_to_id = {i: id for id, i in circuit.inputs_id.items()}
        for i, inp in enumerate(circuit.inputs):
            synapses = {f"{name}.{ints_to_id[i]}": params for i, params in inp.items()}
            self._include_inputs[f"{name}.{inputs_int_to_id[i]}"] = synapses
            self._include_inputs[f"{name}.in_{i}"] = synapses

        # each neuron is converted from `int` to extended name `name.neuron_id`
        for neu_i, neu in circuit.neurons.items():
            self._neurons[f"{name}.{ints_to_id[neu_i]}"] = (neu.params, {
                f"{name}.{ints_to_id[i]}": syn_params
                for i, syn_params in neu.synapses.items()
            })

    def _check_iter_is_contained_in_neurons(
        self, name: str, iter: Iterable[str],
        include_output_aliases: bool = False
    ) -> None:
        not_in_neurons = set(v for v in iter if v not in self._neurons)
        if include_output_aliases:
            not_in_neurons = set(v for v in not_in_neurons
                                 if v not in self._include_output_aliases)
        if not_in_neurons:
            raise Exception(f"{name} {not_in_neurons} are not neurons")

    def generate(self) -> SNCircuit:
        self._check_iter_is_contained_in_neurons("Outputs", self._outputs, True)

        ids_to_int = {id: i for i, id in enumerate(self._neurons)}
        inputs_id = {id: i for i, id in enumerate(self._inputs)}
        outputs = [ids_to_int[id] if id in ids_to_int
                   else ids_to_int[self._include_output_aliases[id]]
                   for id in self._outputs]

        inputs = []
        for inp, (synapses, inputs_to_inp) in self._inputs.items():
            self._check_iter_is_contained_in_neurons(f"Input `{inp}` synapses", synapses)
            synapses_input = {ids_to_int[neu_id]: synap_params
                              for neu_id, synap_params in synapses.items()}
            for inp_to_inp in inputs_to_inp:
                if inp_to_inp not in self._include_inputs:
                    raise Exception(f"Input `{inp_to_inp}` cannot be found")
                # TODO: raise another exception if there are duplicated connections (if
                # the union below replaces values)
                synapses_input |= {ids_to_int[neu_id]: synap_params
                                   for neu_id, synap_params in
                                   self._include_inputs[inp_to_inp].items()}
            inputs.append(synapses_input)

        neurons = {}
        for neu_id, (neu_params, synapses) in self._neurons.items():
            self._check_iter_is_contained_in_neurons(f"Neuron `{neu_id}` synapses", synapses)
            neurons[ids_to_int[neu_id]] = Neuron(
                params=neu_params,
                synapses={ids_to_int[n_id]: synap_params
                          for n_id, synap_params in synapses.items()})

        self.__circuit = SNCircuit(outputs=outputs, inputs=inputs, inputs_id=inputs_id,
                                   neurons=neurons, ids_to_int=ids_to_int)
        return self.__circuit
