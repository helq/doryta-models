from __future__ import annotations

# Note: this code requires Python 3.7+ because it depends on ordered dicts
# Note: this code requires Python 3.9+ because of typing from generics

import re
import sys
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from types import TracebackType
from typing import NamedTuple, Optional, Any, Union, Literal
from collections.abc import Iterable, Container

from .visualize import base as vis, positioning

token_re = re.compile(r'[a-z/_+-][a-zA-Z0-9/_+-]*')
circuit_re = re.compile(r'[A-Z][a-zA-Z0-9/_+-]*')
reserved_ids_re = re.compile('(out|in)_([0-9]+)')
incd_token_re = re.compile(rf'({circuit_re.pattern})\.(.+)')
incd_out_re = re.compile(rf'({circuit_re.pattern})\.out_([0-9]+)')
incd_in_re = re.compile(rf'({circuit_re.pattern})\.in_([0-9]+)')


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
    outputs: list[frozenset[int]]
    inputs: list[dict[int, SynapParams]]
    neurons: dict[int, Neuron]
    # outputs_id: dict[str, int]
    inputs_id: dict[str, int]
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
            for out_id in self.outputs[output]:
                old_neuron = neurons[out_id]
                synapses = old_neuron.synapses.copy()
                synapses.update(self.inputs[input])
                neurons[out_id] = Neuron(
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
            frozenset(out_id + shift_id_other for out_id in out)
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
            for output_ in this.outputs[output]:
                out_id = output_ + shift_id_this
                old_neuron = neurons[out_id]
                synapses = old_neuron.synapses.copy()
                synapses.update({
                    i + shift_id_that: syn
                    for i, syn in that.inputs[input].items()
                })
                neurons[out_id] = Neuron(
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
        # Only params without dash (_) are "stable" api, everything else (below) is not
        self._outputs: list[frozenset[str]] = []
        self._inputs: dict[str, tuple[dict[str, SynapParams], list[str]]] = {}
        self._neurons: dict[str, tuple[NeuronType, dict[str, SynapParams], frozenset[str]]] = {}
        self._connections: dict[str, list[tuple[str, Optional[SynapParams]]]] = {}
        self._include_circuit_names: set[str] = set()
        # TODO: replace these _include variables for the circuits themselves. The current
        # strategy is simple to implement but wastes a lot of memory (SNCreate consumes
        # SNCircuits and generates SNCircuits that are fed to other SNCreate, potentially)
        self._include_inputs: dict[str, dict[str, SynapParams]] = {}
        self._include_output_aliases: dict[str, frozenset[str]] = {}
        # Visuals and SNCircuits for includes are used when generating larger Visuals
        self._include_circuit_obj: dict[str, SNCircuit] = {}
        # These are generated only once and returned every single time after that
        self.__circuit: Optional[SNCircuit] = None
        self.__visual: Optional[SNCreateVisual] = None

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

    def output(self, output: str | Iterable[str]) -> None:
        assert self.__circuit is None
        if isinstance(output, str):
            self._outputs.append(frozenset({output}))
        else:
            self._outputs.append(frozenset(output))

    def input(self, name: str,
              synapses: Optional[dict[str, dict[str, int | float]] | set[str]] = None,
              inputs: Optional[Iterable[str]] = None) -> None:
        assert self.__circuit is None

        if reserved_ids_re.match(name) is not None:
            raise ValueError(f"`{name}` is a reserved name and cannot be used as a neuron name")

        if inputs is None and synapses is None:
            raise ValueError("Both `synapses` and `inputs` cannot be None at the same time")
        if isinstance(synapses, set):
            synapses = {n: {} for n in synapses}
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
               synapses: Optional[dict[str, dict[str, int | float]] | set[str]] = None,
               to_inputs: Optional[Iterable[str]] = None
               ) -> None:
        assert self.__circuit is None

        # TODO: check that input does not follow the convention `out_X` or `in_X`, those
        # are reserved tokens

        match = token_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid neuron name")
        if isinstance(synapses, set):
            synapses = {s: {} for s in synapses}

        neuron_params = self.neuron_params if params is None \
            else self.neuron_params | params
        neuron_ = self.neuron_type(**neuron_params)

        neuron_synapses = {}
        if synapses is not None:
            for neu_id, syn_params in synapses.items():
                syn_params = self.synapse_params | syn_params
                assert 'delay' in syn_params and isinstance(syn_params['delay'], int)
                neuron_synapses[neu_id] = SynapParams(**syn_params)  # type: ignore

        to_inputs = frozenset() if to_inputs is None else frozenset(to_inputs)

        self._neurons[name] = (neuron_, neuron_synapses, to_inputs)

    def include(
        self,
        name: str,
        circuit: SNCircuit
    ) -> None:
        self.include_sncircuit(name, circuit)

    def include_sncircuit(self, name: str, circuit: SNCircuit) -> None:
        assert self.__circuit is None
        match = circuit_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid circuit name")
        if name in self._include_circuit_names:
            raise ValueError(f"Circuit with `{name}` has already been included")
        self._include_circuit_names.add(name)
        self._include_circuit_obj[name] = circuit

        ints_to_id = {i: id for id, i in circuit.ids_to_int.items()}

        # for each output, an alias is generated (out_X)
        for i, out in enumerate(circuit.outputs):
            self._include_output_aliases[f"{name}.out_{i}"] = frozenset(
                f"{name}.{ints_to_id[out_id]}" for out_id in out)

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
            }, frozenset())

    def connection(
        self, from_: str, to: str,
        synapse_params: Optional[dict[str, Union[int, float]]] = None
    ) -> None:
        assert self.__circuit is None
        val: tuple[str, Optional[SynapParams]] = (to, None)
        if synapse_params is not None:
            syn_params = self.synapse_params | synapse_params
            assert 'delay' in syn_params and isinstance(syn_params['delay'], int)
            val = (to, SynapParams(**syn_params))  # type: ignore
        if from_ not in self._connections:
            self._connections[from_] = []
        self._connections[from_].append(val)

    def _check_iter_is_contained_in_neurons(
        self, name: str, iter: Iterable[str],
        consider_output_aliases: bool = False,
        consider_input_aliases: bool = False,
    ) -> None:
        not_in_neurons = set(v for v in iter if v not in self._neurons)
        if consider_output_aliases:
            not_in_neurons = set(v for v in not_in_neurons
                                 if v not in self._include_output_aliases)
        if consider_input_aliases:
            not_in_neurons = set(v for v in not_in_neurons
                                 if v not in self._include_inputs)
        if not_in_neurons:
            raise Exception(f"{name} {not_in_neurons} are not neurons")

    def _out_ids_to_out_ints(
        self, ids_to_int: dict[str, int], out_set: frozenset[str]
    ) -> frozenset[int]:
        to_ret = set()
        for out_id in out_set:
            # out_id is a neuron
            if out_id in ids_to_int:
                to_ret.add(ids_to_int[out_id])
                continue
            # out_id is an output from an `include`
            for id_ in self._include_output_aliases[out_id]:
                if id_ not in ids_to_int:
                    raise Exception(f"Neuron `{id_}` couldn't be found")
                to_ret.add(ids_to_int[id_])
        return frozenset(to_ret)

    def _check_additional_connections_and_simplify(  # noqa: C901
        self
    ) -> dict[str, dict[str, SynapParams]]:
        """
        Checks that all additional connections are possible and dissentangles all
        `from_` outputs (groups of neurons) into only neurons, and `to`s into neurons.
        This way, neurons `_connections` can be cleanely used by the generator function
        and no need for more fancy complex coding.

        Yes, this function its too complex according to flake8, but it's just because of
        all the loops. Totally necessary loops btw
        """
        simp_connections: dict[frozenset[str], dict[str, SynapParams]] = {}
        for from_, to_conns in self._connections.items():
            if from_ in self._include_output_aliases:
                from_list = self._include_output_aliases[from_]
            elif from_ in self._neurons:
                from_list = frozenset({from_})
            else:
                raise Exception(f"Neuron/output `{from_}` couldn't be found")

            simp_connections[from_list] = {}
            for to, params in to_conns:
                if params is None:
                    if to in self._include_inputs:
                        simp_connections[from_list] |= self._include_inputs[to].items()
                    elif to in self._neurons:
                        raise Exception(f"`{to}` should be an input because no synapse "
                                        "parameters were given but it's a neuron")
                    else:
                        raise Exception(f"Input `{to}` couldn't be found")
                else:
                    if to in self._neurons:
                        simp_connections[from_list][to] = params
                    elif to in self._include_inputs:
                        raise Exception(f"`{to}` should be a neuron because synapse parameters "
                                        "were defined for it but it's an input")
                    else:
                        raise Exception(f"Neuron `{to}` couldn't be found")

        to_return: dict[str, dict[str, SynapParams]] = {}
        for from_list, conns in simp_connections.items():
            for from_ in from_list:
                if from_ in to_return:
                    to_return[from_] |= dict(conns)
                else:
                    to_return[from_] = conns
        return to_return

    def generate(self) -> SNCircuit:
        self._check_iter_is_contained_in_neurons(
            "Outputs", [out for out_s in self._outputs for out in out_s], True)
        simpf_connections = self._check_additional_connections_and_simplify()

        ids_to_int = {id: i for i, id in enumerate(self._neurons)}
        inputs_id = {id: i for i, id in enumerate(self._inputs)}
        outputs = [self._out_ids_to_out_ints(ids_to_int, out_s)
                   for out_s in self._outputs]

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
        for neu_id, (neu_params, synapses, to_inputs) in self._neurons.items():
            self._check_iter_is_contained_in_neurons(f"Neuron `{neu_id}` synapses", synapses)
            # Adding new synapses from connections defined before
            if neu_id in simpf_connections:
                synapses |= simpf_connections[neu_id]
            n_synapses = {ids_to_int[n_id]: synap_params
                          for n_id, synap_params in synapses.items()}
            for input in to_inputs:
                if input not in self._include_inputs:
                    raise Exception(f"Input `{input}` cannot be found")
                # TODO: see above
                n_synapses |= {ids_to_int[neu_id]: synap_params
                               for neu_id, synap_params in self._include_inputs[input].items()}
            neurons[ids_to_int[neu_id]] = Neuron(params=neu_params, synapses=n_synapses)

        self.__circuit = SNCircuit(outputs=outputs, inputs=inputs, inputs_id=inputs_id,
                                   neurons=neurons, ids_to_int=ids_to_int)
        return self.__circuit

    def is_synapse_start(self, from_: str) -> bool:
        match_out = incd_out_re.match(from_)
        from_in_outputnames = False
        if match_out:
            name_inc, out_i = match_out[1], int(match_out[2])
            include_obj_ = self._include_circuit_obj[name_inc]
            from_in_outputnames = out_i < len(include_obj_.outputs)
        return from_ in self._neurons or from_ in self._inputs \
            or from_in_outputnames

    def is_synapse_end(self, to: str) -> bool:
        match_out = reserved_ids_re.match(to)
        match_in = incd_in_re.match(to)
        match_token = incd_token_re.match(to)
        to_in_outputs = False
        to_in_inputnames = False
        if match_out:
            res, out_i = match_out[1], int(match_out[2])
            to_in_outputs = (res == 'out') and out_i < len(self._outputs)
        if match_in:
            name_inc, in_i = match_in[1], int(match_in[2])
            include_obj_ = self._include_circuit_obj[name_inc]
            to_in_inputnames = in_i < len(include_obj_.inputs)
        elif match_token:
            name_inc, token_name = match_token[1], match_token[2]
            include_obj_ = self._include_circuit_obj[name_inc]
            to_in_inputnames = token_name in include_obj_.inputs_id
        return to in self._neurons or to_in_outputs or to_in_inputnames


class ElementType(Enum):
    neuron = 0
    input = 1
    output = 2


class SNCreateVisual:
    def __init__(
        self,
        sncreate: SNCreate,
        graph_drawing: positioning.SugiyamaGraphDrawing | None = None
    ):
        self._sncreate = sncreate
        self._size: Optional[vis.Size] = None
        self._inputs: list[vis.Pos] = []
        self._outputs: list[vis.Pos] = []
        self._nodes: dict[str, vis.Node] = {}
        self._includes: dict[str, vis.Pos] = {}
        self._arrows: dict[str, dict[str, list[vis.Pos]]] = {}
        self._include_visuals: dict[str, vis.CircuitDisplay] = {}
        self.__circuit_display: Optional[vis.CircuitDisplay] = None
        self._graph_drawing = graph_drawing

    def _get_path_for_connection(self, from_: str, to: str) -> Optional[list[vis.Pos]]:
        if from_ in self._arrows and to in self._arrows[from_]:
            return self._arrows[from_][to]
        return None

    def _check_all_neurons_pos_defined(self) -> None:
        neurons = {n for n in self._sncreate._neurons if token_re.fullmatch(n) is not None}
        pos_not_defined = neurons - set(self._nodes)
        if pos_not_defined:
            raise Exception("Position for some neurons not yet defined. Not defined "
                            f"for: {pos_not_defined}")

    def _check_all_includes_are_defined(self) -> None:
        pos_not_defined = set(self._sncreate._include_circuit_names) - set(self._includes)
        if pos_not_defined:
            raise Exception("Position for some included circuits not yet defined. Not "
                            f"defined for: {pos_not_defined}")

    def _find_element_pos(self, name: str) -> tuple[vis.Pos, ElementType]:
        """
        Find neuron, input or output position for a given name
        """
        match_incd_out = incd_out_re.match(name)
        match_incd_in = incd_in_re.match(name)
        match_incd = incd_token_re.match(name)
        match_neuron = token_re.match(name)
        weird_element = False

        any_match_inc = match_incd_out or match_incd_in or match_incd
        # The element is part of an include
        if any_match_inc:
            # extract include name and neuron name inside the included circuit
            name_inc, name_elem = any_match_inc[1], any_match_inc[2]
            include_ = self._get_include_visual(name_inc)
            include_offset = self._includes[name_inc]
            include_obj = self._sncreate._include_circuit_obj[name_inc]

            # The element is an output inside an include
            if match_incd_out:
                # extract include and output id from out_name
                out_i = int(name_elem)
                pos = include_offset + include_.outputs[out_i]
                elem_type = ElementType.output
            # The element is an output inside an include
            elif match_incd_in:
                # extract include and output id from out_name
                in_i = int(name_elem)
                pos = include_offset + include_.outputs[in_i]
                elem_type = ElementType.input
            # The element is a neuron inside an included circuit
            elif match_incd:
                # finding specific neuron position inside the included circuit
                if name_elem in include_obj.ids_to_int:
                    neu_i = include_obj.ids_to_int[name_elem]
                    pos = include_offset + include_.nodes[neu_i]
                    elem_type = ElementType.neuron
                else:
                    assert name_elem in include_obj.inputs_id
                    in_i = include_obj.inputs_id[name_elem]
                    pos = include_offset + include_.inputs[in_i]
                    elem_type = ElementType.input
            else:
                weird_element = True
        # The element is a neuron defined by the top circuit
        elif match_neuron:
            assert name in self._nodes
            pos = vis.Pos(*self._nodes[name])
            elem_type = ElementType.neuron
        else:
            weird_element = True

        if weird_element:
            raise Exception(f"This is weird, `{name}` doesn't correspond to a neuron in "
                            "the circuit nor the included circuits")

        return pos, elem_type

    def _add_connections_from_inputs(
        self, connections: list[vis.Connection]
    ) -> None:
        for i, (name_in, (synaps, to_inputs)) in enumerate(self._sncreate._inputs.items()):
            # connections to neurons
            connections.extend(
                self._connection_for(i, syn_name, ElementType.input, ElementType.neuron)
                for syn_name in synaps
            )

            # connections to inputs from includes
            connections.extend(
                self._connection_for(i, to_in, ElementType.input)
                for to_in in to_inputs
            )

    def _add_connections_for_outputs(self, connections: list[vis.Connection]) -> None:
        connections.extend(
            self._connection_for(out_name, i,
                                 (ElementType.neuron, ElementType.output),
                                 ElementType.output)
            for i, out_set in enumerate(self._sncreate._outputs)
            for out_name in out_set
        )

    def _add_connections_from_neurons(
        self, connections: list[vis.Connection]
    ) -> None:
        # creating connections between neurons
        for neu_name, (neu_params, synaps, to_inputs) \
                in self._sncreate._neurons.items():

            # only neurons defined in circuit are worth adding
            match = token_re.match(neu_name)
            if match is None:
                continue

            # connections to other neurons
            connections.extend(
                self._connection_for(neu_name, syn_name, ElementType.neuron, ElementType.neuron)
                for syn_name in synaps
            )

            # connections to inputs from includes
            connections.extend(
                self._connection_for(neu_name, to_in, ElementType.neuron, ElementType.input)
                for to_in in to_inputs
            )

    def _add_connections_by_explicitely_def_conns(
        self, connections: list[vis.Connection]
    ) -> None:
        # creating custom connections
        connections.extend(
            self._connection_for(elem_name, conn)
            for elem_name, elem_conns in self._sncreate._connections.items()
            for conn, _ in elem_conns
        )

    def _find_element_pos_restricted_to_type(
        self,
        elem: str | int,
        elem_type: Optional[tuple[ElementType, ...]]
    ) -> tuple[str, vis.Pos, ElementType]:
        if isinstance(elem, str):
            pos, elem_t = self._find_element_pos(elem)
        else:  # isinstance(from_, int)
            assert elem_type is not None and len(elem_type) == 1 \
                and elem_type[0] != ElementType.neuron, \
                "If the element is an int, the type of the element must be either " \
                f"`input` or `output`, but it is `{elem_type}`"

            if elem_type[0] == ElementType.input:
                pos = self._inputs[elem]
                elem = f"in_{elem}"
                # elem = list(self._sncreate._inputs.keys())[elem]
            else:  # elem_type[0] == ElementType.output:
                pos = self._outputs[elem]
                elem = f"out_{elem}"
            elem_t = elem_type[0]

        # This should never be triggered if all the other checks are in place. It helps
        # debugging inconsistencies
        if elem_type and elem_t not in elem_type:
            raise Exception(f"`{elem}` should be of type {elem_type} but it is "
                            f"of type {elem_t}")
        return elem, pos, elem_t

    def _connection_for(
        self,
        from_: str | int,
        to: str | int,
        from_type: ElementType | tuple[ElementType, ...] | None = None,
        to_type: ElementType | tuple[ElementType, ...] | None = None
    ) -> vis.Connection:
        # Converting from_type and to_type into tuples
        if isinstance(from_type, ElementType):
            from_type = (from_type,)
        if isinstance(to_type, ElementType):
            to_type = (to_type,)

        # Actual code to elements positions!
        from_, from_pos, from_t = self._find_element_pos_restricted_to_type(from_, from_type)
        to, to_pos, to_t = self._find_element_pos_restricted_to_type(to, to_type)

        return vis.straight_line_connection(
            from_=from_pos,
            to=to_pos,
            from_size=0.5 if from_t == ElementType.neuron else 0.125,
            to_size=0.5 if to_t == ElementType.neuron else 0.125,
            path=self._get_path_for_connection(from_, to)
        )

    def _get_include_visual(self, name_inc: str) -> vis.CircuitDisplay:
        if name_inc not in self._include_visuals:
            raise Exception(f"No visuals provided for include {name_inc}")
        return self._include_visuals[name_inc]

    def _get_graph_drawing(self) -> positioning.SugiyamaGraphDrawing:
        if self._graph_drawing is None:
            return positioning.SugiyamaGraphDrawing(
                remove_cycles=positioning.RemoveCycleDFS(reverse=True),
                layer_assignment=positioning.LayerAssignmentCoffmanGraham(
                    w=2, crossings_in_layer=1
                ))
        return self._graph_drawing

    def _use_graph_drawing_to_fill_in_params(self) -> None:
        # Defining graph from circuit data
        circuit = self._sncreate.circuit
        n_inputs = len(circuit.inputs)
        n_outputs = len(circuit.outputs)

        max_neu_id = max(circuit.neurons) + 10
        input_nodes = list(range(max_neu_id, max_neu_id + n_inputs))
        max_neu_id += n_inputs
        output_nodes = list(range(max_neu_id, max_neu_id + n_outputs))

        # Extending edges with input and output connections
        edges: dict[int, set[int]] = \
            {n: set(synapses) for n, (_, synapses) in circuit.neurons.items()} \
            | {n: set(synapses) for n, synapses in zip(input_nodes, circuit.inputs)} \
            | {n: set() for n in output_nodes}
        for out, neurs in zip(output_nodes, circuit.outputs):
            for n in neurs:
                edges[n].add(out)
        graph = positioning.Graph(
            vertices=set(circuit.neurons) | set(input_nodes) | set(output_nodes),
            edges=edges
        )
        pretty_g = self._get_graph_drawing().find_pos(
            graph,
            inputs=input_nodes if input_nodes else None,
            outputs=output_nodes if output_nodes else None
        )

        ints_to_id = {i: id for id, i in circuit.ids_to_int.items()} \
            | {n: f'in_{i}' for i, n in enumerate(input_nodes)} \
            | {n: f'out_{i}' for i, n in enumerate(output_nodes)}
        ignore_nodes = set(input_nodes) | set(output_nodes)

        x_shift = 0 if input_nodes else 1

        # Converting positions defined by graph drawing algorithm into
        # coordinates for SNCircuit to present
        self._nodes = {
            ints_to_id[v]: vis.Node(pos[0] * 1.5 + x_shift + .75, pos[1] * 1.5 + 1)
            for v, pos in pretty_g.vertices.items()
            if v not in ignore_nodes}
        self._arrows = {}
        for v, ws in pretty_g.edges.items():
            for w, path in ws:
                # ignoring connections that don't have intermediate steps
                if path:
                    from_ = ints_to_id[v]
                    if from_ not in self._arrows:
                        self._arrows[from_] = {}
                    self._arrows[from_][ints_to_id[w]] = [
                        vis.Pos(pos[0] * 1.5 + x_shift + .75, pos[1] * 1.5 + 1)
                        for pos in path]

        out_shift = 0 if output_nodes else 1
        self._size = vis.Size(pretty_g.width * 1.5 + x_shift + out_shift,
                              pretty_g.height * 1.5 + .5)

        n_inputs = len(self._sncreate._inputs)
        n_outputs = len(self._sncreate._outputs)
        self._inputs = [vis.Pos(0, i * 1.5 + 1) for i in range(n_inputs)]
        self._outputs = [vis.Pos(self._size.x, i * 1.5 + 1) for i in range(n_outputs)]

    def generate(
        self,
        mode: Literal['auto', 'manual', 'empty'] | None = None
    ) -> vis.CircuitDisplay:
        if self.__circuit_display is not None:
            return self.__circuit_display
        if mode is None:
            mode = 'manual' if self._nodes or self._includes else 'empty'

        # Filling in parameters automatically if auto mode has been selected
        if mode == 'auto':
            if self._nodes or self._includes:
                raise Exception("No node or include location can be assigned in `auto` mode")
            if self._sncreate._include_circuit_names:
                raise Exception("`auto` mode only works when there are no includes")
            self._use_graph_drawing_to_fill_in_params()
        elif mode == 'manual':
            # all nodes and includes must be defined
            self._check_all_includes_are_defined()
            self._check_all_neurons_pos_defined()

        # Defining extra parameters
        n_inputs = len(self._sncreate._inputs)
        n_outputs = len(self._sncreate._outputs)
        size = vis.Size(1, max(n_inputs, n_outputs)) if self._size is None else self._size
        if not self._inputs:
            self._inputs = [vis.Pos(0, i + 0.5) for i in range(n_inputs)]
        else:
            assert len(self._inputs) == len(self._sncreate._inputs)
        if not self._outputs:
            self._outputs = [vis.Pos(size.x, i + 0.5) for i in range(n_outputs)]
        else:
            assert len(self._outputs) == len(self._sncreate._outputs)

        nodes_pos = {}
        connections: list[vis.Connection] = []
        includes: list[tuple[vis.Pos, vis.CircuitDisplay]] = []

        if mode != 'empty':
            # defining position of nodes and includes
            nodes_pos = {self._sncreate.circuit.ids_to_int[name]: pos
                         for name, pos in self._nodes.items()}
            includes = [(pos_in, self._get_include_visual(name_in))
                        for name_in, pos_in in self._includes.items()]

            # creating connections from inputs, to outputs, between neurons and includes
            self._add_connections_from_inputs(connections)
            self._add_connections_for_outputs(connections)
            self._add_connections_from_neurons(connections)
            self._add_connections_by_explicitely_def_conns(connections)

        self.__circuit_display = vis.CircuitDisplay(
            name=None,
            size=size,
            nodes=nodes_pos,
            inputs=self._inputs,
            outputs=self._outputs,
            connections=connections,
            includes=includes
        )
        return self.__circuit_display

    def def_node_pos(self, name: str, pos: tuple[float, float]) -> None:
        self.__visual = None
        match = token_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid neuron name")
        if name not in self._sncreate._neurons:
            raise ValueError(f"Neuron `{name}` has not been defined in circuit")
        self._nodes[name] = vis.Node(*pos)

    def def_include_pos(self, name: str, pos: tuple[float, float]) -> None:
        self.__visual = None
        match = circuit_re.fullmatch(name)
        if match is None:
            raise ValueError(f"`{name}` is not a valid circuit include name")
        if name not in self._sncreate._include_circuit_names:
            raise ValueError(f"No circuit named `{name}` has been included. "
                             "Names of included circuits: "
                             f"{self._sncreate._include_circuit_names}")
        self._includes[name] = vis.Pos(*pos)

    def def_size(self, width: float, height: float) -> None:
        self.__visual = None
        assert width > 0 and height > 0
        self._size = vis.Size(width, height)

    def def_path(self, from_: str, to: str, path: list[tuple[float, float]]) -> None:
        self.__visual = None

        # Checking that start and end points of path exists
        if not self._sncreate.is_synapse_start(from_):
            raise ValueError(f"No such neuron, input or output named `{from_}`")
        if not self._sncreate.is_synapse_end(to):
            raise ValueError(f"No such neuron, input or output named `{to}`")

        # converting input token into `in_X`
        if from_ in self._sncreate._inputs:
            from_ = f"in_{self._sncreate.circuit.inputs_id[from_]}"

        # Adding new connection/arrow path to the collection
        if from_ not in self._arrows:
            self._arrows[from_] = {}
        self._arrows[from_][to] = [vis.Pos(*p) for p in path]

    def def_inputs(self, inputs_pos: list[tuple[float, float]]) -> None:
        self.__visual = None
        n_inputs = len(self._sncreate._inputs)
        if n_inputs != len(inputs_pos):
            raise Exception(f'Incorrect number of input positions. There are {n_inputs} '
                            "inputs for the circuit.")
        self._inputs = [vis.Pos(*inpos) for inpos in inputs_pos]

    def def_outputs(self, outputs_pos: list[tuple[float, float]]) -> None:
        self.__visual = None
        n_outputs = len(self._sncreate._outputs)
        if n_outputs != len(outputs_pos):
            raise Exception(f'Incorrect number of output positions. There are {n_outputs} '
                            "inputs for the circuit.")
        self._outputs = [vis.Pos(*outpos) for outpos in outputs_pos]

    def include_visual(self, name: str, visual: vis.CircuitDisplay) -> None:
        self.__visual = None
        if name not in self._sncreate._include_circuit_names:
            raise ValueError(f"No circuit named `{name}` has been included. "
                             "Names of included circuits: "
                             f"{self._sncreate._include_circuit_names}")
        if name in self._include_visuals:
            print(f"Overwriting included circuit visual {name}", file=sys.stderr)
        self._include_visuals[name] = visual
