"""
Utilities to save NN models into files that doryta can load.
"""
from __future__ import annotations

import struct
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from abc import ABC, abstractmethod
import numbers
from itertools import chain

from ..circuits.sncircuit import SNCircuit

from typing import BinaryIO, List, Union, Any, Tuple, NamedTuple, Optional, Dict


# ceildiv from stackoverflow: https://stackoverflow.com/a/17511341
def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


class NeuralConnection(ABC):
    """
    A neurnal connection indicates how a group of neurons connects to another (which
    neurons are connected to which (synapes) and their weights).

    `from_` is a tuple indicating the first and last neuron ids (integers) to start the
    connection.
    `to` is a tuple indicating the first and last neuron ids where the connections should
    arrive.
    """

    from_: Tuple[int, int]
    to: Tuple[int, int]

    @abstractmethod
    def set_from_and_to(
        self,
        from_: Tuple[int, int],
        to: Tuple[int, int]
    ) -> None:
        pass

    @property
    @abstractmethod
    def conn_type(self) -> int:
        ...


class All2AllConn(NeuralConnection):
    """
    Connects all "output" neurons to all "input" neurons. The synapse's weights are given
    one by one.
    """

    @property
    def conn_type(self) -> int:
        return 0x1

    def __init__(self, weights: NDArray[Any]) -> None:
        if len(weights.shape) != 2:
            raise Exception("The number of dimensions of weights must be 2, "
                            f"but {len(weights.shape)} was received.")
        self.weights = weights

    def set_from_and_to(
        self,
        from_: Tuple[int, int],
        to: Tuple[int, int]
    ) -> None:
        if self.weights.shape[0] != from_[1] - from_[0] + 1:
            raise Exception("The number of neurons needed at input "
                            f"{self.weights.shape[0]} does not "
                            f"coincide with given range {from_}")

        if self.weights.shape[1] != to[1] - to[0] + 1:
            raise Exception("The number of neurons needed at output "
                            f"{self.weights.shape[1]} does not coincide "
                            f"with given range {to}")

        self.from_ = from_
        self.to = to

    def __repr__(self) -> str:
        return f"All2AllConn(from_={self.from_}, to={self.to}, weights={self.weights})"


class Conv2DConn(NeuralConnection):
    """
    A convolutional 2D connection requires kernel, padding and striding parameters.
    """

    @property
    def conn_type(self) -> int:
        return 0x2

    def __init__(
        self,
        kernel: NDArray[Any],
        input_size: Tuple[int, int],
        padding: Tuple[int, int],
        striding: Tuple[int, int],
    ) -> None:
        if len(kernel.shape) != 2:
            raise Exception("A kernel must be only two dimensions in size!")

        self.kernel = kernel
        self.padding = padding
        self.input_size = input_size
        input_height = input_size[0] + 2 * padding[0]
        input_width = input_size[1] + 2 * padding[1]
        self.striding = striding
        to_height = ceildiv(input_height - kernel.shape[0] + 1, striding[0])
        to_width = ceildiv(input_width - kernel.shape[1] + 1, striding[1])
        self.output_size = to_height, to_width

    def set_from_and_to(
        self,
        from_: Tuple[int, int],
        to: Tuple[int, int]
    ) -> None:
        input_size = self.input_size
        to_height, to_width = self.output_size
        if input_size[0] * input_size[1] != from_[1] - from_[0] + 1:
            raise Exception("The number of neurons needed at input "
                            f"{input_size[0] * input_size[1]} does not "
                            f"coincide with given range {from_}")

        if to_height * to_width != to[1] - to[0] + 1:
            raise Exception("The number of neurons needed at output "
                            f"{to_height * to_width} does not coincide "
                            f"with given range {to}")

        self.from_ = from_
        self.to = to

    def __repr__(self) -> str:
        return f"Conv2DConn(from_={self.from_}, to={self.to}, " \
            f"kernel={self.kernel.tolist()}, " \
            f"padding={self.padding}, " \
            f"input_size={self.input_size}, " \
            f"striding={self.striding}, " \
            f"output_size={self.output_size})"


class LIFParams(NamedTuple):
    """
    These parameters are given for an array of `number` neurons. The group of neurons
    should be breakable into a number of `partitions` (all of the same size).
    """
    number: int
    partitions: int
    connections: List[NeuralConnection]
    thresholds: NDArray[Any]
    # The parameters below can be passed as a dictionary args
    tau:        Union[float, NDArray[Any]]
    resistance: Union[float, NDArray[Any]] = 1.0
    potential:  Union[float, NDArray[Any]] = 0.0
    current:    Union[float, NDArray[Any]] = 0.0
    resting_potential: Union[float, NDArray[Any]] = 0.0
    reset_potential:   Union[float, NDArray[Any]] = 0.0

    def are_parameters_consistent(self) -> bool:
        # partitions should be precise
        assert self.number % self.partitions == 0

        for param_name in ['tau', 'resistance', 'potential', 'current',
                           'resting_potential', 'reset_potential']:
            param = getattr(self, param_name)
            if isinstance(param, numbers.Number):
                continue
            else:
                assert isinstance(param, np.ndarray)
                if param.shape != (self.number,):
                    return False
        return True

    def get_value(self, param_name: str, pos: int) -> float:
        param = getattr(self, param_name)
        if isinstance(param, numbers.Number):
            return param  # type: ignore
        elif isinstance(param, np.ndarray):
            return param[pos]  # type: ignore
        raise Exception("Parameters can only be float's or ndarrays, but"
                        f" `{type(param)}` was given.")

    @property
    def partition_size(self) -> int:
        return self.number // self.partitions

    #  - Neuron params:
    #    + float potential = 0;          // V
    #    + float current = 0;            // I(t)
    #    + float resting_potential = 0;  // V_e
    #    + float reset_potential = 0;    // V_r
    #    + float threshold = 0.5 + bias; // V_th
    #    + float tau_m = dt = 1/256;     // C * R
    #    + float resistance = 1;         // R
    def save_to_file(self, f: BinaryIO, i: int) -> None:
        for param in ['potential', 'current', 'resting_potential', 'reset_potential',
                      'thresholds', 'tau', 'resistance']:
            f.write(struct.pack('>f', self.get_value(param, i)))


class LIFPassThruParams(NamedTuple):
    """
    These parameters are given for an array of `number` neurons. The group of neurons
    should be breakable into a number of `partitions` (all of the same size).

    The passthru paramater forces the neuron to pass-on its current potential state as a
    spike regardless if it reached the threshold or not. It resets after this.
    """
    number: int
    partitions: int
    connections: List[NeuralConnection]
    thresholds: NDArray[Any]
    # The parameters below can be passed as a dictionary args
    tau:        Union[float, NDArray[Any]]
    resistance: Union[float, NDArray[Any]] = 1.0
    potential:  Union[float, NDArray[Any]] = 0.0
    current:    Union[float, NDArray[Any]] = 0.0
    resting_potential: Union[float, NDArray[Any]] = 0.0
    reset_potential:   Union[float, NDArray[Any]] = 0.0
    passthru: bool = False

    def are_parameters_consistent(self) -> bool:
        # partitions should be precise
        assert self.number % self.partitions == 0
        assert isinstance(self.passthru, bool)

        for param_name in ['tau', 'resistance', 'potential', 'current',
                           'resting_potential', 'reset_potential']:
            param = getattr(self, param_name)
            if isinstance(param, numbers.Number):
                continue
            else:
                assert isinstance(param, np.ndarray)
                if param.shape != (self.number,):
                    return False
        return True

    def get_value(self, param_name: str, pos: int) -> float:
        param = getattr(self, param_name)
        if isinstance(param, numbers.Number):
            return param  # type: ignore
        elif isinstance(param, np.ndarray):
            return param[pos]  # type: ignore
        raise Exception("Parameters can only be float's or ndarrays, but"
                        f" `{type(param)}` was given.")

    @property
    def partition_size(self) -> int:
        return self.number // self.partitions

    #  - Neuron params:
    #    + float potential = 0;          // V
    #    + float current = 0;            // I(t)
    #    + float resting_potential = 0;  // V_e
    #    + float reset_potential = 0;    // V_r
    #    + float threshold = 0.5 + bias; // V_th
    #    + float tau_m = dt = 1/256;     // C * R
    #    + float resistance = 1;         // R
    def save_to_file(self, f: BinaryIO, i: int) -> None:
        for param in ['potential', 'current', 'resting_potential', 'reset_potential',
                      'thresholds', 'tau', 'resistance']:
            f.write(struct.pack('>f', self.get_value(param, i)))
        f.write(struct.pack('b', bool(self.passthru)))


NeuronParams = LIFParams | LIFPassThruParams


class ModelSaverLayers(object):
    def __init__(
        self,
        dt: float = 1/256,
        initial_threshold: float = .5,
        neuron_type: type[NeuronParams] = LIFParams
    ) -> None:
        self.neuron_type: type[NeuronParams] = neuron_type
        self.neuron_group: List[NeuronParams] = []
        self.dt = dt
        self.initial_threshold = initial_threshold

    @property
    def all_connections(self) -> List[NeuralConnection]:
        return list(chain(*(group.connections for group in self.neuron_group)))

    def _check_at_least_one_group(self, input_size: int, msg: str) -> None:
        if len(self.neuron_group) > 0 and self.neuron_group[-1].number != input_size:
            raise Exception("The number of input neurons (last layer) "
                            f"{self.neuron_group[-1].number} does not coincide with "
                            "the number of neurons described by the layer "
                            f"({msg}) {input_size}")

        # The actual code starts from here
        if len(self.neuron_group) == 0:
            threshold0 = self.initial_threshold * np.ones((input_size,))
            self.add_neuron_group(threshold0)

    @property
    def total_neurons(self) -> int:
        return sum(g.number for g in self.neuron_group)

    def add_neuron_group(
        self,
        thresholds: NDArray[Any],
        partitions: int = 1,
        args: Optional[Dict[str, Any]] = None
    ) -> None:
        if len(thresholds.shape) != 1:
            raise Exception("The shape of thresholds must be (n,) where n is the number of "
                            "neurons to add. The shape of the given thresholds was "
                            f"{thresholds.shape}")

        if args is None:
            args = {}
        else:
            args = args.copy()
        if 'connections' not in args:
            args['connections'] = []
        if 'tau' not in args:
            args['tau'] = self.dt

        num_neurons = thresholds.shape[0]
        nparams = self.neuron_type(num_neurons, partitions, thresholds=thresholds, **args)
        if nparams.are_parameters_consistent():
            self.neuron_group.append(nparams)
        else:
            raise Exception("Some given parameter given as ndarray does not correspond "
                            "with the number of neurons defined.")

    def add_all2all_conn(
        self,
        from_: int | tuple[int, int],
        to: int | tuple[int, int],
        weights: NDArray[Any]
    ) -> None:
        from_start, from_end, num_neurons_in_from, input_neuron_group = \
            self.__find_start_end_for_neuron_group(from_)
        to_start, to_end, num_neurons_in_to, _ = \
            self.__find_start_end_for_neuron_group(to)

        if weights.shape != (num_neurons_in_from, num_neurons_in_to):
            raise Exception(f"The shape of the weights {weights.shape} does not coincide with "
                            f"the number of output neurons ({num_neurons_in_from}) "
                            f"and the number of input neurons ({num_neurons_in_to})")

        all2all_conn = All2AllConn(weights)
        all2all_conn.set_from_and_to((from_start, from_end), (to_start, to_end))

        from_ = from_[0] if isinstance(from_, tuple) else from_
        self.neuron_group[from_].connections.append(all2all_conn)

    def add_one2all_conn(
        self,
        from_layer: Optional[int],
        from_neuron: int,
        to: Tuple[int, int],
        weight: float
    ) -> None:
        """
        Connects a single neuron to a whole layer. If `from_layer` is None, `from_neuron`
        indicates the global id of the input neuron. Otherwise, `from_layer` indicates the
        layer in which the neuron falls into and `from_neuron` indicates the position
        within the layer.

        `to` indicates the layer and partition the connection should get to
        """

        num_neuron_groups = len(self.neuron_group)
        if not (0 <= to[0] < num_neuron_groups):
            raise Exception(f"`to[0]` must be inside the range [0, {num_neuron_groups-1}], "
                            f"but `to[0] = {to[0]}`")
        if from_layer is None:
            total_neurons = sum(group_i.number for group_i in self.neuron_group)
            if not (0 <= from_neuron < total_neurons):
                raise Exception("`from_neuron` must be inside the range "
                                f"[0, {total_neurons-1}], "
                                f"but `from_neuron = {from_neuron}`")
        else:
            if not (0 <= from_layer < num_neuron_groups):
                raise Exception("`from_layer` must be inside the range "
                                f"[0, {num_neuron_groups-1}], "
                                f"but `from_layer = {from_layer}`")
            neurons_layer = self.neuron_group[from_layer].number
            if not (0 <= from_neuron < neurons_layer):
                raise Exception("`from_neuron` must be inside the range "
                                f"[0, {neurons_layer-1}], "
                                f"but `from_neuron = {from_neuron}`")

        num_neurons_in_to = self.neuron_group[to[0]].partition_size
        weights = np.ones((1, num_neurons_in_to)) * weight

        if from_layer is None:
            # finding layer in which neuron falls
            tot = 0
            for i, group_i in enumerate(self.neuron_group):
                tot += group_i.number
                if tot > from_neuron:
                    break

            from_layer = i
            from_start = from_neuron
            from_end = from_neuron
        else:
            from_start = sum(self.neuron_group[i].number for i in range(from_layer)) \
                + from_neuron
            from_end = from_start

        all2all_conn = All2AllConn(weights)
        to_start = sum(self.neuron_group[i].number for i in range(to[0])) \
            + num_neurons_in_to * to[1]
        to_end = to_start + num_neurons_in_to - 1
        all2all_conn.set_from_and_to((from_start, from_end), (to_start, to_end))

        self.neuron_group[from_layer].connections.append(all2all_conn)

    def __find_start_end_for_neuron_group(
        self, neuron_group_id: int | tuple[int, int]
    ) -> tuple[int, int, int, NeuronParams]:
        """
        Given a neuron group id (either a single `int` (neuron group number) or tuple of
        `int`s (neuron group number and partition number)), this function returns:
        - the neuron id in which the given neuron group starts,
        - the last neuron id of the neuron group, and
        - number of neurons in group/partition (last id - first id + 1),
        - the input neuron group.
        """
        # finding neuron id ranges for input layer
        if isinstance(neuron_group_id, int):
            assert 0 <= neuron_group_id < len(self.neuron_group)
            neuron_group = self.neuron_group[neuron_group_id]
            num_neurons = neuron_group.number
            neuron_group_id = (neuron_group_id, 0)
        else:
            assert 0 <= neuron_group_id[0] < len(self.neuron_group)
            neuron_group = self.neuron_group[neuron_group_id[0]]
            num_neurons = neuron_group.partition_size

        from_start = sum(self.neuron_group[i].number for i in range(neuron_group_id[0])) \
            + num_neurons * neuron_group_id[1]
        from_end = from_start + num_neurons - 1
        return from_start, from_end, num_neurons, neuron_group

    def add_one2one_conn(
        self,
        from_: Union[int, Tuple[int, int]],
        to: Union[int, Tuple[int, int]],
        weight: float
    ) -> None:
        """
        When `from` (or `to`) is a tuple, the first value indicates the number of the
        layer and the second indicates the partition number within the layer.
        """
        from_start, from_end, num_neurons, input_neuron_group = \
            self.__find_start_end_for_neuron_group(from_)
        to_start, to_end, num_neurons_in_to, _ = \
            self.__find_start_end_for_neuron_group(to)

        if num_neurons != num_neurons_in_to:
            raise Exception(
                f"The input size to one2one connection `{num_neurons}` "
                f"is not the same as the output `{num_neurons_in_to}`"
            )

        # Conv2DConn params
        input_size = (1, num_neurons)
        padding = (0, 0)
        striding = (1, 1)

        kernel = np.float_(weight).reshape((1, 1))
        conv = Conv2DConn(kernel, input_size, padding, striding)
        conv.set_from_and_to((from_start, from_end), (to_start, to_end))
        input_neuron_group.connections.append(conv)

    def add_conv2d_conn(
        self,
        kernel: NDArray[Any],
        input_size: tuple[int, int],
        from_: int | tuple[int, int],
        to: int | tuple[int, int],
        padding: tuple[int, int] = (0, 0),
        striding: tuple[int, int] = (1, 1)
    ) -> None:
        # check that kernel is 2 dimensions
        assert len(kernel.shape) == 2

        # check from and to
        from_start, from_end, num_neurons, input_neuron_group = \
            self.__find_start_end_for_neuron_group(from_)
        to_start, to_end, num_neurons_in_to, _ = \
            self.__find_start_end_for_neuron_group(to)

        # create connection
        conv = Conv2DConn(kernel, input_size, padding, striding)
        conv.set_from_and_to((from_start, from_end), (to_start, to_end))
        # add connection to respective neuron group
        input_neuron_group.connections.append(conv)

    def add_fully_layer(
        self,
        weights: NDArray[Any],
        thresholds: NDArray[Any],
        neuron_args: Optional[Dict[str, Any]] = None
    ) -> None:
        if not (len(weights.shape) == 2
                and len(thresholds.shape) == 1
                and weights.shape[1] == thresholds.shape[0]):
            raise Exception("The shape of weights must be (input, output)"
                            " and for thresholds (output,). "
                            f"Their shapes are {weights.shape} and {thresholds.shape}"
                            " for weights and thresholds, respectively.")

        self._check_at_least_one_group(
            weights.shape[0], "equal to the first dimension of weights")

        if np.any(thresholds <= 0):
            print("Warning: given threshold is smaller than 0 which means that neuron "
                  "always fires when it receives any spike. This value will be changed "
                  "to account for behaviour.")
            neg_indices = thresholds <= 0
            weights[:, neg_indices] = 1
            thresholds[neg_indices] = 0.5

        # The actual code starts here
        self.add_neuron_group(thresholds, args=neuron_args)
        to = len(self.neuron_group) - 1  # last layer
        self.add_all2all_conn(from_=to - 1, to=to, weights=weights)

    def add_conv2d_layer(
        self,
        kernel: NDArray[Any],
        threshold: NDArray[Any],
        input_size: Tuple[int, int],
        padding: Tuple[int, int] = (0, 0),
        striding: Tuple[int, int] = (1, 1),
        neuron_args: Optional[Dict[str, Any]] = None
    ) -> None:
        if len(kernel.shape) != 4:
            raise Exception("A kernel must be of four dimensions "
                            "(height, width, channels, filters). "
                            f"The shape of kernel is `{kernel.shape}`")
        channels = kernel.shape[2]
        filters = kernel.shape[3]

        input_neurons = channels * input_size[0] * input_size[1]
        if len(threshold.shape) != 1:
            raise Exception("Threshold is a 1 dimensional array. "
                            f"{threshold.shape} was given.")
        if threshold.size != filters:
            raise Exception("The size of the threshold must coincide with the number of "
                            f"filters. {filters} filters and {threshold.size} threshold "
                            "size were given")

        self._check_at_least_one_group(
            input_neurons, "equal to: input_height * input_width * channnels")

        # This convolution connection is only used to compute the output size
        conv = Conv2DConn(kernel[..., 0, 0], input_size, padding, striding)
        out_height, out_width = conv.output_size
        output_neurons = filters * out_height * out_width

        connections = self.neuron_group[-1].connections
        neurons_to_date = self.total_neurons - self.neuron_group[-1].number
        n_input = input_size[0] * input_size[1]
        n_output = out_height * out_width
        new_connections = []

        for chan in range(channels):
            for fil in range(filters):
                conv = Conv2DConn(kernel[..., chan, fil], input_size, padding, striding)
                from_start = neurons_to_date + n_input * chan
                from_end = from_start + n_input - 1
                to_start = neurons_to_date + input_neurons + n_output * fil
                to_end = to_start + n_output - 1
                conv.set_from_and_to((from_start, from_end), (to_start, to_end))
                new_connections.append(conv)

        connections.extend(new_connections)

        thresholds = np.repeat(threshold, out_height * out_width)
        assert output_neurons == thresholds.size
        self.add_neuron_group(thresholds, partitions=filters, args=neuron_args)

    def add_maxpool_layer(
        self,
        input_size: Tuple[int, int, int],
        striding: Tuple[int, int]
    ) -> None:
        input_neurons = input_size[0] * input_size[1] * input_size[2]
        self._check_at_least_one_group(
            input_neurons, "equal to: input_height * input_width * channnels")

        channels = input_size[2]

        kernel = np.ones(striding)
        padding = (0, 0)
        input_size_channel = (input_size[0], input_size[1])

        conv = Conv2DConn(kernel, input_size_channel, padding, striding)
        out_height, out_width = conv.output_size
        output_neurons = channels * out_height * out_width

        connections = self.neuron_group[-1].connections
        neurons_to_date = sum([g.number for g in self.neuron_group]) \
            - self.neuron_group[-1].number
        n_input = input_size[0] * input_size[1]
        n_output = out_height * out_width
        new_connections = []

        for chan in range(channels):
            conv = Conv2DConn(kernel, input_size_channel, padding, striding)
            from_start = neurons_to_date + n_input * chan
            from_end = from_start + n_input - 1
            to_start = neurons_to_date + input_neurons + n_output * chan
            to_end = to_start + n_output - 1
            conv.set_from_and_to((from_start, from_end), (to_start, to_end))
            new_connections.append(conv)

        connections.extend(new_connections)

        thresholds: NDArray[Any] = 0.5 * np.ones((output_neurons,))
        self.add_neuron_group(thresholds, channels)

    # TODO: admit in list of input layers to determine a unique neuron to connect in which
    # case the connection type should be 1-to-many!
    def add_sncircuit_layer(
        self,
        snc: SNCircuit,
        input_layers: Optional[List[Union[int, Tuple[int, int]]]] = None,
    ) -> None:
        """
        input_layers:  indicates which neurons groups (and partitions) to take as input.
                       `None` means the same as [-1]; [3, 6] indicate that neuron groups 3
                       and 6 will be inputs 0 and 1 to the circuit; [(2, 3), (4, 2)]
                       indicate that neuron group 2 partition 3 will be used as input 0 to
                       the circuit, and group 4 partition 2 will be used as input 1
        """
        assert len(self.neuron_group) > 0, "A sn-circuit cannot be the first layer"

        if input_layers is None:
            input_layers = [(len(self.neuron_group) - 1, 0)]

        # assert number of inputs to circuit must be the same as input layers
        if len(input_layers) != len(snc.inputs):
            raise Exception("The number of input layers must coincide with the number of "
                            "inputs to the circuit")

        # Finding number of times the circuit has to be repeated
        size_input_layers = [
            self.neuron_group[inp].number if isinstance(inp, int)
            else self.neuron_group[inp[0]].partition_size
            for inp in input_layers]
        num_circuits = max(size_input_layers)

        if not all(inp == num_circuits or inp == 1 for inp in size_input_layers):
            raise Exception("All input layers must be the same size or a single neuron. "
                            f"Layer sizes are {size_input_layers}")

        # defining a new layer with number of partitions equal to the number of neurons in snc
        neuron_args = snc.get_params_in_bulk()
        thresholds = np.repeat(neuron_args['threshold'], num_circuits)
        capacitance = np.repeat(neuron_args['capacitance'], num_circuits)
        resistance = np.repeat(neuron_args['resistance'], num_circuits)
        neuron_args = {
            'tau': capacitance * resistance,
            'resistance': resistance,
            'potential': np.repeat(neuron_args['potential'], num_circuits),
            'current': np.repeat(neuron_args['current'], num_circuits),
            'resting_potential': np.repeat(
                neuron_args['resting_potential'], num_circuits),
            'reset_potential': np.repeat(neuron_args['reset_potential'], num_circuits),
        }
        self.add_neuron_group(thresholds, partitions=snc.num_neurons, args=neuron_args)
        last_layer = len(self.neuron_group) - 1

        # Creating connections from input layers to the circuit as determined by the
        # circuit `inputs` itself
        for input_layer, input_params in zip(input_layers, snc.inputs):
            # for each synapse in input
            for snc_neuron_id, synap_params in input_params.items():
                assert synap_params.delay == 1, "The binary format only allows for delay = 1"
                # create a connection
                if self.__is_layer_one_neuron(input_layer):
                    if isinstance(input_layer, tuple):
                        input_layer_, neuron_pos = input_layer
                    else:
                        input_layer_ = input_layer
                        neuron_pos = 0
                    self.add_one2all_conn(
                        input_layer_, neuron_pos, (last_layer, snc_neuron_id), synap_params.weight)
                else:
                    self.add_one2one_conn(
                        input_layer, (last_layer, snc_neuron_id), synap_params.weight)

        # Defining connections for each synapse between the SN-circuit neurons
        for neuron_id, neuron_params in enumerate(snc.neurons):
            for to_neuron_id, synap_params in neuron_params.synapses.items():
                assert synap_params.delay == 1, "The binary format only allows for delay = 1"
                self.add_one2one_conn(
                    (last_layer, neuron_id), (last_layer, to_neuron_id), synap_params.weight)

    def __is_layer_one_neuron(self, group_id: Union[int, Tuple[int, int]]) -> bool:
        """Checks whether the given `neuron_group` + partition contains a single neuron."""
        if isinstance(group_id, int):
            return self.neuron_group[group_id].number == 1
        else:  # isinstance(group_id, tuple)
            return self.neuron_group[group_id[0]].partition_size == 1

    def save(self, path: Union[str, Path], version: Optional[int] = None) -> None:
        if not self.neuron_group:
            raise Exception("Nothing to do. No layers defined")

        all_connections = self.all_connections

        if version is None:
            if all(isinstance(conn, All2AllConn) for conn in all_connections) \
                    and self.neuron_type == LIFParams:
                version = 1
            else:
                version = 2

        if version == 1:
            if not all(isinstance(conn, All2AllConn) for conn in all_connections):
                raise Exception("Version 1 can only save All2AllConn on layers")
            if self.neuron_type != LIFParams:
                raise Exception("Version 1 can only save networks with the vanilla LIF neuron type")
            with open(path, 'wb') as fh:
                self.save_v1(fh)
        elif version == 2:
            with open(path, 'wb') as fh:
                self.save_v2plus(fh)
        else:
            raise Exception(f"There is no such thing as 'version = {version}'")

    def save_v1(self, fh: BinaryIO) -> None:
        all_connections = self.all_connections

        # -- Magic number
        fh.write(struct.pack('>I', 0x23432BC4))
        # -- File format
        fh.write(struct.pack('>H', 0x1))
        # -- Total number of neurons (N)
        fh.write(struct.pack('>i', self.total_neurons))
        # -- Total number of groups
        total_groups_broken = sum([g.partitions for g in self.neuron_group])
        fh.write(struct.pack('B', total_groups_broken))
        # -- Total number of connections
        fh.write(struct.pack('B', len(all_connections)))
        # -- dt
        fh.write(struct.pack('>f', self.dt))
        # -- Neuron groups ("layers")
        for group in self.neuron_group:
            fh.write(struct.pack('>i', group.number))
        # -- Synapses groups
        acc = 0
        for i in range(len(self.neuron_group) - 1):
            # Defining i-th all2all connection
            to_start = acc + self.neuron_group[i].number
            to_end = to_start + self.neuron_group[i+1].number - 1
            from_start = acc
            from_end = to_start - 1

            # Checking old method works with new
            assert all_connections[i].from_[0] == from_start
            assert all_connections[i].from_[1] == from_end
            assert all_connections[i].to[0] == to_start
            assert all_connections[i].to[1] == to_end

            fh.write(struct.pack('>i', from_start))
            fh.write(struct.pack('>i', from_end))
            fh.write(struct.pack('>i', to_start))
            fh.write(struct.pack('>i', to_end))
            acc += self.neuron_group[i].number

        # -- Actual neuron parameters (model itself)
        #  N x neurons:

        for i, group in enumerate(self.neuron_group):
            thresholds = group.thresholds
            assert group.number == thresholds.shape[0]

            last = i + 1 == len(self.neuron_group)
            # Saving for each neuron neuron
            #  - Neuron params
            #  - Number of synapses (M)
            #  - M x synapses
            for j in range(group.number):
                # Neuron params
                group.save_to_file(fh, j)

                if last:
                    # number of synapses per neuron
                    fh.write(struct.pack('>i', 0))
                else:
                    assert len(group.connections) == 1
                    conn = group.connections[0]
                    assert isinstance(conn, All2AllConn)
                    assert self.neuron_group[i+1].number == conn.weights.shape[1]

                    # number of synapses per neuron
                    fh.write(struct.pack('>i', self.neuron_group[i+1].number))
                    # synapses
                    conn.weights[j].astype('>f4').tofile(fh)

    def save_v2plus(self, fh: BinaryIO) -> None:  # noqa: C901
        all_connections = self.all_connections

        # -- Magic number
        fh.write(struct.pack('>I', 0x23432BC4))
        # -- File format
        if self.neuron_type == LIFParams:
            fh.write(struct.pack('>H', 0x2))
        if self.neuron_type == LIFPassThruParams:
            fh.write(struct.pack('>H', 0x32))
        else:
            raise Exception(f"Unknown neuron type {type(self.neuron_type)}")
        # -- Total number of neurons (N)
        fh.write(struct.pack('>i', self.total_neurons))
        # -- Total number of groups
        total_groups_broken = sum([g.partitions for g in self.neuron_group])
        fh.write(struct.pack('>H', total_groups_broken))
        # -- Total number of connections
        fh.write(struct.pack('>H', len(all_connections)))
        # -- dt
        fh.write(struct.pack('>f', self.dt))
        # -- Neuron groups ("layers")
        for group in self.neuron_group:
            assert group.number % group.partitions == 0
            for i in range(group.partitions):
                fh.write(struct.pack('>i', group.number//group.partitions))

        # -- Synapses groups
        for conn in all_connections:
            # saving type of connection (0x1 all2all, 0x2 conv2d)
            fh.write(struct.pack('B', conn.conn_type))

            from_start, from_end = conn.from_
            to_start, to_end = conn.to
            fh.write(struct.pack('>i', from_start))
            fh.write(struct.pack('>i', from_end))
            fh.write(struct.pack('>i', to_start))
            fh.write(struct.pack('>i', to_end))

            # Kernel is saved as part of the convolution parameters
            if isinstance(conn, Conv2DConn):
                fh.write(struct.pack('>i', conn.input_size[0]))
                fh.write(struct.pack('>i', conn.input_size[1]))
                fh.write(struct.pack('>i', conn.output_size[0]))
                fh.write(struct.pack('>i', conn.output_size[1]))
                fh.write(struct.pack('>i', conn.padding[0]))
                fh.write(struct.pack('>i', conn.padding[1]))
                fh.write(struct.pack('>i', conn.striding[0]))
                fh.write(struct.pack('>i', conn.striding[1]))
                fh.write(struct.pack('>i', conn.kernel.shape[0]))
                fh.write(struct.pack('>i', conn.kernel.shape[1]))
                conn.kernel.astype('>f4').tofile(fh)

        neuron_id = -1
        # -- Actual neuron parameters
        #   (Only synapses for Fully Connected layers are stored!)
        #  N x neuron:
        for i, group in enumerate(self.neuron_group):
            assert group.number == group.thresholds.shape[0]

            # For each neuron, saving
            #  - Neuron params
            #  - Number of fully connected connections to it
            #  Per each fully connection
            #    - Range of neurons connected (to_start, to_end)
            #    - Synapses for range
            for j in range(group.number):
                neuron_id += 1
                # Neuron params
                group.save_to_file(fh, j)

                total_fully_conn_for_neuron = \
                    sum(1 for conn in group.connections
                        if isinstance(conn, All2AllConn)
                        and conn.from_[0] <= neuron_id <= conn.from_[1])
                # number of fully connections that neuron takes part of
                fh.write(struct.pack('>H', total_fully_conn_for_neuron))

                for conn in group.connections:
                    # synapses
                    if isinstance(conn, All2AllConn) \
                            and conn.from_[0] <= neuron_id <= conn.from_[1]:
                        fh.write(struct.pack('>i', conn.to[0]))
                        fh.write(struct.pack('>i', conn.to[1]))
                        conn.weights[neuron_id - conn.from_[0]].astype('>f4').tofile(fh)


if __name__ == '__main__':
    dump_folder = Path('various/snn-models/')

    w = np.array([[1,  0],
                  [2, -1],
                  [1,  1]])
    b = np.array([.5, 1])

    w1 = np.array([[-1, 1, 0, -2, 1],
                   [4, -5, 0, 3, 0]])
    b1 = np.array([0, 0, 1, 0, 4])

    msaver = ModelSaverLayers()
    msaver.add_fully_layer(w, b + 0.5)
    msaver.add_fully_layer(w1, b1 + 0.5)
    msaver.save(dump_folder / "3-to-2-to-5-neurons.doryta.bin")

    k1 = np.zeros((3, 3, 1, 2))
    k1[..., 0, 0] = [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]
    k1[..., 0, 1] = [[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]]
    t1 = np.array([2.5, 3.5])

    k2 = np.zeros((1, 1, 2, 1))
    k2[..., 0, 0] = [[1]]
    k2[..., 1, 0] = [[-1]]
    t2 = np.array([.5])

    msaver = ModelSaverLayers()
    msaver.add_conv2d_layer(k1, t1, (20, 20), (1, 1), (1, 1))
    msaver.add_conv2d_layer(k2, t2, (20, 20), (0, 0), (1, 1))
    msaver.save(dump_folder / "another-gol.doryta.bin")
