"""
Utilities to save NN models into files that doryta can load.
"""
from __future__ import annotations

import struct
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from abc import ABC, abstractmethod

from typing import BinaryIO, List, Union, Any, Tuple, NamedTuple


# ceildiv from stackoverflow: https://stackoverflow.com/a/17511341
def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


class NeuralConnection(ABC):
    from_: Tuple[int, int]
    to: Tuple[int, int]

    @abstractmethod
    def add_from_to_ranges(
        self,
        from_: Tuple[int, int],
        to: Tuple[int, int]
    ) -> None:
        pass

    @property
    @abstractmethod
    def type(self) -> int:
        ...


class FullyConn(NeuralConnection):
    @property
    def type(self) -> int:
        return 0x1

    def __init__(self, weights: NDArray[Any]) -> None:
        if len(weights.shape) != 2:
            raise Exception("The number of dimensions of weights must be 2, "
                            f"but {len(weights.shape)} was received.")
        self.weights = weights

    def add_from_to_ranges(
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


class Conv2DConn(NeuralConnection):
    @property
    def type(self) -> int:
        return 0x2

    def __init__(
        self,
        kernel: NDArray[Any],
        input_size: Tuple[int, int],
        padding: Tuple[int, int],
        striding: Tuple[int, int],
    ) -> None:
        if len(kernel.shape) != 2:
            raise Exception("A kernel only has two dimensions!")

        self.kernel = kernel
        self.padding = padding
        self.input_size = input_size
        input_height = input_size[0] + 2 * padding[0]
        input_width = input_size[1] + 2 * padding[1]
        self.striding = striding
        to_height = ceildiv(input_height - kernel.shape[0] + 1, striding[0])
        to_width = ceildiv(input_width - kernel.shape[1] + 1, striding[1])
        self.output_size = to_height, to_width

    def add_from_to_ranges(
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


class NeuronParams(NamedTuple):
    number: int
    partitions: int
    thresholds: NDArray[Any]
    connections: List[NeuralConnection]


class ModelSaverLayers(object):
    def __init__(self, dt: float = 1/256, initial_threshold: float = .5) -> None:
        self.neuron_group: List[NeuronParams] = []
        self.all_connections: List[NeuralConnection] = []
        self.dt = dt
        self.initial_threshold = initial_threshold

    def add_fully_layer(self, weights: NDArray[Any], thresholds: NDArray[Any]) -> None:
        if not (len(weights.shape) == 2
                and len(thresholds.shape) == 1
                and weights.shape[1] == thresholds.shape[0]):
            raise Exception("The shape of weights must be (input, output)"
                            " and for thresholds (output,). "
                            f"Their shapes are {weights.shape} and {thresholds.shape}"
                            " for weights and thresholds, respectively.")

        fully_conn = FullyConn(weights)

        if len(self.neuron_group) == 0:
            threshold0: NDArray[Any] = self.initial_threshold * np.ones((weights.shape[0],))
            self.neuron_group = [
                NeuronParams(weights.shape[0], 1, threshold0, [])
            ]

        if self.neuron_group[-1].number != weights.shape[0]:
            raise Exception(
                f"The shape of the input weights {self.neuron_group[-1].number} doesn't "
                "match with the previous layer output weights "
                f"{weights.shape[0]}")

        self.neuron_group[-1].connections.append(fully_conn)
        self.neuron_group.append(
            NeuronParams(weights.shape[1], 1, thresholds, []))

        total_neurons = sum([g.number for g in self.neuron_group])
        to_start = total_neurons - self.neuron_group[-1].number
        to_end = total_neurons - 1
        from_start = to_start - self.neuron_group[-2].number
        from_end = to_start - 1
        fully_conn.add_from_to_ranges((from_start, from_end), (to_start, to_end))

        self.all_connections.append(fully_conn)

    def add_conv2d_layer(
        self,
        kernel: NDArray[Any],
        threshold: NDArray[Any],
        input_size: Tuple[int, int],
        padding: Tuple[int, int] = (0, 0),
        striding: Tuple[int, int] = (1, 1)
    ) -> None:
        if len(kernel.shape) != 4:
            raise Exception("A kernel must be of four dimensions "
                            "(height, width, channels, filters). "
                            f"The shape of kernel is `{kernel.shape}`")
        channels = kernel.shape[2]
        filters = kernel.shape[3]

        input_neurons = channels * input_size[0] * input_size[1]
        if self.neuron_group and input_neurons != self.neuron_group[-1].number:
            raise Exception("The number of input neurons (last layer) "
                            f"{self.neuron_group[-1].number} does not coincide with "
                            "the needed number for the given kernel "
                            "{input_neurons} (input_height * input_width * channnels)")
        if len(threshold.shape) != 1:
            raise Exception("Threshold is a 1 dimensional array. "
                            f"{threshold.shape} was given.")
        if threshold.size != filters:
            raise Exception("The size of the threshold must coincide with the number of "
                            f"filters. {filters} filters and {threshold.size} threshold "
                            "size were given")

        # This convolution connection is only used to compute the output size
        conv = Conv2DConn(kernel[..., 0, 0], input_size, padding, striding)
        out_height, out_width = conv.output_size
        output_neurons = filters * out_height * out_width

        if len(self.neuron_group) == 0:
            threshold0: NDArray[Any] = self.initial_threshold * np.ones((input_neurons))
            self.neuron_group = [
                NeuronParams(input_neurons, channels, threshold0, [])
            ]

        connections = self.neuron_group[-1].connections
        neurons_to_date = sum([g.number for g in self.neuron_group]) \
            - self.neuron_group[-1].number
        n_input = input_size[0] * input_size[1]
        n_output = out_height * out_width

        for chan in range(channels):
            for fil in range(filters):
                conv = Conv2DConn(kernel[..., chan, fil], input_size, padding, striding)
                from_start = neurons_to_date + n_input * chan
                from_end = from_start + n_input - 1
                to_start = neurons_to_date + input_neurons + n_output * fil
                to_end = to_start + n_output - 1
                conv.add_from_to_ranges((from_start, from_end), (to_start, to_end))
                connections.append(conv)

        self.all_connections.extend(connections)

        thresholds = np.repeat(threshold, out_height * out_width)
        assert output_neurons == thresholds.size
        self.neuron_group.append(
            NeuronParams(output_neurons, channels, thresholds, []))

    def add_maxpool_layer(
        self,
        input_size: Tuple[int, int, int],
        striding: Tuple[int, int]
    ) -> None:
        input_neurons = input_size[0] * input_size[1] * input_size[2]
        if self.neuron_group and input_neurons != self.neuron_group[-1].number:
            raise Exception("The number of input neurons (last layer) "
                            f"{self.neuron_group[-1].number} does not coincide with "
                            "the needed number for the given kernel "
                            "{input_neurons} (input_height * input_width * channnels)")

        channels = input_size[2]

        kernel = np.ones(striding)
        padding = (0, 0)
        input_size_channel = (input_size[0], input_size[1])

        conv = Conv2DConn(kernel, input_size_channel, padding, striding)
        out_height, out_width = conv.output_size
        output_neurons = channels * out_height * out_width

        if len(self.neuron_group) == 0:
            threshold0: NDArray[Any] = self.initial_threshold * np.ones((input_neurons))
            self.neuron_group = [
                NeuronParams(input_neurons, channels, threshold0, [])
            ]

        connections = self.neuron_group[-1].connections
        neurons_to_date = sum([g.number for g in self.neuron_group]) \
            - self.neuron_group[-1].number
        n_input = input_size[0] * input_size[1]
        n_output = out_height * out_width

        for chan in range(channels):
            conv = Conv2DConn(kernel, input_size_channel, padding, striding)
            from_start = neurons_to_date + n_input * chan
            from_end = from_start + n_input - 1
            to_start = neurons_to_date + input_neurons + n_output * chan
            to_end = to_start + n_output - 1
            conv.add_from_to_ranges((from_start, from_end), (to_start, to_end))
            connections.append(conv)

        self.all_connections.extend(connections)

        thresholds: NDArray[Any] = 0.5 * np.ones((output_neurons,))
        self.neuron_group.append(
            NeuronParams(output_neurons, channels, thresholds, []))

    #  - Neuron params:
    #    + float potential = 0;          // V
    #    + float current = 0;            // I(t)
    #    + float resting_potential = 0;  // V_e
    #    + float reset_potential = 0;    // V_r
    #    + float threshold = 0.5 - bias; // V_th
    #    + float tau_m = dt = 1/256;     // C * R
    #    + float resistance = 1;         // R
    def _save_neuron_params(self, f: BinaryIO, threshold: float) -> None:
        f.write(struct.pack('>f', 0))     # potential
        f.write(struct.pack('>f', 0))     # current
        f.write(struct.pack('>f', 0))     # resting_potential
        f.write(struct.pack('>f', 0))     # reset_potential
        f.write(struct.pack('>f', threshold))  # threshold
        f.write(struct.pack('>f', self.dt))   # tau_m
        f.write(struct.pack('>f', 1))    # resistance

    def save(self, path: Union[str, Path]) -> None:
        if not self.neuron_group:
            raise Exception("Nothing to do. No layers defined")

        if all(isinstance(conn, FullyConn) for conn in self.all_connections):
            with open(path, 'wb') as fh:
                self.save_v1(fh)
        else:
            with open(path, 'wb') as fh:
                self.save_v2(fh)

    def save_v1(self, fh: BinaryIO) -> None:
        # -- Magic number
        fh.write(struct.pack('>I', 0x23432BC4))
        # -- File format
        fh.write(struct.pack('>H', 0x1))
        # -- Total number of neurons (N)
        total_neurons = sum([g.number for g in self.neuron_group])
        fh.write(struct.pack('>i', total_neurons))
        # -- Total number of groups
        total_groups_broken = sum([g.partitions for g in self.neuron_group])
        fh.write(struct.pack('B', total_groups_broken))
        # -- Total number of connections
        fh.write(struct.pack('B', len(self.all_connections)))
        # -- dt
        fh.write(struct.pack('>f', self.dt))
        # -- Neuron groups ("layers")
        for group in self.neuron_group:
            fh.write(struct.pack('>i', group.number))
        # -- Synapses groups
        acc = 0
        for i in range(len(self.neuron_group) - 1):
            # Defining i-th fully connection
            to_start = acc + self.neuron_group[i].number
            to_end = to_start + self.neuron_group[i+1].number - 1
            from_start = acc
            from_end = to_start - 1

            # Checking old method works with new
            assert self.all_connections[i].from_[0] == from_start
            assert self.all_connections[i].from_[1] == from_end
            assert self.all_connections[i].to[0] == to_start
            assert self.all_connections[i].to[1] == to_end

            fh.write(struct.pack('>i', from_start))
            fh.write(struct.pack('>i', from_end))
            fh.write(struct.pack('>i', to_start))
            fh.write(struct.pack('>i', to_end))
            acc += self.neuron_group[i].number

        # -- Actual neuron parameters (model itself)
        #  N x neurons:

        for i, group in enumerate(self.neuron_group):
            thresholds = group.thresholds
            assert(group.number == thresholds.shape[0])

            last = i + 1 == len(self.neuron_group)
            # Saving for each neuron neuron
            #  - Neuron params
            #  - Number of synapses (M)
            #  - M x synapses
            for j in range(group.number):
                # Neuron params
                self._save_neuron_params(fh, thresholds[j])

                if last:
                    # number of synapses per neuron
                    fh.write(struct.pack('>i', 0))
                else:
                    assert len(group.connections) == 1
                    conn = group.connections[0]
                    assert isinstance(conn, FullyConn)
                    assert self.neuron_group[i+1].number == conn.weights.shape[1]

                    # number of synapses per neuron
                    fh.write(struct.pack('>i', self.neuron_group[i+1].number))
                    # synapses
                    conn.weights[j].astype('>f4').tofile(fh)

    def save_v2(self, fh: BinaryIO) -> None:
        # -- Magic number
        fh.write(struct.pack('>I', 0x23432BC4))
        # -- File format
        fh.write(struct.pack('>H', 0x2))
        # -- Total number of neurons (N)
        total_neurons = sum([g.number for g in self.neuron_group])
        fh.write(struct.pack('>i', total_neurons))
        # -- Total number of groups
        total_groups_broken = sum([g.partitions for g in self.neuron_group])
        fh.write(struct.pack('H', total_groups_broken))
        # -- Total number of connections
        fh.write(struct.pack('H', len(self.all_connections)))
        # -- dt
        fh.write(struct.pack('>f', self.dt))
        # -- Neuron groups ("layers")
        for group in self.neuron_group:
            assert group.number % group.partitions == 0
            for i in range(group.partitions):
                fh.write(struct.pack('>i', group.number//group.partitions))

        # -- Synapses groups
        for conn in self.all_connections:
            # saving type of connection (0x1 fully, 0x2 conv2d)
            fh.write(struct.pack('B', conn.type))

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

        # -- Actual neuron parameters (model itself)
        #  N x neurons:
        for i, group in enumerate(self.neuron_group):
            assert(group.number == group.thresholds.shape[0])

            # Only fully connected layers parameters are saved in each neuron synapses
            total_synapses_per_neuron = \
                sum(conn.weights.shape[1] for conn in group.connections
                    if isinstance(conn, FullyConn))

            # Saving for each neuron neuron
            #  - Neuron params
            #  - Number of synapses (M)
            #  - M x synapses
            for j in range(group.number):
                # Neuron params
                self._save_neuron_params(fh, group.thresholds[j])
                # number of synapses per neuron
                fh.write(struct.pack('>i', total_synapses_per_neuron))

                for conn in group.connections:
                    # synapses
                    if isinstance(conn, FullyConn):
                        conn.weights[j].astype('>f4').tofile(fh)


if __name__ == '__main__':
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
    msaver.save("3-to-2-to-5-neurons.doryta.bin")

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
    msaver.save("another-gol.doryta.bin")
