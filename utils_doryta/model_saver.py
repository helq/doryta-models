"""
Utilities to save NN models into files that doryta can load.
"""
from __future__ import annotations

import struct
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from typing import BinaryIO, List, Union, Any


class Conv2DConn(object):
    def __init__(self) -> None:
        pass


class FullyConn(object):
    def __init__(self, weights: NDArray[Any], biases: NDArray[Any]) -> None:
        self.weights = weights
        self.biases = biases


class ModelSaverLayers(object):
    def __init__(self, dt: float = 1/256) -> None:
        self.neuron_group: List[int] = []
        self.connections: List[Union[Conv2DConn, FullyConn]] = []
        self.dt = dt

    def add_fully_layer(self, weights: NDArray[Any], biases: NDArray[Any]) -> None:
        assert len(weights.shape) == 2
        assert weights.shape[1] == biases.shape[0]

        if len(self.neuron_group) == 0:
            self.neuron_group = list(weights.shape)
        else:
            if self.neuron_group[-1] != weights.shape[0]:
                raise Exception(
                    f"The shape of the input weights {self.neuron_group[-1]} doesn't "
                    "match with the previous layer output weights "
                    f"{weights.shape[0]}")
            self.neuron_group.append(weights.shape[1])
        self.connections.append(FullyConn(weights, biases))

    #  - Neuron params:
    #    + float potential = 0;          // V
    #    + float current = 0;            // I(t)
    #    + float resting_potential = 0;  // V_e
    #    + float reset_potential = 0;    // V_r
    #    + float threshold = 0.5 - bias; // V_th
    #    + float tau_m = dt = 1/256;     // C * R
    #    + float resistance = 1;         // R
    def _save_neuron_params(self, f: BinaryIO, bias: float) -> None:
        f.write(struct.pack('>f', 0))    # potential
        f.write(struct.pack('>f', 0))    # current
        f.write(struct.pack('>f', 0))    # resting_potential
        f.write(struct.pack('>f', 0))    # reset_potential
        f.write(struct.pack('>f', 0.5 - bias))  # threshold
        f.write(struct.pack('>f', self.dt))   # tau_m
        f.write(struct.pack('>f', 1))    # resistance

    def save(self, path: Union[str, Path]) -> None:
        if not self.neuron_group:
            raise Exception("Nothing to do. No layers defined")

        with open(path, 'wb') as fh:
            # -- Magic number
            fh.write(struct.pack('>I', 0x23432BC4))
            # -- File format
            fh.write(struct.pack('>H', 0x1))
            # -- Total number of neurons (N)
            fh.write(struct.pack('>i', sum(self.neuron_group)))
            # -- Total number of groups
            fh.write(struct.pack('B', len(self.neuron_group)))
            # -- Total number of connections
            fh.write(struct.pack('B', len(self.neuron_group) - 1))
            # -- dt
            fh.write(struct.pack('>f', self.dt))
            # -- Neuron groups ("layers")
            for size in self.neuron_group:
                fh.write(struct.pack('>i', size))
            # -- Synapses groups
            acc = 0
            for i in range(len(self.neuron_group) - 1):
                from_start = acc
                to_start = acc + self.neuron_group[i]
                to_end = to_start + self.neuron_group[i+1] - 1
                # Defining i-th fully connection
                fh.write(struct.pack('>i', from_start))    # from_start
                fh.write(struct.pack('>i', to_start - 1))  # from_end
                fh.write(struct.pack('>i', to_start))      # to_start
                fh.write(struct.pack('>i', to_end))        # to_end
                acc += self.neuron_group[i]

            # -- Actual neuron parameters (model itself)
            #  N x neurons:

            b_prev = np.zeros((self.neuron_group[0]))
            for i, conn in enumerate(self.connections):
                assert isinstance(conn, FullyConn)
                w = conn.weights
                b = b_prev
                b_prev = conn.biases
                assert(self.neuron_group[i+1] == w.shape[1])
                assert(self.neuron_group[i] == b.shape[0])

                # Saving for each neuron neuron
                #  - Neuron params
                #  - Number of synapses (M)
                #  - M x synapses
                for j in range(self.neuron_group[i]):
                    # Neuron params
                    self._save_neuron_params(fh, b[j])
                    # number of synapses per neuron
                    fh.write(struct.pack('>i', self.neuron_group[i+1]))
                    w[j].astype('>f4').tofile(fh)

            # Last layer
            for i in range(self.neuron_group[-1]):
                # Saving for each neuron neuron
                #  - Neuron params
                #  - Number of synapses (0)
                self._save_neuron_params(fh, b_prev[i])
                fh.write(struct.pack('>i', 0))  # number of synapses


if __name__ == '__main__':
    from tensorflow.keras.models import load_model
    model = load_model("/home/helq/HPC/code/doryta/data/models/whetstone/keras-simple-mnist")

    msaver = ModelSaverLayers()
    msaver.add_fully_layer(*model.layers[0].get_weights())
    msaver.add_fully_layer(*model.layers[2].get_weights())
    msaver.add_fully_layer(*model.layers[4].get_weights())
    msaver.save("one-layer-simple.bin")
