import struct
from typing import BinaryIO

import pathlib
import numpy as np

from ..circuits.sncircuit import SNCircuit, LIF, Neuron, SynapParams


def save(
    circuit: SNCircuit,
    path_name: str | pathlib.Path,
    heartbeat: float = 1 / 256,
    verbose: bool = False
) -> None:
    n_inputs = len(circuit.inputs)
    n_outputs = len(circuit.outputs)
    additional_ids = n_inputs + n_outputs

    # Shifting neuron ids `additional_ids` ahead and adding input/output neuron slots
    memoryless_neuron = LIF(1, heartbeat, 0.8)
    neurons_to_save: list[Neuron] = [
        Neuron(params=memoryless_neuron,
               synapses={i+additional_ids: synaps for i, synaps in input.items()})
        for input in circuit.inputs]
    neurons_to_save.extend(Neuron(params=memoryless_neuron, synapses={})
                           for _ in circuit.outputs)
    neurons_to_save.extend(
        Neuron(neuron.params,
               {i+additional_ids: synaps for i, synaps in neuron.synapses.items()})
        for neuron in circuit.neurons)
    # Adding connections to outputs
    for i, outputs_i in enumerate(circuit.outputs):
        for out_neu in outputs_i:
            neurons_to_save[out_neu + additional_ids].synapses[n_inputs + i] = \
                SynapParams(weight=1, delay=1)

    if verbose:
        if n_inputs:
            print(f"Neuron inputs: [0-{n_inputs-1}]")
        if n_outputs:
            print(f"Neuron outputs: [{n_inputs}-{n_inputs+n_outputs-1}]")

    # Saving network!
    with open(path_name, 'wb') as fh:
        # -- Magic number
        fh.write(struct.pack('>I', 0x23432BC4))
        # -- File format
        fh.write(struct.pack('>H', 0x11))
        # -- Neuron Type
        fh.write(struct.pack('>B', 0x1))
        # -- Total number of neurons
        fh.write(struct.pack('>i', len(neurons_to_save)))
        # -- Total number of synapses
        num_synapses = sum(len(n.synapses) for n in neurons_to_save)
        fh.write(struct.pack('>q', num_synapses))
        # -- Heartbeat
        fh.write(struct.pack('>d', heartbeat))

        for neuron in neurons_to_save:
            # Neuron params
            _save_lif_neuron_params(neuron.params, fh)

            # Neuron synapses
            num_synapses_i = len(neuron.synapses)
            fh.write(struct.pack('>i', num_synapses_i))

            synap_ids = np.zeros((num_synapses_i,), dtype='>i4')
            synap_weights = np.zeros((num_synapses_i,), dtype='>f4')
            synap_delays = np.zeros((num_synapses_i,), dtype='>i4')
            for j, (s_id, (s_weight, s_delay)) in enumerate(neuron.synapses.items()):
                synap_ids[j] = s_id
                synap_weights[j] = s_weight
                synap_delays[j] = s_delay
            synap_ids.tofile(fh)
            synap_weights.tofile(fh)
            synap_delays.tofile(fh)


def _save_lif_neuron_params(
    neuron: LIF,
    f: BinaryIO
) -> None:
    f.write(struct.pack('>f', neuron.potential))
    f.write(struct.pack('>f', neuron.current))
    f.write(struct.pack('>f', neuron.resting_potential))
    f.write(struct.pack('>f', neuron.reset_potential))
    f.write(struct.pack('>f', neuron.threshold))
    f.write(struct.pack('>f', neuron.capacitance * neuron.resistance))  # tau_m
    f.write(struct.pack('>f', neuron.resistance))
