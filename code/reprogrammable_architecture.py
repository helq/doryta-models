import numpy as np
from pathlib import Path

from typing import List, Tuple, Union, Dict, Any
from numpy.typing import NDArray

from .utils.doryta.model_saver import ModelSaverLayers
from .utils.doryta.spikes import save_spikes_for_doryta
from .circuits.base import SNCircuit


def load_NOTable_ANDOR_gate(circuits_folder: Path, heartbeat: float = 1) -> SNCircuit:
    xor_extern = SNCircuit.load_json(
        circuits_folder / 'xor-externally-activated.json',
        {'heartbeat': heartbeat})
    cycle3 = SNCircuit.load_json(
        circuits_folder / 'turnable-cycle-3-long.json',
        {'heartbeat': heartbeat})
    and3 = SNCircuit.load_json(
        circuits_folder / 'and-gate-3-inputs.json',
        {'heartbeat': heartbeat})

    not_gate_act = xor_extern.connect(
        cycle3, incoming=[(0, 0)], self_id='xor', other_id='cycle3')
    andOr_gate_act = and3.connect(
        cycle3, incoming=[(0, 0)], self_id='and3', other_id='cycle3')

    notableAndOr_gate_act = andOr_gate_act.connect(
        not_gate_act, outgoing=[(0, 0)], self_id='andOr', other_id='not'
    )

    return notableAndOr_gate_act


def build_reconf_gate_model(
    notableAndOr_gate_act: SNCircuit, num_gates: int = 5, heartbeat: float = 1
) -> ModelSaverLayers:
    msaver = ModelSaverLayers(dt=heartbeat)
    # Layer 0: multiple settings: deactivate AND/OR, deactivate NOT, gate pass-on
    msaver.add_neuron_group(thresholds=np.array([0.8, 0.8, 0.8]), partitions=3)
    # Layer 1: to activate AND/OR
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)
    # Layer 2: to activate NOT
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)
    # Layer 3-4: input to AND/OR neuron inputs
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)

    input_layers: List[Union[Tuple[int, int], int]]
    input_layers = [3, 4, 1, (0, 0), (0, 2), 2, (0, 1)]
    #                                            inputs to notableAndOr_gate_act circuit
    # layer 3      <- 'andOr.and3.gate-output' <- {0: SynapParams(weight=0.5, delay=1)},
    # layer 4      <- 'andOr.and3.gate-output' <- {0: SynapParams(weight=0.5, delay=1)},
    # layer 1      <- 'andOr.cycle3.start'     <- {3: SynapParams(weight=1.0, delay=1)},
    # layer (0, 0) <- 'andOr.cycle3.stop'      <- {2: SynapParams(weight=1.0, delay=1)},
    # layer (0, 2) <- 'not.xor.pass-on'        <- {6: SynapParams(weight=0.5, delay=1)},
    # layer 2      <- 'not.cycle3.start'       <- {9: SynapParams(weight=1.0, delay=1)},
    # layer (0, 1) <- 'not.cycle3.stop'        <- {8: SynapParams(weight=1.0, delay=1)}
    msaver.add_sncircuit_layer(notableAndOr_gate_act, input_layers)

    return msaver


if __name__ == '__main__':
    dump_folder = Path('snn-circuits/')
    num_gates = 5

    # ===== Saving circuit ======
    circuit = load_NOTable_ANDOR_gate(circuits_folder=Path('snn-circuits/json/'))
    msaver = build_reconf_gate_model(circuit, num_gates=5)
    msaver.save(dump_folder / 'snn-models' / f"reconfigurable-{num_gates}-gates.doryta.bin")

    # ====== Saving spikes ======
    # params
    stop_andOr = 0
    stop_not = 1
    pass_on = 2
    andOr_act_shift = 3
    not_act_shift = andOr_act_shift + num_gates
    input1_shift = andOr_act_shift + 2 * num_gates
    input2_shift = input1_shift + num_gates
    output_layer_shift = input2_shift + num_gates + circuit.outputs[0] * num_gates

    spikes: Dict[int, NDArray[Any]] = {
        # Global parameters (pass_on for NOT gate, and reset params)
        pass_on:             4 + 3 * np.array([0, 1, 2]),
        stop_andOr:          1 + 3 * np.array([2]),  # deactivates on the next iter
        stop_not:            2 + 3 * np.array([2]),
        # The inputs for all gates are the same:
        # - two spikes at time "0" (0)
        # - one spike at time "1" (3)
        # - no spikes at time "2" (6)
        # AND gate (output only at time "1" (6))
        (input1_shift + 0):  1 + 3 * np.array([0, 1]),
        (input2_shift + 0):  1 + 3 * np.array([0]),
        # OR gate (output at times "1" and "2" (6 and 9))
        (andOr_act_shift + 1):   3 * np.array([0]),
        (input1_shift + 1):  1 + 3 * np.array([0, 1]),
        (input2_shift + 1):  1 + 3 * np.array([0]),
        # NAND gate (output at times "2" and "3" (9 and 12))
        (not_act_shift + 2): 1 + 3 * np.array([0]),
        (input1_shift + 2):  1 + 3 * np.array([0, 1]),
        (input2_shift + 2):  1 + 3 * np.array([0]),
        # NOR gate (output at time "3" (12))
        (andOr_act_shift + 3):   3 * np.array([0]),
        (not_act_shift + 3): 1 + 3 * np.array([0]),
        (input1_shift + 3):  1 + 3 * np.array([0, 1]),
        (input2_shift + 3):  1 + 3 * np.array([0]),
    }
    save_spikes_for_doryta(
        None, None,
        dump_folder / 'spikes' / f'testing-num_gates={num_gates}',
        additional_spikes=spikes
    )

    print("The output gate neurons reside on range "
          f"[{output_layer_shift}, {output_layer_shift + num_gates - 1}]")
