import numpy as np
from pathlib import Path

from typing import List, Tuple, Union, Dict, Any
from numpy.typing import NDArray

from .doryta_io.model_saver import ModelSaverLayers
from .doryta_io.spikes import save_spikes_for_doryta
from .circuits import SNCircuit, from_json_path


def load_NOTable_ANDOR_gate(circuits_folder: Path, heartbeat: float = 1) -> SNCircuit:
    xor_extern = from_json_path(
        circuits_folder / 'xor-externally-activated.json',
        {'heartbeat': heartbeat})
    cycle3 = from_json_path(
        circuits_folder / 'turnable-cycle-3-long.json',
        {'heartbeat': heartbeat})
    and3 = from_json_path(
        circuits_folder / 'and-gate-3-inputs.json',
        {'heartbeat': heartbeat})

    not_gate_act = xor_extern.connect(
        cycle3, incoming=[(0, 0)], self_id='xor', other_id='cycle3')
    andOr_gate_act = and3.connect(
        cycle3, incoming=[(0, 0)], self_id='and3', other_id='cycle3')

    aor_gate_act = andOr_gate_act.connect(
        not_gate_act, outgoing=[(0, 0)], self_id='andOr', other_id='not'
    )

    return aor_gate_act


def build_reconf_gate_model_1(
    aor_gate_act: SNCircuit, num_gates: int = 5, heartbeat: float = 1
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
    #                                            inputs to aor_gate_act circuit
    # layer 3      <- 'andOr.and3.gate-in/output' <- {0: SynapParams(weight=0.5, delay=1)},
    # layer 4      <- 'andOr.and3.gate-in/output' <- {0: SynapParams(weight=0.5, delay=1)},
    # layer 1      <- 'andOr.cycle3.start'        <- {3: SynapParams(weight=1.0, delay=1)},
    # layer (0, 0) <- 'andOr.cycle3.stop'         <- {2: SynapParams(weight=1.0, delay=1)},
    # layer (0, 2) <- 'not.xor.pass-on'           <- {6: SynapParams(weight=0.5, delay=1)},
    # layer 2      <- 'not.cycle3.start'          <- {9: SynapParams(weight=1.0, delay=1)},
    # layer (0, 1) <- 'not.cycle3.stop'           <- {8: SynapParams(weight=1.0, delay=1)}
    msaver.add_sncircuit_layer(aor_gate_act, input_layers)

    return msaver


def build_reconf_gate_model_2(
    aor_gate_act: SNCircuit,
    gate_keeper: SNCircuit,
    num_gates: int = 5, heartbeat: float = 1
) -> ModelSaverLayers:
    # this line removes the duplicated input to neuron 'andOr.and3.in/gate-output'
    # This connection is later realized by a conv2d connection
    aor_gate_act = aor_gate_act.remove_inputs([0, 1])

    msaver = ModelSaverLayers(dt=heartbeat)
    # Layer 0: multiple settings: deactivate AND/OR, NOT and gate keeper; and gate pass-on
    msaver.add_neuron_group(thresholds=np.array([0.8, 0.8, 0.8, 0.8]), partitions=4)
    # Layer 1: to activate AND/OR
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)
    # Layer 2: to activate NOT
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)
    # Layer 3: to activate keeper gates (routing from input to AND/OR gates)
    msaver.add_neuron_group(thresholds=np.ones((num_gates * num_gates,)) * 0.8)
    # Layer 4: input to circuit
    msaver.add_neuron_group(thresholds=np.ones((num_gates,)) * 0.8)

    # Layer 5: inputs to gate circuits
    # 'start' -> layer 3
    # 'stop' ->  layer (0, 2)
    msaver.add_sncircuit_layer(gate_keeper, [3, (0, 2)])
    # connections from input to gate keepers
    msaver.add_conv2d_conn(
        kernel=np.ones((num_gates, 1)) * 0.5,
        input_size=(1, num_gates),
        from_=4,
        to=(5, gate_keeper.ids_to_int['gate']),
        padding=(num_gates-1, 0))

    # Layer 6:
    input_layers: List[Union[Tuple[int, int], int]]
    input_layers = [1, (0, 0), (0, 3), 2, (0, 1)]
    #                                            inputs to aor_gate_act circuit
    # layer 1      <- 'andOr.cycle3.start'     <- {3: SynapParams(weight=1.0, delay=1)},
    # layer (0, 0) <- 'andOr.cycle3.stop'      <- {2: SynapParams(weight=1.0, delay=1)},
    # layer (0, 3) <- 'not.xor.pass-on'        <- {6: SynapParams(weight=0.5, delay=1)},
    # layer 2      <- 'not.cycle3.start'       <- {9: SynapParams(weight=1.0, delay=1)},
    # layer (0, 1) <- 'not.cycle3.stop'        <- {8: SynapParams(weight=1.0, delay=1)}
    msaver.add_sncircuit_layer(aor_gate_act, input_layers)

    # connections from gate keepers to AND/OR gates
    msaver.add_conv2d_conn(
        kernel=np.ones((1, num_gates)) * 0.5,
        input_size=(num_gates, num_gates),
        from_=(5, gate_keeper.ids_to_int['gate']),
        to=(6, aor_gate_act.ids_to_int['andOr.and3.gate-in/output']))

    return msaver


def spikes_gate_model_1(
    num_gates: int, aor_gate_act: SNCircuit
) -> Tuple[Dict[int, NDArray[Any]], int]:
    # params
    stop_andOr = 0
    stop_not = 1
    pass_on = 2
    andOr_act_shift = 3
    not_act_shift = andOr_act_shift + num_gates
    input1_shift = andOr_act_shift + 2 * num_gates
    input2_shift = input1_shift + num_gates
    # the circuit has only one output
    output_layer_shift = input2_shift + num_gates + aor_gate_act.outputs[0] * num_gates

    spikes = {
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

    return spikes, output_layer_shift


def spikes_gate_model_2(
    num_gates: int, aor_gate_act: SNCircuit, gate_keeper_circuit: SNCircuit
) -> Tuple[Dict[int, NDArray[Any]], int]:
    assert num_gates >= 4
    # params
    print("----- Configuration neurons -----")
    # layer 0
    stop_andOr = 0
    stop_not = 1
    stop_gate_keeper = 2
    pass_on = 3
    print(f"Layer 0: [stop_andOr = {stop_andOr}, stop_not = {stop_not}, "
          f"stop_gate_keeper = {stop_gate_keeper}, pass_on = {pass_on}]")
    # layer 1
    andOr_act_shift = 4
    layer_size = num_gates
    print(f"Layer 1: [{andOr_act_shift}, {andOr_act_shift + layer_size - 1}]\t"
          "activate AND/OR gate")
    # layer 2
    not_act_shift = andOr_act_shift + layer_size
    layer_size = num_gates
    print(f"Layer 2: [{not_act_shift}, {not_act_shift + layer_size - 1}]\t"
          "activate NOT gate")
    # layer 3
    gate_keeper_act_shift = not_act_shift + layer_size
    layer_size = num_gates**2
    print(f"Layer 3: [{gate_keeper_act_shift}, {gate_keeper_act_shift + layer_size - 1}]\t"
          "activate keeper gates")
    # layer 4
    input_shift = gate_keeper_act_shift + layer_size
    layer_size = num_gates
    print(f"Layer 4: [{input_shift}, {input_shift + layer_size - 1}]\t"
          "input to circuit")
    print("----- Boolean logic neurons -----")
    # layer 5
    gate_keeper_shift = input_shift + layer_size
    layer_size = gate_keeper_circuit.num_neurons * num_gates**2
    print(f"Layer 5: [{gate_keeper_shift}, {gate_keeper_shift + layer_size - 1}]\t"
          "keeper gate circuit")
    for neuron_name, neu_num in gate_keeper_circuit.ids_to_int.items():
        neuron_shift = gate_keeper_shift + neu_num * num_gates**2
        print(f" Neuron '{neuron_name}': [{neuron_shift}, {neuron_shift + num_gates**2 - 1}]")
    # layer 6
    andOr_circuit_shift = gate_keeper_shift + layer_size
    layer_size = aor_gate_act.num_neurons * num_gates
    print(f"Layer 6: [{andOr_circuit_shift}, {andOr_circuit_shift + layer_size - 1}]\t"
          "AND/OR NOT circuit")
    for neuron_name, neu_num in aor_gate_act.ids_to_int.items():
        neuron_shift = andOr_circuit_shift + neu_num * num_gates
        print(f" Neuron '{neuron_name}': [{neuron_shift}, {neuron_shift + num_gates - 1}]")
    # layer 6.out_0
    output_layer_shift = andOr_circuit_shift + aor_gate_act.outputs[0] * num_gates

    n = num_gates

    spikes = {
        # Global parameters (pass_on for NOT gate, and reset params)
        pass_on:               5 + 3 * np.array([0, 1, 2, 3]),
        stop_andOr:            2 + 3 * np.array([3]),  # deactivates on the next iter
        stop_not:              3 + 3 * np.array([3]),
        stop_gate_keeper:      1 + 3 * np.array([3]),
        # Connecting inputs 0 and 1 to gates 0, 1, 2 and 3
        (gate_keeper_act_shift + 0 * n + 0): 3 * np.array([0]),
        (gate_keeper_act_shift + 0 * n + 1): 3 * np.array([0]),
        (gate_keeper_act_shift + 1 * n + 0): 3 * np.array([0]),
        (gate_keeper_act_shift + 1 * n + 1): 3 * np.array([0]),
        (gate_keeper_act_shift + 2 * n + 0): 3 * np.array([0]),
        (gate_keeper_act_shift + 2 * n + 1): 3 * np.array([0]),
        (gate_keeper_act_shift + 3 * n + 0): 3 * np.array([0]),
        (gate_keeper_act_shift + 3 * n + 1): 3 * np.array([0]),
        # The inputs for all gates are the same:
        # - two spikes at time "0" (1)
        # - one spike at time "1" (4)
        # - one spike at time "2" (7)
        # - no spikes at time "3" (10)
        (input_shift + 0):     1 + 3 * np.array([0, 1]),
        (input_shift + 1):     1 + 3 * np.array([0, 2]),
        # AND gate (output only at time "1" (7))
        # no activation
        # OR gate (output at times "1" and "2" (7 and 10))
        (andOr_act_shift + 1): 1 + 3 * np.array([0]),
        # NAND gate (output at times "2" and "3" (10, 13 and 16))
        (not_act_shift + 2):   2 + 3 * np.array([0]),
        # NOR gate (output at time "3" (16))
        (andOr_act_shift + 3): 1 + 3 * np.array([0]),
        (not_act_shift + 3):   2 + 3 * np.array([0]),
    }

    return spikes, output_layer_shift


if __name__ == '__main__':
    dump_folder = Path('snn-circuits/')
    num_gates = 5

    heartbeat = 1
    aor_gate_act = load_NOTable_ANDOR_gate(dump_folder / 'json', heartbeat)

    # Gate structure model 1
    if True:
        # ===== Saving circuit ======
        msaver = build_reconf_gate_model_1(aor_gate_act, num_gates=5)
        msaver.save(dump_folder / 'snn-models' / f"reconfigurable-{num_gates}-gates.doryta.bin")

        # ====== Saving spikes ======
        spikes, output_layer_shift = spikes_gate_model_1(num_gates, aor_gate_act)
        save_spikes_for_doryta(
            None, None,
            dump_folder / 'spikes' / f'reconfigurable-gate-model-v1-test1-gates={num_gates}',
            additional_spikes=spikes
        )

        print("The output gate neurons reside on range "
              f"[{output_layer_shift}, {output_layer_shift + num_gates - 1}]")

    # Gate structure model 2
    if False:
        # ===== Saving circuit ======
        gate_keeper_circuit = from_json_path(dump_folder / 'json' / 'gate-keeper.json',
                                             {'heartbeat': heartbeat})
        msaver = build_reconf_gate_model_2(aor_gate_act, gate_keeper_circuit,
                                           num_gates=num_gates)
        msaver.save(dump_folder / 'snn-models'
                    / f"reconfigurable-gate-model-v2-gates={num_gates}.doryta.bin")

        # ====== Saving spikes ======
        spikes, output_layer_shift = \
            spikes_gate_model_2(num_gates, aor_gate_act, gate_keeper_circuit)
        save_spikes_for_doryta(
            None, None,
            dump_folder / 'spikes' / f'reconfigurable-gate-model-v2-test1-gates={num_gates}',
            additional_spikes=spikes
        )

        print("The output gate neurons reside on range "
              f"[{output_layer_shift}, {output_layer_shift + num_gates - 1}]")
