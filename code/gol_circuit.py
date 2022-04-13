import numpy as np
from pathlib import Path

from .utils.doryta.model_saver import ModelSaverLayers
from .circuits.base import SNCircuit

if __name__ == '__main__':
    dump_folder = Path('gol/snn-models/')

    k1 = np.zeros((3, 3, 1, 2))
    k1[..., 0, 0] = [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]
    k1[..., 0, 1] = [[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]]
    t1 = np.array([2.5, 3.5])  # type: ignore

    heartbeat = 1 / 5

    sngol = SNCircuit.load_json(
        'gol/json/gol-nonnegative-v1.json', {'heartbeat': heartbeat})

    msaver = ModelSaverLayers(dt=heartbeat)
    msaver.add_conv2d_layer(k1, t1, (20, 20), (1, 1), (1, 1))
    msaver.add_sncircuit_layer(sngol, [(1, 0), (1, 1)])
    msaver.add_one2one_conn(from_=(2, 4), to=0, weight=1.0)
    msaver.save(dump_folder / "gol-20x20-circuit-nonnegative.doryta.bin")
