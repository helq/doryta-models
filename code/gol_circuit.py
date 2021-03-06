import numpy as np
from pathlib import Path

from .doryta_io.model_saver import ModelSaverLayers
from .circuits import from_json_path

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

    # NO NEGATIVE WEIGHTS
    heartbeat = 1 / 5
    sngol = from_json_path('gol/json/gol-nonnegative-v1.json', {'heartbeat': heartbeat})

    msaver = ModelSaverLayers(dt=heartbeat)
    msaver.add_conv2d_layer(k1, t1, (20, 20), (1, 1), (1, 1))
    msaver.add_sncircuit_layer(sngol, [(1, 0), (1, 1)])
    out_0 = next(iter(sngol.outputs[0]))
    msaver.add_one2one_conn(from_=(2, out_0), to=0, weight=1.0)
    msaver.save(dump_folder / "gol-20x20-circuit-nonnegative.doryta.bin")

    # NO NEGATIVE WEIGHTS AND NO LEAK
    heartbeat = 1 / 5
    sngol2 = from_json_path('gol/json/gol-no-leak.json', {'heartbeat': heartbeat})

    msaver = ModelSaverLayers(dt=heartbeat)
    msaver.add_conv2d_layer(k1, t1, (20, 20), (1, 1), (1, 1))
    msaver.add_sncircuit_layer(sngol2, [(1, 0), (1, 1)])
    out_0 = next(iter(sngol2.outputs[0]))
    msaver.add_one2one_conn(from_=(2, out_0), to=0, weight=1.0)
    msaver.save(dump_folder / "gol-20x20-no-leak.doryta.bin")
