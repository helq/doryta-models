# Doryta Models

This repo contains both, models ready to be loaded by [doryta][] and the raw models (from
other libraries like keras) used to generate them in the first place. Some come with the
code used to generate them.

[doryta]: https://github.com/helq/doryta

## Running in Doryta examples

Running fully connected:

```bash
mpirun -np 2 path_to/doryta --spike-driven --synch=2 --extramem=1000000 \
    --load-model=models/mnist/snn-models/ffsnn-mnist.doryta.bin \
    --load-spikes=models/mnist/spikes/spikified-mnist/spikified-images-all.bin \
    --output-dir=fully-20 \
    --probe-firing --probe-firing-output-only --probe-firing-buffer=100000 --end=19.5
```

Running LeNet:

```bash
mpirun -np 2 path_to/doryta --load-model=models/whetstone/lenet-mnist-filters=6,16.doryta.bin \
    --spike-driven --probe-{stats,firing} --extramem=1000000 \
    --load-spikes=models/mnist/spikes/spikified-mnist/spikified-images-all.bin \
    --end=19.9 --probe-firing-output-only --synch=2 --output-dir=lenet-20 \
    --probe-firing-buffer=100000
```
