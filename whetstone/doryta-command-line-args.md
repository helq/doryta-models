# Current (to whenever this is written) doryta arguments to run the examples in this folder

## Fully Connected

```bash
mpirun -np 2 src/doryta --spike-driven --synch=3 --extramem=1000000 \
    --load-model=../data/models/whetstone/simple-mnist.doryta.bin \
    --load-spikes=../data/models/whetstone/spikified-mnist/spikified-images-all.bin \
    --output-dir=fully-20 \
    --probe-firing --probe-firing-output-only --probe-firing-buffer=100000 --end=19.5
```

## For LeNet

```bash
mpirun -np 2 src/doryta --load-model=../data/models/whetstone/lenet-mnist.doryta.bin \
    --spike-driven --probe-{stats,firing} --extramem=1000000 \
    --load-spikes=../data/models/whetstone/spikified-mnist/spikified-images-all.bin \
    --end=19.9 --probe-firing-output-only --synch=3 --output-dir=lenet-20 \
    --probe-firing-buffer=100000
```
