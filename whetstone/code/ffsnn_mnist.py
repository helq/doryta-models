"""
Simple Feed-forward Spiking Neural Network for MNIST

This file implements a SNN model using whetstone and keras. The "main" is in charge of
everything. Each of the following options can be activated by switching from False to True
on any of them:

- Train model / Load model from memory
- Check accuracy of model (about 96%)
- Save model
- Save images
"""

from __future__ import annotations

from typing import Any, Tuple, BinaryIO, Union
import pathlib
import os
# import sys
import numpy as np
import struct

# import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from .whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from .whetstone.callbacks import SimpleSharpener, WhetstoneLogger

from .common_mnist import my_key, plot_img


def load_data(
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:  # type: ignore
    # Loading and preprocessing data
    numClasses = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    y_train = to_categorical(y_train, numClasses)
    y_test = to_categorical(y_test, numClasses)

    x_train = np.reshape(x_train, (60000, 28*28))
    x_test = np.reshape(x_test, (10000, 28*28))
    return (x_train, y_train), (x_test, y_test)


def create_model(initializer: Any = 'glorot_uniform',
                 use_my_key: bool = True) -> Tuple[Any, Any]:
    if use_my_key:
        key = my_key()
    else:
        key = key_generator(num_classes=10, width=100)

    model = Sequential()
    model.add(Dense(256, input_shape=(28*28,), kernel_initializer=initializer))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(64, kernel_initializer=initializer))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(100, kernel_initializer=initializer))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Softmax_Decode(key))  # type: ignore

    # Disabling training of Softmax Decode layer. I really don't need any of its magic as
    # it obfuscates the inference process (and I would have to carry around the keras
    # model, or the trained key, in order to make sense of the SNN output). This is also
    # the reason why `use_my_key` is True by default
    model.layers[-1].trainable = False

    model_intermediate = Model(
        inputs=model.inputs,
        outputs=[model.layers[2*i + 1].output for i in range(3)])

    return model, model_intermediate


def create_callbacks() -> Any:
    simple = SimpleSharpener(start_epoch=5, steps=5, epochs=True, bottom_up=True)  # type: ignore

    # Create a new directory to save the logs in.
    log_dir = './simple_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)  # type: ignore

    return [simple, logger]


def save_model_for_doryta(model: Any, path: str) -> None:
    with open(path, 'wb') as fh:
        size_groups = [28*28, 256, 64, 100]

        # -- Magic number
        fh.write(struct.pack('>I', 0x23432BC4))
        # -- File format
        fh.write(struct.pack('>H', 0x1))
        # -- Total number of neurons (N)
        fh.write(struct.pack('>i', sum(size_groups)))
        # -- Total number of groups
        fh.write(struct.pack('B', len(size_groups)))
        # -- Total number of connections
        fh.write(struct.pack('B', len(size_groups) - 1))
        # -- dt
        fh.write(struct.pack('>f', 1/256))
        # -- Neuron groups (layers)
        for size in size_groups:
            fh.write(struct.pack('>i', size))
        # -- Synapses groups
        acc = 0
        for i in range(len(size_groups) - 1):
            # Defining i-th fully connection
            fh.write(struct.pack('>i', acc))    # from_start
            fh.write(struct.pack('>i', acc + size_groups[i] - 1))  # from_end
            fh.write(struct.pack('>i', acc + size_groups[i]))  # to_start
            fh.write(struct.pack('>i', acc + size_groups[i] + size_groups[i+1] - 1))  # to_end
            acc += size_groups[i]

        # -- Actual neuron parameters (model itself)
        #  N x neurons:
        #  - Neuron params:
        #    + float potential = 0;          // V
        #    + float current = 0;            // I(t)
        #    + float resting_potential = 0;  // V_e
        #    + float reset_potential = 0;    // V_r
        #    + float threshold = 0.5 - bias; // V_th
        #    + float tau_m = dt = 1/256;     // C * R
        #    + float resistance = 1;         // R
        #  - Synapses for neuron:
        #    + Number of synapses (M)
        #    + Range of synapses [This is basically: to_start and to_end]
        #    + M x synapses

        # The parameters for all neurons are almost the same
        def neuron_params(f: BinaryIO, bias: float) -> None:
            f.write(struct.pack('>f', 0))    # potential
            f.write(struct.pack('>f', 0))    # current
            f.write(struct.pack('>f', 0))    # resting_potential
            f.write(struct.pack('>f', 0))    # reset_potential
            f.write(struct.pack('>f', 0.5 - bias))  # threshold
            f.write(struct.pack('>f', 1/256))  # tau_m
            f.write(struct.pack('>f', 1))    # resistance

        w0, b0 = model.layers[0].get_weights()
        w1, b1 = model.layers[2].get_weights()
        w2, b2 = model.layers[4].get_weights()
        # Neurons for first layer
        for i in range(28 * 28):
            neuron_params(fh, 0)
            fh.write(struct.pack('>i', 256))   # number of synapses per neuron
            w0[i].astype('>f4').tofile(fh)
        assert(256 == w0.shape[1])

        # Neurons for second layer
        for i in range(256):
            neuron_params(fh, b0[i])
            fh.write(struct.pack('>i', 64))   # number of synapses
            w1[i].astype('>f4').tofile(fh)
        assert(64 == w1.shape[1])

        # Third layer
        for i in range(64):
            neuron_params(fh, b1[i])
            fh.write(struct.pack('>i', 100))  # number of synapses
            w2[i].astype('>f4').tofile(fh)
        assert(100 == w2.shape[1])

        # Fourth layer
        for i in range(100):
            neuron_params(fh, b2[i])
            fh.write(struct.pack('>i', 0))   # number of synapses


# NOTE: shift cannot be arbitrarily small because time stamps are being stored as 32-bit
# floating point numbers! A small number like 0.00001 will dissapear when added up to
# 4000. I thought 32-bits were enough for those kind of computations, they are not. This
# might be the root of many subtle bugs, dammit
def save_spikes_for_doryta(
    img: np.ndarray[Any, Any],
    path: str,
    format: int = 2,
    shift: float = 0.0
) -> None:
    assert(len(img.shape) == 2)
    assert(img.shape[1] == 28*28)
    with open(f"{path}.bin", 'wb') as fh:
        # Magic number
        fh.write(struct.pack('>I', 0x23432BC5))
        if format == 1:
            # File format (0x1 spikes ordered in chunks of time - 0x2 ordered by neuron)
            fh.write(struct.pack('>H', 0x1))
            # Total instants in time
            fh.write(struct.pack('>i', img.shape[0]))
            # Total spikes
            fh.write(struct.pack('>i', img.sum()))
            # For each instant in time
            for i, img_i in enumerate(img):
                # - Instant in time
                fh.write(struct.pack('>f', i))
                # - Number of neurons
                fh.write(struct.pack('>i', img_i.sum()))
                # - Neuron ids
                np.flatnonzero(img_i).astype('>i4').tofile(fh)
        elif format == 2:
            # File format (0x1 spikes ordered in chunks of time - 0x2 ordered by neuron)
            fh.write(struct.pack('>H', 0x2))
            n_spikes_per_neuron = img.sum(axis=0)
            active_neurons = np.flatnonzero(n_spikes_per_neuron)
            # Total neurons
            fh.write(struct.pack('>i', active_neurons.shape[0]))
            # Total spikes
            fh.write(struct.pack('>i', img.sum()))
            # For each instant in time
            for neuron_i in active_neurons:
                # - Neuron id
                fh.write(struct.pack('>i', neuron_i))
                # - Number of spikes for neuron
                fh.write(struct.pack('>i', n_spikes_per_neuron[neuron_i]))
                # - Neuron times
                (np.flatnonzero(img[:, neuron_i]) + shift).astype('>f4').tofile(fh)  # type: ignore
        else:
            raise Exception("No other way to store spikes has been defined yet")


def save_tags_for_doryta(tags: Any, path: str) -> None:
    with open(f"{path}.tags.bin", 'wb') as fp:
        tags.argmax(axis=1).astype('b').tofile(fp)


def load_models(path: Union[str, pathlib.Path]) -> Tuple[Any, Any]:
    model = load_model(path)
    model_intermediate = Model(
        inputs=model.inputs,
        outputs=[model.layers[2*i + 1].output for i in range(3)])
    return model, model_intermediate


def show_prediction(
    model: Any, model_intermediate: Any,
    img: Any, show_all_layers: bool = False
) -> None:

    output_layers = model_intermediate.predict(img)
    output_spike = output_layers[-1]

    if show_all_layers:
        # Checking other things model
        img_spiked = output_layers[0]
        print("Spikes on first layer:")
        print(img_spiked)
        print("Spikes at (shifted): ", np.argwhere(img_spiked.flatten()).flatten() + 28*28)

        # Checking other things model
        img_spiked = output_layers[1]
        print("Spikes on second layer:")
        print(img_spiked)
        print("Spikes at (shifted): ",
              np.argwhere(img_spiked.flatten()).flatten() + 28*28 + 256)

        # print("Input for network as intesity spikes (too large to show properly):")
        # print(spikes_before_bias)
        # with np.printoptions(threshold=sys.maxsize):
        #     print(spikes_before_bias)

    print("Predicted value:", model.predict(img).argmax(1))

    print("Predicted value (spike output):")
    print(output_spike.reshape((-1, 10, 10)))

    out_spikes_num = np.argwhere(output_spike.flatten() == 1).flatten()
    print("Spikes at (shifted): ", out_spikes_num + 28*28 + 256 + 64)
    print("Spikes at: ", out_spikes_num)
    plot_img(img)

    # w0, b0 = model.layers[0].get_weights()
    # spikes_before_bias = w0 * img.reshape((28*28, 1))
    # img_spiked2 = ((img @ w0 + b0) >= 0.5).astype(int)
    # img_spiked3 = ((spikes_before_bias.sum(0) + b0) >= 0.5).astype(int)
    # assert((img_spiked == img_spiked2).all())
    # assert((img_spiked == img_spiked3).all())


def save_spikes_slice(x_test: Any, y_test: Any, sl: slice) -> None:
    if sl.start:
        name = f"spikified-mnist/spikified-images-{sl.start}-to-{sl.stop}"
    else:
        name = f"spikified-mnist/spikified-images-{sl.stop}"
    img = (x_test[sl] > .5).astype(int)
    save_spikes_for_doryta(img, name)
    save_tags_for_doryta(y_test[sl], name)
    print("Classes of images:", y_test[sl].argmax(axis=1))


if __name__ == '__main__':  # noqa: C901
    # This is super good but produces negative values for the matrix, ie, negative currents :S
    initializer = 'glorot_uniform'
    # initializer = RandomUniform(minval=0.0, maxval=1.0)

    loading_model = True
    train_model = False
    checking_model = False
    save_model = False

    model_path = 'keras-simple-mnist'

    (x_train, y_train), (x_test, y_test) = load_data()

    if loading_model:
        model, model_intermediate = load_models(model_path)

    elif train_model:
        model, model_intermediate = create_model(initializer=initializer)
        callbacks = create_callbacks()

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128,
                  epochs=21, callbacks=callbacks)
        model.save(model_path)

    if loading_model or train_model:
        if checking_model:
            print("Evaluating model (loss, accuracy):", model.evaluate(x_test, y_test))
            # new model allows us to extract the results of a layer

            imgs = (x_test > .5).astype(float)  # type: ignore
            prediction = model.predict(imgs).argmax(axis=1)
            correct_predictions = (prediction == y_test.argmax(axis=1)).sum()
            print("Evaluating model (accuracy on black&white images):",
                  correct_predictions / imgs.shape[0])

        # Saving doryta model to memory
        if save_model:
            save_model_for_doryta(model, "simple-mnist.doryta.bin")

    # Saving first 20 images in testing dataset
    if False:
        # save_spikes_slice(x_test, y_test, slice(20))
        save_spikes_slice(x_test, y_test, slice(1950, 2000))

    # Saving all images as spikes
    if True:
        range_ = ...
        path = "spikified-mnist/spikified-images-all-shifted"

        imgs = (x_test[range_] > .5).astype(int)
        print("Total images:", y_test[range_].shape[0])

        save_spikes_for_doryta(imgs, path, shift=.01)
        save_tags_for_doryta(y_test[range_], path)

        print(f"Spikified images stored to `{path}.bin` and "
              f"its tags to `{path}.tags.bin`")

    # Saving one random image
    if False:
        i = np.random.randint(0, x_test.shape[0]-1)
        img = (x_test[i:i+1] > .5).astype(int)
        klass = y_test[i].argmax()
        save_spikes_for_doryta(img, f"spikified-mnist/spikified-images-class={klass}", format=1)
        print("Classes of images:", klass)
        if loading_model or train_model:
            show_prediction(model, img)

    # Checking a single model
    if False:
        i = 43
        print("Classes of images:", y_test[i].argmax())
        if loading_model:
            show_prediction(model, model_intermediate, (x_test[i:i+1] > .5).astype(int))
