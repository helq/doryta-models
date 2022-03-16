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

from typing import Any, Tuple, Union
import pathlib
import os
# import sys
import numpy as np
import struct
from utils.doryta.model_saver import ModelSaverLayers
from utils.doryta.spikes import save_spikes_for_doryta
from utils.temp_encoding import img_to_tempencoding

# import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomUniform
import tensorflow.keras.constraints as constraints

from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
# from whetstone.callbacks import SimpleSharpener, WhetstoneLogger, AdaptiveSharpener
from whetstone.callbacks import WhetstoneLogger, AdaptiveSharpener

from utils.common_mnist import my_key, plot_img, load_data
from utils.common import keras_model_path, doryta_model_path


def create_model(initializer: Any = 'glorot_uniform',
                 use_my_key: bool = True,
                 weight_constraints: Any = None) -> Tuple[Any, Any]:
    if isinstance(initializer, str):
        val = initializer
        initializer = lambda _: val  # noqa: E731

    if use_my_key:
        key = my_key()
    else:
        key = key_generator(num_classes=10, width=100)

    model = Sequential()
    model.add(Dense(256, input_shape=(28*28,),
                    kernel_initializer=initializer(28*28),
                    kernel_constraint=weight_constraints))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(64, kernel_initializer=initializer(256),
                    kernel_constraint=weight_constraints))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(100, kernel_initializer=initializer(64),
                    kernel_constraint=weight_constraints))
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


# NOTE: shift cannot be arbitrarily small because time stamps are being stored as 32-bit
# floating point numbers! A small number like 0.00001 will dissapear when added up to
# 4000. I thought 32-bits had enough precision for that kind of computation, they are not
def save_spikes_for_doryta_legacy(
    img: np.ndarray[Any, Any],
    path: str,
    format: int = 2,
    shift: float = 0.0
) -> None:
    assert(len(img.shape) == 2)
    # assert(img.shape[1] == 28*28)
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
                (np.flatnonzero(img[:, neuron_i]) + shift).astype('>f4').tofile(fh)
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
    save_spikes_for_doryta_legacy(img, name)
    save_tags_for_doryta(y_test[sl], name)
    print("Classes of images:", y_test[sl].argmax(axis=1))


if __name__ == '__main__':  # noqa: C901
    # This is super good but produces negative values for the matrix, ie, negative currents :S
    # initializer = 'glorot_uniform'
    # weight_constraints = None
    # # sharpener params
    # start_epoch = 5
    # sharp_steps = 5
    # total_brelus = 3

    # This is experimental. Forcing non-negative weights
    # It works, but the initialization has to be precise (and there is a considerable loss
    # in accuracy!)
    initializer: Any = lambda x: RandomUniform(minval=0.0, maxval=1.0 / x)  # noqa: E731
    weight_constraints = constraints.NonNeg()
    # sharpener params
    start_epoch = 10
    sharp_steps = 5
    total_brelus = 3

    temporal_encoding = True

    loading_model = True
    training_model = False
    checking_model = True
    saving_model = False

    dataset = 'mnist'
    # dataset = 'fashion-mnist'

    if initializer == 'glorot_uniform':
        model_path = f'ffsnn-{dataset}'
    elif isinstance(weight_constraints, constraints.NonNeg):
        model_path = f'ffsnn-{dataset}-nonnegative'
    else:
        model_path = f'ffsnn-{dataset}-nonstandard-weights'

    (x_train, y_train), (x_test, y_test) = load_data(dataset)

    if loading_model:
        model, model_intermediate = load_models(keras_model_path / model_path)

    elif training_model:
        model, model_intermediate = create_model(initializer=initializer,
                                                 weight_constraints=weight_constraints)

        # Create a new directory to save the logs in.
        log_dir = keras_model_path / 'logs' / model_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        epochs = start_epoch + sharp_steps * total_brelus + 1

        # simple = SimpleSharpener(start_epoch=start_epoch, steps=sharp_steps,  # type: ignore
        #                          epochs=True, bottom_up=True)
        # logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)  # type: ignore

        adapt = AdaptiveSharpener(min_init_epochs=start_epoch)  # type: ignore
        logger = WhetstoneLogger(logdir=log_dir, sharpener=adapt)  # type: ignore

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128,
                  # epochs=epochs, callbacks=[logger, simple])
                  epochs=epochs, callbacks=[adapt, logger])
        model.save(keras_model_path / model_path)

    if loading_model or training_model:
        if checking_model:
            model.summary()
            print("Evaluating model (loss, accuracy):", model.evaluate(x_test, y_test))
            # new model allows us to extract the results of a layer

            imgs = (x_test > .5).astype(float)  # type: ignore
            prediction = model.predict(imgs).argmax(axis=1)
            correct_predictions = (prediction == y_test.argmax(axis=1)).sum()
            print("Evaluating model (accuracy on black&white images):",
                  correct_predictions / imgs.shape[0])

        # Saving doryta model to memory
        if saving_model:
            if not temporal_encoding:
                msaver = ModelSaverLayers()
                w1, b1 = model.layers[0].get_weights()
                w2, b2 = model.layers[2].get_weights()
                w3, b3 = model.layers[4].get_weights()
                msaver.add_fully_layer(w1, .5 - b1)
                msaver.add_fully_layer(w2, .5 - b2)
                msaver.add_fully_layer(w3, .5 - b3)
                msaver.save(doryta_model_path / f"ffsnn-{dataset}.doryta.bin")
            else:
                msaver = ModelSaverLayers(dt=1/256)
                w1, b1 = model.layers[0].get_weights()
                w2, b2 = model.layers[2].get_weights()
                w3, b3 = model.layers[4].get_weights()
                R = 4
                capacitance = 1/256  # = dt
                neuron_args = {
                    'resistance': R,
                    'tau': capacitance * R
                }
                msaver.add_fully_layer(w1, .5 - b1 + 10, neuron_args=neuron_args)
                msaver.add_fully_layer(w2, .5 - b2)
                msaver.add_fully_layer(w3, .5 - b3)

                # Adding a neuron that triggers the second layer
                msaver.add_neuron_group(0.5 * np.ones((1,)))
                weights = 10 * np.ones((1, w1.shape[1]))
                msaver.add_fully_conn(from_=4, to=1, weights=weights)

                msaver.save(doryta_model_path /
                            f"ffsnn-{dataset}-tempencode-R={R}.doryta.bin", version=2)

    # finding images that were correctly classified when using full grayscale images but
    # were not anymore when converted to white&black
    if False:
        prediction = model.predict(x_test).argmax(axis=1)
        prediction_bin = model.predict((x_test > .5).astype(float)).argmax(axis=1)
        lost_in_spikying = np.flatnonzero(
            np.bitwise_and(
                prediction == y_test.argmax(axis=1),
                prediction_bin != y_test.argmax(axis=1)))

    # Saving first 20 images in testing dataset
    if False:
        # save_spikes_slice(x_test, y_test, slice(20))
        save_spikes_slice(x_test, y_test, slice(1950, 2000))

    # Saving all images as spikes
    if False:
        range_ = ...
        path = f"spikified-{dataset}/spikified-images-all-shifted"

        imgs = (x_test[range_] > .5).astype(int)
        print("Total images:", y_test[range_].shape[0])

        save_spikes_for_doryta_legacy(imgs, path, shift=.01)
        save_tags_for_doryta(y_test[range_], path)

        print(f"Spikified images stored to `{path}.bin` and "
              f"its tags to `{path}.tags.bin`")

    # Saving one random image
    if False:
        i = np.random.randint(0, x_test.shape[0]-1)
        img = (x_test[i:i+1] > .5).astype(int)
        klass = y_test[i].argmax()
        save_spikes_for_doryta_legacy(img, f"spikified-mnist/spikified-images-class={klass}",
                                      format=1)
        print("Classes of images:", klass)
        if loading_model or training_model:
            show_prediction(model, model_intermediate, img)

    # Checking a single model
    if False:
        i = 43
        print("Classes of images:", y_test[i].argmax())
        if loading_model:
            show_prediction(model, model_intermediate, (x_test[i:i+1] > .5).astype(int))

    # Saving one (or many) images (TEMPORAL ENCODING)
    if False and temporal_encoding:
        # interval = slice(0, 1)
        # interval = slice(0, 3)
        # interval = slice(0, 100)
        interval = slice(0, 10000)
        # i = np.random.randint(0, x_test.shape[0]-1)
        # interval = slice(i, i+1)

        # cuts = [.01, 0.31, 0.42, 0.56, 0.75]
        # cuts = [.01, 0.2, 0.4, 0.6]
        cuts = [.01, 0.2, 0.3, 0.4, 0.5]
        # cuts = [.5]  # This should coincide with Keras output

        spikes, times, additional_spikes = img_to_tempencoding(
            x_test[interval], cuts,
            position_trigger_neuron=28 * 28 + 256 + 64 + 100)

        klass = y_test[interval].argmax(axis=1)

        if interval.start == interval.stop + 1:
            path = f"spikified-{dataset}/ffsnn-tempcode/" \
                   f"spikified-images-class={klass[0]}-" \
                   f"grayscale=[{','.join(str(c) for c in cuts)}]"
        # THIS DOESN'T WORK CURRENTLY. Adding up 1/512 + 10000 with single precision
        # floating point numbers becomes (approx.) 10000 + 1/500 :(
        elif interval.start == 0 and interval.stop == 10000:
            path = f"spikified-{dataset}/ffsnn-tempcode/" \
                   f"spikified-images-all-" \
                   f"grayscale=[{','.join(str(c) for c in cuts)}]"
        else:
            path = f"spikified-{dataset}/ffsnn-tempcode/" \
                   f"spikified-images-" \
                   f"interval-{interval.start}-to-{interval.stop - 1}-" \
                   f"grayscale=[{','.join(str(c) for c in cuts)}]"

        save_spikes_for_doryta(spikes, times, path, additional_spikes=additional_spikes)
        save_tags_for_doryta(y_test[interval], path)

        print("Classes of images:", klass)
        if False and (loading_model or training_model):
            for j in range(len(cuts)):
                show_prediction(model, model_intermediate, spikes[j:j+1, :28*28])
