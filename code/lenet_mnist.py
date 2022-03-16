from __future__ import annotations

from typing import Any, Tuple, Union, Optional
import pathlib
import os
# import sys
import numpy as np
# import struct

# import matplotlib.pyplot as plt

# import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras import Model

from whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from whetstone.callbacks import SimpleSharpener, WhetstoneLogger

from utils.common_mnist import my_key, load_data
from utils.doryta.model_saver import ModelSaverLayers
from utils.temp_encoding import img_to_tempencoding
from utils.doryta.spikes import save_spikes_for_doryta
from ffsnn_mnist import save_tags_for_doryta
from utils.common import keras_model_path, doryta_model_path


def create_model(initializer: Any = 'glorot_uniform',
                 use_my_key: bool = True,
                 filters: Optional[Tuple[int, int]] = None) -> Tuple[Any, Any]:
    if filters is None:
        # These are the number of filters from the kaggle implementation. The original
        # LeNet used (6, 16) instead
        filters = (32, 48)

    if use_my_key:
        key = my_key()
    else:
        key = key_generator(num_classes=10, width=100)

    # Original implementation from:
    # https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48/notebook
    model = Sequential()
    # stride = 2
    model.add(Conv2D(filters=filters[0], kernel_size=(5, 5),
                     padding='same', input_shape=(28, 28, 1)))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(MaxPool2D(strides=2))
    # stride = 0
    model.add(Conv2D(filters=filters[1], kernel_size=(5, 5), padding='valid'))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(120, kernel_initializer=initializer))
    # model.add(Dense(256, kernel_initializer=initializer))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(84, kernel_initializer=initializer))
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
        outputs=[model.layers[i].output for i in
                 [1, 2, 4, 5, 8, 10, 12]])

    return model, model_intermediate


def load_models(path: Union[str, pathlib.Path]) -> Tuple[Any, Any]:
    model = load_model(path)
    model_intermediate = Model(
        inputs=model.inputs,
        outputs=[model.layers[i].output for i in
                 [1, 2, 4, 5, 8, 10, 12]])
    return model, model_intermediate


if __name__ == '__main__':  # noqa: C901
    # filters = (32, 48)
    filters = (6, 16)
    # dataset = 'mnist'
    dataset = 'fashion-mnist'
    model_path = pathlib.Path(f'lenet-{dataset}-filters={filters[0]},{filters[1]}')

    # This is super good but produces negative values for the matrix, ie, negative currents :S
    initializer = 'glorot_uniform'
    # initializer = RandomUniform(minval=0.0, maxval=1.0)

    temporal_encoding = True

    loading_model = True
    training_model = False
    checking_model = True
    saving_model = False

    (x_train, y_train), (x_test, y_test) = load_data(dataset)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    if loading_model:
        model, model_intermediate = load_models(keras_model_path / model_path)

    elif training_model:
        # Create a new directory to save the logs in.
        log_dir = str(keras_model_path / 'logs' / model_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        model, model_intermediate = create_model(initializer=initializer, filters=filters)

        # Parameters for shaperner
        start_epoch = 5
        steps = 5
        total_brelus = 5

        simple = SimpleSharpener(start_epoch=start_epoch, steps=steps,
                                 epochs=True, bottom_up=True)  # type: ignore
        logger = WhetstoneLogger(logdir=log_dir, sharpener=simple)  # type: ignore

        epochs = start_epoch + steps * total_brelus + 1

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128,
                  epochs=epochs, callbacks=[simple, logger])
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

        if saving_model:
            if not temporal_encoding:
                msaver = ModelSaverLayers()
                k1, t1 = model.layers[0].get_weights()  # conv2d
                k2, t2 = model.layers[3].get_weights()  # conv2d
                w3, t3 = model.layers[7].get_weights()  # fully
                w4, t4 = model.layers[9].get_weights()  # fully
                w5, t5 = model.layers[11].get_weights()  # fully
                w3 = w3.reshape((5, 5, filters[1], 120)).transpose((2, 0, 1, 3)).reshape((-1, 120))
                msaver.add_conv2d_layer(k1, .5 - t1, (28, 28), padding=(2, 2))
                msaver.add_maxpool_layer((28, 28, filters[0]), (2, 2))
                msaver.add_conv2d_layer(k2, .5 - t2, (14, 14))
                msaver.add_maxpool_layer((10, 10, filters[1]), (2, 2))
                msaver.add_fully_layer(w3, .5 - t3)
                msaver.add_fully_layer(w4, .5 - t4)
                msaver.add_fully_layer(w5, .5 - t5)
                msaver.save(doryta_model_path /
                            f"lenet-{dataset}-filters={filters[0]},{filters[1]}.doryta.bin")
            else:
                msaver = ModelSaverLayers(dt=1/256)
                k1, t1 = model.layers[0].get_weights()  # conv2d
                k2, t2 = model.layers[3].get_weights()  # conv2d
                w3, t3 = model.layers[7].get_weights()  # fully
                w4, t4 = model.layers[9].get_weights()  # fully
                w5, t5 = model.layers[11].get_weights()  # fully
                w3 = w3.reshape((5, 5, filters[1], 120)).transpose((2, 0, 1, 3)).reshape((-1, 120))

                R = 4
                capacitance = 1/256  # = dt
                neuron_args = {
                    'resistance': R,
                    'tau': capacitance * R
                }

                msaver.add_conv2d_layer(
                    k1, .5 - t1 + 10,
                    (28, 28),
                    padding=(2, 2),
                    neuron_args=neuron_args)
                msaver.add_maxpool_layer((28, 28, filters[0]), (2, 2))
                msaver.add_conv2d_layer(k2, .5 - t2, (14, 14))
                msaver.add_maxpool_layer((10, 10, filters[1]), (2, 2))
                msaver.add_fully_layer(w3, .5 - t3)
                msaver.add_fully_layer(w4, .5 - t4)
                msaver.add_fully_layer(w5, .5 - t5)
                msaver.save(doryta_model_path /
                            f"lenet-{dataset}-filters={filters[0]},{filters[1]}.doryta.bin")

                # Adding a neuron that triggers the second layer
                msaver.add_neuron_group(0.5 * np.ones((1,)))
                weights = 10 * np.ones((1, 28 * 28 * filters[0]))
                msaver.add_fully_conn(from_=8, to=1, weights=weights)

                msaver.save(doryta_model_path /
                            f"lenet-{dataset}-tempencode-R={R}-"
                            f"filters={filters[0]},{filters[1]}.doryta.bin")

    # Saving one (or many) images (TEMPORAL ENCODING)
    if False and temporal_encoding:
        # interval = slice(0, 1)
        # interval = slice(0, 3)
        interval = slice(0, 100)

        cuts = [.01, 0.31, 0.42, 0.56, 0.75]
        # cuts = [.01, 0.2, 0.4, 0.6]
        # cuts = [.01, 0.2, 0.3, 0.4, 0.5]
        # cuts = [.5]  # This should coincide with Keras output

        if filters[0] == 32:
            trigger_neuron = 38448
        else:
            trigger_neuron = 8968

        spikes, times, additional_spikes = img_to_tempencoding(
            x_test[interval].reshape((-1, 28 * 28)), cuts,
            position_trigger_neuron=trigger_neuron)

        klass = y_test[interval].argmax(axis=1)

        path = f"spikified-{dataset}/lenet-filters={filters[0]},{filters[1]}-tempcode/" \
               f"spikified-images-" \
               f"interval-{interval.start}-to-{interval.stop - 1}-" \
               f"grayscale=[{','.join(str(c) for c in cuts)}]"

        save_spikes_for_doryta(spikes, times, path, additional_spikes=additional_spikes)
        save_tags_for_doryta(y_test[interval], path)

        print("Classes of images:", klass)

    # Saving all images using (TEMPORAL ENCODING)
    if False and temporal_encoding:
        cuts = [.01, 0.2, 0.3, 0.4, 0.5]
        # cuts = [.5]  # This should coincide with Keras output

        if filters[0] == 32:
            trigger_neuron = 38448
        else:
            trigger_neuron = 8968

        spikes, times, additional_spikes = img_to_tempencoding(
            x_test.reshape((-1, 28 * 28)), cuts,
            position_trigger_neuron=trigger_neuron)

        klass = y_test.argmax(axis=1)

        path = f"spikified-{dataset}/lenet-filters={filters[0]},{filters[1]}-tempcode/" \
               f"spikified-images-all-" \
               f"grayscale=[{','.join(str(c) for c in cuts)}]"

        save_spikes_for_doryta(spikes, times, path, additional_spikes=additional_spikes)
        save_tags_for_doryta(y_test, path)

        print("Classes of images:", klass)
