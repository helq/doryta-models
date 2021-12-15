from __future__ import annotations

from typing import Any, Tuple, Union
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
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from .whetstone.layers import Spiking_BRelu, Softmax_Decode, key_generator
from .whetstone.callbacks import SimpleSharpener, WhetstoneLogger

from .common_mnist import my_key
from .utils_doryta.model_saver import ModelSaverLayers


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

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)


def create_model(initializer: Any = 'glorot_uniform',
                 use_my_key: bool = True) -> Tuple[Any, Any]:
    if use_my_key:
        key = my_key()
    else:
        key = key_generator(num_classes=10, width=100)

    # Original implementation from:
    # https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48/notebook
    model = Sequential()
    # stride = 2
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(MaxPool2D(strides=2))
    # stride = 0
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid'))
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


if __name__ == '__main__':
    model_path = pathlib.Path('keras-lecun-mnist')

    # This is super good but produces negative values for the matrix, ie, negative currents :S
    initializer = 'glorot_uniform'
    # initializer = RandomUniform(minval=0.0, maxval=1.0)

    loading_model = True
    train_model = False
    checking_model = False
    save_model = False

    (x_train, y_train), (x_test, y_test) = load_data()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if loading_model:
        model, model_intermediate = load_models(model_path)

    elif train_model:
        # Create a new directory to save the logs in.
        log_dir = './simple_logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        model, model_intermediate = create_model(initializer=initializer)

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
        model.save(model_path)

    if loading_model or train_model:
        if checking_model:
            model.summary()
            print("Evaluating model (loss, accuracy):", model.evaluate(x_test, y_test))
            # new model allows us to extract the results of a layer

            imgs = (x_test > .5).astype(float)  # type: ignore
            prediction = model.predict(imgs).argmax(axis=1)
            correct_predictions = (prediction == y_test.argmax(axis=1)).sum()
            print("Evaluating model (accuracy on black&white images):",
                  correct_predictions / imgs.shape[0])

        if save_model:
            msaver = ModelSaverLayers()
            k1, t1 = model.layers[0].get_weights()  # conv2d
            k2, t2 = model.layers[3].get_weights()  # conv2d
            w3, t3 = model.layers[7].get_weights()  # fully
            w4, t4 = model.layers[9].get_weights()  # fully
            w5, t5 = model.layers[11].get_weights()  # fully
            w3 = w3.reshape((5, 5, 48, 120)).transpose((2, 0, 1, 3)).reshape((-1, 120))
            msaver.add_conv2d_layer(k1, .5 - t1, (28, 28), padding=(2, 2))
            msaver.add_maxpool_layer((28, 28, 32), (2, 2))
            msaver.add_conv2d_layer(k2, .5 - t2, (14, 14))
            msaver.add_maxpool_layer((10, 10, 48), (2, 2))
            msaver.add_fully_layer(w3, .5 - t3)
            msaver.add_fully_layer(w4, .5 - t4)
            msaver.add_fully_layer(w5, .5 - t5)
            msaver.save("lenet-mnist.doryta.bin")
