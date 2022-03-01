from __future__ import annotations

from typing import Any, Tuple
import pathlib
import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.initializers import RandomUniform
import tensorflow.keras.constraints as constraints
import tensorflow.keras as keras

from .whetstone.layers import Spiking_BRelu, Softmax_Decode
from .whetstone.callbacks import WhetstoneLogger, AdaptiveSharpener

from .common_mnist import my_key, load_data
from .utils_doryta.model_saver import ModelSaverLayers


def create_model(filters: Tuple[int, int],
                 initializer_stage: int) -> Any:
    assert 0 <= initializer_stage
    key = my_key()

    initializer: Any = lambda x: RandomUniform(minval=0.0, maxval=1.0 / x)  # noqa: E731

    # Original implementation from:
    # https://www.kaggle.com/curiousprogrammer/lenet-5-cnn-with-keras-99-48/notebook
    model = Sequential()
    # stride = 2
    model.add(Conv2D(
        filters=filters[0], kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1),
        kernel_initializer=initializer(28 * 28) if initializer_stage == 0 else None,
        kernel_constraint=constraints.NonNeg()
    ))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(MaxPool2D(strides=2))
    # stride = 0
    model.add(Conv2D(
        filters=filters[1], kernel_size=(5, 5), padding='valid',
        kernel_initializer=initializer(5 * 5 * filters[0]) if initializer_stage == 1 else None,
        kernel_constraint=constraints.NonNeg() if initializer_stage >= 1 else None
    ))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(
        120,
        kernel_initializer=initializer(5 * 5 * filters[1]) if initializer_stage == 2 else None,
        kernel_constraint=constraints.NonNeg() if initializer_stage >= 2 else None
    ))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(
        84,
        kernel_initializer=initializer(120) if initializer_stage == 3 else None,
        kernel_constraint=constraints.NonNeg() if initializer_stage == 3 else None
    ))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Dense(
        100,
        kernel_initializer=initializer(84) if initializer_stage == 3 else None,
        kernel_constraint=constraints.NonNeg() if initializer_stage == 3 else None
    ))
    model.add(Spiking_BRelu())  # type: ignore
    model.add(Softmax_Decode(key))  # type: ignore

    # Disabling training of Softmax Decode layer. I really don't need any of its magic as
    # it obfuscates the inference process (and I would have to carry around the keras
    # model, or the trained key, in order to make sense of the SNN output). This is also
    # the reason why `use_my_key` is True by default
    model.layers[-1].trainable = False

    return model


if __name__ == '__main__':  # noqa: C901
    # filters = (32, 48)
    filters = (6, 16)
    dataset = 'mnist'
    # dataset = 'fashion-mnist'

    # Number of levels to discretize neuron weights (0 inactivates the levels)
    # Only used when saving network
    levels: int = 60

    # Parameters for shaperner
    start_epoch = 5

    loading_model = True
    training_model = False
    checking_model = True
    saving_model = True

    # keras.utils.set_random_seed(2900522)

    path_name = f'keras-lenet-{dataset}-filters={filters[0]},{filters[1]}'
    model_path = pathlib.Path(f'{path_name}-nonnegative')

    (x_train, y_train), (x_test, y_test) = load_data(dataset)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    if loading_model:
        model = load_model(model_path)

    elif training_model:
        # Create a new directory to save the logs in.
        log_dir = str('./logs' / model_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # The following training strategy has been deviced to find a more suitable
        # starting point for training the network. It's complicated because the constraint
        # of nonnegative values change dramatically a good easy initialization
        #
        # First model (constraining only first layer to be nonnegative)
        model = create_model(filters=filters, initializer_stage=0)

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128, epochs=1)

        w1, b1 = model.layers[0].get_weights()

        # Second model (constraining only two first layers)
        keras.backend.clear_session()

        model = create_model(filters=filters, initializer_stage=1)
        model.layers[0].set_weights((w1+0.001, b1+0.001))

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128, epochs=1)

        w2, b2 = model.layers[3].get_weights()

        # Third model (constraining only three first layers)
        keras.backend.clear_session()

        model = create_model(filters=filters, initializer_stage=2)
        model.layers[0].set_weights((w1+0.001, b1+0.001))
        model.layers[3].set_weights((w2+0.001, b2+0.001))

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128, epochs=1)

        w3, b3 = model.layers[7].get_weights()

        # Final model (constraining all layers)
        keras.backend.clear_session()

        model = create_model(filters=filters, initializer_stage=3)
        model.layers[0].set_weights((w1+0.001, b1+0.001))
        model.layers[3].set_weights((w2+0.001, b2+0.001))
        model.layers[7].set_weights((w3, b3))

        adapt = AdaptiveSharpener(  # type: ignore
            min_init_epochs=start_epoch,
            sig_increase=0.15,
            sig_decrease=0.05,
            patience=2,
            cz_rate=0.063,
            verbose=True
        )
        logger = WhetstoneLogger(logdir=log_dir, sharpener=adapt)  # type: ignore

        model.compile(loss='categorical_crossentropy', optimizer=Adadelta(
            learning_rate=4.0, rho=0.95, epsilon=1e-8, decay=0.0), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128,
                  epochs=100, callbacks=[adapt, logger])
        model.save(model_path)

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
            msaver = ModelSaverLayers(dt=1/256)
            k1, b1 = model.layers[0].get_weights()  # conv2d
            k2, b2 = model.layers[3].get_weights()  # conv2d
            w3, b3 = model.layers[7].get_weights()  # fully
            w4, b4 = model.layers[9].get_weights()  # fully
            w5, b5 = model.layers[11].get_weights()  # fully
            w3 = w3.reshape((5, 5, filters[1], 120)).transpose((2, 0, 1, 3)).reshape((-1, 120))

            R = 4
            capacitance = 1/256  # = dt
            neuron_args = {
                'resistance': R,
                'tau': capacitance * R
            }

            t1 = .5 - b1
            t2 = .5 - b2
            t3 = .5 - b3
            t4 = .5 - b4
            t5 = .5 - b5

            if levels > 0:
                k1 = (k1 * (levels / t1).reshape((1, 1, 1, -1))).astype(int)
                k2 = (k2 * (levels / t2).reshape((1, 1, 1, -1))).astype(int)
                w3 = (w3 * (levels / t3).reshape((1, -1))).astype(int)
                w4 = (w4 * (levels / t4).reshape((1, -1))).astype(int)
                w5 = (w5 * (levels / t5).reshape((1, -1))).astype(int)
                t1[t1 > 0] = levels
                t2[t2 > 0] = levels
                t3[t3 > 0] = levels
                t4[t4 > 0] = levels
                t5[t5 > 0] = levels

                weight_shift = 10 * levels
            else:
                weight_shift = 10

            t1 = t1 + weight_shift

            msaver.add_conv2d_layer(k1, t1, (28, 28), padding=(2, 2),
                                    neuron_args=neuron_args)
            msaver.add_maxpool_layer((28, 28, filters[0]), (2, 2))
            msaver.add_conv2d_layer(k2, t2, (14, 14))
            msaver.add_maxpool_layer((10, 10, filters[1]), (2, 2))
            msaver.add_fully_layer(w3, t3)
            msaver.add_fully_layer(w4, t4)
            msaver.add_fully_layer(w5, t5)

            # Adding a neuron that triggers the second layer
            msaver.add_neuron_group(0.5 * np.ones((1,)))
            weights = weight_shift * np.ones((1, 28 * 28 * filters[0]))
            msaver.add_fully_conn(from_=8, to=1, weights=weights)

            basename = f"lenet-{dataset}-tempencode-R={R}-" \
                f"filters={filters[0]},{filters[1]}"

            if levels > 0:
                msaver.save(f"{basename}-nonnegative-lvls={levels}.doryta.bin")
            else:
                msaver.save(f"{basename}-nonnegative.doryta.bin")
