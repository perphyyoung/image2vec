# coding=utf-8

from keras.layers import Conv2D, Dropout
from keras.layers.core import Dense, Flatten
from keras.models import Sequential

from .base import py_conv2d, py_maxpool2d


def alexnet(model_name, input_shape, nb_output):
    print(model_name, input_shape, nb_output)

    model = Sequential()
    model.add(Conv2D(filters=96,
                     kernel_size=11,
                     strides=4,
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_last'))
    model.add(py_maxpool2d())
    model.add(py_conv2d(256, kernel_size=5))
    model.add(py_maxpool2d())
    model.add(py_conv2d(384))
    model.add(py_conv2d(384))
    model.add(py_conv2d(256))
    model.add(py_maxpool2d())
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_output, activation='tanh'))
    return model
