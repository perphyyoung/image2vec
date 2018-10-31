# coding=utf-8

from keras.layers import Conv2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential

from .base import py_conv2d, py_maxpool2d


def lenet(model_name, input_shape, nb_output):
    print(model_name, input_shape, nb_output)

    model = Sequential()
    model.add(Conv2D(filters=96,
                     kernel_size=11,
                     strides=4,
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(py_maxpool2d())  # 26*26*96
    model.add(py_conv2d(256, kernel_size=5))  # 26*26*256
    model.add(py_maxpool2d())  # 12*12*256
    model.add(Flatten())
    model.add(Dense(nb_output, activation='tanh'))
    return model
