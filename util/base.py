# coding=utf-8

from keras.layers import Conv2D, MaxPool2D


def py_conv2d(filters,
              kernel_size=3,
              strides=1,
              padding='same',
              activation='relu'):
    """wrapper of Conv2D

    Args: default value
        kernel_size=3
        strides=1
        padding='same'
        activation='relu'
    """
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  activation=activation)


def py_maxpool2d(pool_size=3, strides=2):
    """wrapper of MaxPool2D

    Args: default value
        pool_size=3
        strides=2
    """
    return MaxPool2D(pool_size=pool_size, strides=strides)
