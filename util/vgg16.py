# coding=utf-8

from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense
from keras.models import Model


def vgg16(model_name, input_shape, nb_output):
    print(model_name, input_shape, nb_output)

    WEIGHTS_PATH = '/data/home/py/weight/' + 'vgg16_top.h5'
    base_model = VGG16(weights=WEIGHTS_PATH)
    x = base_model.output
    y = Dense(nb_output, activation='tanh')(x)
    model = Model(inputs=base_model.input, outputs=y)
    return model
