# coding=utf-8

from keras.applications.densenet import DenseNet121
from keras.layers.core import Dense
from keras.models import Model


def densenet121(model_name, input_shape, nb_output):
    print(model_name, input_shape, nb_output)

    WEIGHTS_PATH = '/data/home/py/weight/' + 'densenet121_top.h5'
    base_model = DenseNet121(weights=WEIGHTS_PATH)
    x = base_model.output
    y = Dense(nb_output, activation='tanh')(x)
    model = Model(inputs=base_model.input, outputs=y)
    return model
