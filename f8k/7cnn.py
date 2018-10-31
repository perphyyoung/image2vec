# coding=utf-8
"""Train CNNs to generate image representation"""

import gc
import os
from time import ctime

import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split as tts

import sys
sys.path.insert(0, '/data/home/py/project/image')
# sys.path.insert(0, 'c:\\workspace\\pycharm\\image1_final')
from util.alexnet import alexnet  # noqa
from util.zfnet import zfnet  # noqa
from util.resnet50 import resnet50  # noqa
from util.vgg16 import vgg16  # noqa
from util.densenet121 import densenet121  # noqa


def train_and_evaluate(print_summary=False):
    image_size = 224
    image_channels = 3
    input_shape = (image_size, image_size, image_channels)
    nb_output = 300

    # 0:alexnet, 1:zf-net, 2:resnet50, 3:vgg16, 4:densenet121
    index_model = 0
    lr = 1e-3  # learning rate of optimizer

    lst_model_names = [
        'alex_8_', 'zf_8_', 'res_8_', 'vgg_8_', 'densenet_8_'
    ]
    lst_funcs = [
        'alexnet', 'zfnet', 'resnet50', 'vgg16', 'densenet121'
    ]

    model_name = lst_model_names[index_model] + str(lr)
    model = eval(lst_funcs[index_model])(model_name, input_shape, nb_output)

    # densenet: freeze all but the last layer to train
    if index_model == 5:
        for layer in model.layers[:-2]:
            layer.trainable = False
        for layer in model.layers[-2:]:
            layer.trainable = True

    if print_summary:
        model.summary()
        sys.exit()

    print('Compile at %s' % ctime())
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='mse')

    path_npz = './npz/'
    lst_npz_name = ['train_8.npz', 'test_8.npz']

    # train item
    npz_file = np.load(os.path.join(path_npz, lst_npz_name[0]))
    a_image = npz_file['image']
    a_caption = npz_file['caption']
    print('Length of train', np.shape(a_image), np.shape(a_caption), ctime())

    test_size = 0.1
    print('Shuffle at %s' % ctime())
    x_train, x_val, y_train, y_val = tts(
        a_image, a_caption, test_size=test_size, random_state=32)
    print('Done shuffling at %s' % ctime())
    print('train', np.shape(x_train), 'val', np.shape(x_val))

    del a_image
    del a_caption
    gc.collect()

    path_out = './result/'
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    csv_logger = CSVLogger(
        '%s.csv' % os.path.join(path_out, model_name),
        separator=',',
        append=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print("Training model at %s" % ctime())
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=10,
        validation_data=(x_val, y_val),
        callbacks=[csv_logger, early_stopping])
    print('Done training model at %s' % ctime())

    del x_train
    del x_val
    del y_train
    del y_val
    gc.collect()

    model.save('%s.h5' % os.path.join(path_out, model_name))

    # test item
    npz_file = np.load(os.path.join(path_npz, lst_npz_name[1]))
    a_image = npz_file['image']
    a_caption = npz_file['caption']
    print('Length of test', np.shape(a_image), np.shape(a_caption), ctime())

    print('Testing model at %s' % ctime())
    score = model.evaluate(a_image, a_caption, batch_size=50)
    print(score, ctime())

    del a_image
    del a_caption
    gc.collect()

    with open('%s.csv' % os.path.join(path_out, model_name), 'a') as f:
        f.write('\n score: %s' % str(score))

    print('Finish at %s' % ctime())


if __name__ == '__main__':
    print_summary = False
    train_and_evaluate(print_summary)
