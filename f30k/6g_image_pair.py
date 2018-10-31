# coding=utf-8
"""Generate pair of image data and caption"""

import gc
import os
from time import ctime

import h5py
import numpy as np
from PIL import Image


def load_image(group_name, pair_file, image_size):
    print('Loading %s items at %s' % (group_name, ctime()))
    root_path_image = '/data/home/py/data/flickr30k/images/'
    lst_image = list()
    lst_caption = list()
    i = 0
    for k, v in pair_file[group_name].items():
        if i % 1000 == 0:
            print(i, ctime())
        i += 1
        path_image = os.path.join(root_path_image, k)
        try:
            image = Image.open(path_image)
            image = image.resize((image_size, image_size), Image.ANTIALIAS)
            lst_image.append(np.array(image))
            lst_caption.append(v.value)
        except IOError as e:  # python2 doesn't have FileNotFoundError
            print(e)

    return np.array(lst_image), np.array(lst_caption)


def main():
    image_size = 224
    path_npz = './npz/'
    if not os.path.exists(path_npz):
        os.mkdir(path_npz)

    lst_group_names = ['train', 'test']
    path_hdf5 = './hdf5/'
    fpath = os.path.join(path_hdf5, 'filtered_30.hdf5')
    pair_file = h5py.File(fpath)
    lst_npz_name = ['train_30.npz', 'test_30.npz']

    # train items
    a_image, a_caption = load_image(lst_group_names[0], pair_file, image_size)
    print('Length of train', np.shape(a_image), np.shape(a_caption), ctime())
    out_path = os.path.join(path_npz, lst_npz_name[0])
    np.savez(out_path, image=a_image, caption=a_caption)
    print('Done %s at %s' % (lst_npz_name[0], ctime()))
    del a_image
    del a_caption
    gc.collect()

    # test items
    a_image, a_caption = load_image(lst_group_names[1], pair_file, image_size)
    print('Length of test', np.shape(a_image), np.shape(a_caption), ctime())
    out_path = os.path.join(path_npz, lst_npz_name[1])
    np.savez(out_path, image=a_image, caption=a_caption)
    print('Done %s at %s' % (lst_npz_name[1], ctime()))
    del a_image
    del a_caption
    gc.collect()

    print('Finish at %s' % ctime())


if __name__ == '__main__':
    main()
