# coding=utf-8

import os

from PIL import Image


def main():
    rootpath_image = '/data/home/py/data/corel1000/'
    outpath = './corel1000_224/'
    image_size = 224
    for ind, fname in enumerate(os.listdir(rootpath_image)):
        if ind % 100 == 0:
            print(ind)
        fpath = os.path.join(rootpath_image, fname)
        image = Image.open(fpath, 'r')
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        image.save(os.path.join(outpath, '%s.png' % fname))


if __name__ == '__main__':
    main()
