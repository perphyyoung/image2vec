# coding=utf-8
"""Generate filtered pair of file name and vector"""

import os
from time import ctime

import h5py
import numpy as np
from gensim.models import Word2Vec


def gen_vec(path_hdf5, path_model):
    fin = h5py.File(os.path.join(path_hdf5, 'pair_30.hdf5'), 'r')
    fout = h5py.File(os.path.join(path_hdf5, 'filtered_30.hdf5'), 'r')
    model = Word2Vec.load(os.path.join(path_model, '1b_corpus_300.model'))

    lst_group_name = ['train', 'test']
    for group_name in lst_group_name:
        print(group_name, ctime())
        group = fout.create_group(group_name)

        for k, v in fin[group_name].items():
            words = v.value.lower().split()
            lst_vec = [
                model.wv[word] for word in words if word in model.wv.vocab
            ]
            vec = np.mean(lst_vec, axis=0)
            group.create_dataset(k, data=vec)

    fin.close()
    fout.close()
    print('Done at %s' % ctime())


if __name__ == '__main__':
    path_hdf5 = './hdf5/'
    path_model = '../model/'
    gen_vec(path_hdf5, path_model)
