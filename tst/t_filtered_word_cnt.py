# coding=utf-8
"""Generate filtered pair of file name and vector"""

import os
from time import ctime

import h5py
import collections


def main(path_hdf5):
    fin = h5py.File(os.path.join(path_hdf5, 'pair_8.hdf5'), 'r')
    fout = open(os.path.join(path_hdf5, 'cnt.csv'), 'w')

    group_name = 'test'
    print(group_name, ctime())

    dic = dict()
    for k, v in fin[group_name].items():
        line = v.value.lower().split()
        dic = dict(collections.Counter(line).most_common(10))

        # Sort the result
        sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for k, v in dic.items():
            fout.write(k + ':' + str(v) + ',')
        fout.write('\n')

    fin.close()
    fout.close()
    print('Done at %s' % ctime())


if __name__ == '__main__':
    path_hdf5 = './hdf5/'
    main(path_hdf5)
