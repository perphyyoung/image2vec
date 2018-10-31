# coding=utf-8
"""Generate pairs of image and caption"""

import os
from time import ctime

import h5py


def generate_pair(path_in, path_out):
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    out_file_name = os.path.join(path_out, 'pair_8.hdf5')
    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    out_file = h5py.File(out_file_name, 'w')
    group_train = out_file.create_group('train')
    group_test = out_file.create_group('test')

    lst_test = list()

    lst_in_names = ['sorted_test_8.csv', 'sorted_token_8.csv']

    print('Load test item at %s' % ctime())
    with open(os.path.join(path_in, lst_in_names[0]), 'r', encoding='utf-8') as f:
        for line in f:
            lst_test.append(line.strip())

    print('Load token at %s' % ctime())
    with open(os.path.join(path_in, lst_in_names[1]), 'r', encoding='utf-8') as f:
        for line1 in f:
            line2 = f.readline()
            line3 = f.readline()
            line4 = f.readline()
            line5 = f.readline()

            lst_line = [line1, line2, line3, line4, line5]
            lst_cap = list()  # list of 5 lines
            for line in lst_line:
                name, caption = line.split(maxsplit=1)
                str_jpg, _ = name.split('#')
                lst_cap.append(caption.strip())

            str_cap = ' '.join(lst_cap)
            if str_jpg in lst_test:
                group_test.create_dataset(str_jpg, data=str_cap)
            else:  # combine train set and validation set
                group_train.create_dataset(str_jpg, data=str_cap)

    out_file.close()
    print('Finish at %s' % ctime())


if __name__ == '__main__':
    path_in = './csv/'
    path_out = './hdf5/'
    generate_pair(path_in, path_out)
