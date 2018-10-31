# coding=utf-8
"""Sort the names of captions and token (train, validation, test, token)"""

import os


def sort_file(path_in, path_out, ind):
    lst = list()
    lst_in_names = [
        'Flickr_8k.trainImages.txt', 'Flickr_8k.devImages.txt',
        'Flickr_8k.testImages.txt', 'Flickr8k.token.txt'
    ]
    lst_out_names = [
        'sorted_train_8.csv', 'sorted_val_8.csv', 'sorted_test_8.csv',
        'sorted_token_8.csv'
    ]

    in_file_name = os.path.join(path_in, lst_in_names[ind])
    with open(in_file_name, 'r') as f:
        for line in f:
            lst.append(line.strip())

    lst.sort(reverse=False)

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    out_file_name = os.path.join(path_out, lst_out_names[ind])
    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    with open(out_file_name, 'w') as f:
        for line in lst:
            f.write(line + '\n')


if __name__ == '__main__':
    path_in = '/data/home/py/data/flickr8k/captions/'
    path_out = './csv/'

    for i in range(4):
        sort_file(path_in, path_out, i)
