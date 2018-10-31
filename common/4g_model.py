# coding=utf-8
"""
Generate word2vec model of 1b_corpus, time-consuming :(
    Average of Word2Vec vectors
"""

import os
import pickle
from time import ctime

from gensim.models import Word2Vec


def generate_model(path_in, path_out):
    lst_words = list()
    size_model = 300
    model_name = '1b_corpus_%s.model' % size_model
    model_path = os.path.join(path_out, model_name)

    i = 1
    for fname in os.listdir(path_in):
        print(i, fname, ctime())
        i += 1
        with open(os.path.join(path_in, fname), 'rb') as f:
            lst_words.extend(pickle.load(f))

    print('Begin at %s' % ctime())
    model = Word2Vec(
        lst_words, size=size_model, window=10, min_count=10, workers=5)

    print('Save at %s' % ctime())
    model.save(model_path)
    print('Finish at %s' % ctime())

    print(model.wv['see'])  # test


if __name__ == '__main__':
    path_in = './pkl/'
    path_out = './model/'
    generate_model(path_in, path_out)
