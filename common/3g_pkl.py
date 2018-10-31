# coding=utf-8
"""Generate the pickle file of 1b_corpus dataset"""

import os
import pickle
from multiprocessing import Process, Queue
from Queue import Empty  # Exception raised when a Queue is empty
from time import ctime

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wt


def process_data(pid, qs, path_in, path_out):
    # size_print = 10  # print size for test
    while not qs.empty():
        try:
            fname = qs.get()
            print('P%s start %s at %s' % (str(pid), fname, ctime()))
        except Empty:  # seems never reach here
            print('Finish at %s' % ctime())
            break

        fpath = os.path.join(path_in, fname)
        # tokenize and lower the case
        with open(fpath, 'r') as f:
            words_tok = [[word.lower() for word in wt(sent.decode('utf-8'))]
                         for sent in f]
        # print(words_tok[:size_print])

        # remove punctuation and other non-alphabet character
        words_alphabet = [[word for word in words if word.isalpha()]
                          for words in words_tok]
        # print(words_alphabet[:size_print])

        # remove stop words
        english_stopwords = stopwords.words('english')
        words_filtered = [[
            word for word in words if word not in english_stopwords
        ] for words in words_alphabet]
        # print(words_filtered[:size_print])

        with open(os.path.join(path_out, '%s.pkl' % fname), 'wb') as f:
            pickle.dump(words_filtered, f)
        print('P%s done %s at %s' % (str(pid), fname, ctime()))


def provide_data():
    path_in = '/data/home/py/data/1b-corpus/trainset'
    path_out = './pkl/'
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    queue_size = 100
    qs = Queue(queue_size)

    for fname in os.listdir(path_in):
        qs.put(fname)

    size_process = 3
    ps = [
        Process(target=process_data, args=(pid, qs, path_in, path_out))
        for pid in range(size_process)
    ]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == '__main__':
    provide_data()
    print('Finish...')
