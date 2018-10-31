import numpy as np
from gensim.models import Word2Vec

words = 'A large bird stand in the water on a beach'
lst_vec = list()
model = Word2Vec.load('data/model/word2vec_128.model')
for word in words.lower().split():
    if word in model.wv.vocab:
        lst_vec.append(model.wv[word])

print(len(lst_vec))
print(np.mean(lst_vec, axis=0))
