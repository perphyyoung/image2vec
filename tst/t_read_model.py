import os

from gensim.models import Word2Vec

path_model = '../model/'
model = Word2Vec.load(os.path.join(path_model, '1b_corpus_300.model'))
print(model.wv['see'])
