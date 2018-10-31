import random

import h5py
import keras.backend as K
import numpy as np
from keras.models import load_model

import cv2

model = load_model('data/model/alexnet_30.h5')

out_file = open('data/output/output.txt', 'w')
file = h5py.File('data/hdf5/data_filtered_30.hdf5')
images_test = list()
image_size = 224
root_image_path = 'data/origin30/flickr30k-images/'

for k, _ in file['test'].items:
    image_path = root_image_path + k
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    images_test.append(image)

len_images = len(images_test)
images_test_array = np.array(images_test)
lst_vector = list()
print('predict')
for i in range(0, len_images + 1):
    vector = model.predict(images_test_array[i], batch_size=1)
    lst_vector.append(vector)

print('compare')
similarity_threshold = 0.95


def cosine_distance(vec1, vec2):
    vec1 = K.l2_normalize(vec1, axis=-1)
    vec2 = K.l2_normalize(vec2, axis=-1)
    return K.mean(1 - K.sum((vec1 * vec2), axis=-1))


random_int = random.randint(0, len_images)
out_file.write(lst_vector[random_int] + '\n')
sample_vector = lst_vector[random_int]
for i in range(0, len_images + 1):
    if cosine_distance(sample_vector, lst_vector[i]) > similarity_threshold:
        out_file.write(lst_vector[i] + '\n')

out_file.close()
