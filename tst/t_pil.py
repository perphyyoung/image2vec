import os
from glob import glob

import numpy as np
from PIL import Image

root_path = './file/'
lst_arr = list()

for path_file in glob(os.path.join(root_path, '*.jpg')):
    print(path_file)
    image = Image.open(path_file)
    image = image.resize((224, 224), Image.ANTIALIAS)
    arr = np.array(image)
    print(np.shape(arr))
    lst_arr.append(arr)

arr = np.array(lst_arr)
print(np.shape(arr))
