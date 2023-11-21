import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])

v100_20K = np.load(root_path+'/data/numpy_processed_data/no_aug/20K/train.npy',allow_pickle=True)
copy_20K = np.load(root_path+'/data/numpy_processed_data/tri_aug/20K_copy/train.npy',allow_pickle=True)

img_v100 = v100_20K[0]
img_copy = copy_20K[0]
print(len(img_v100))
print(len(img_copy))

for item1,item2 in zip(img_v100[::-1],img_copy[::-1]):
    img1 = Image.fromarray(item1).convert('L')
    img2 = Image.fromarray(item2).convert('L')
    plt.imshow(img1,cmap='gray')
    plt.show()
    plt.imshow(img2,cmap='gray')
    plt.show()