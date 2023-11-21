import os
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose


#要明确均值和方差的计算是以像素为单位还是以图片为单位，思路捋清楚其实就很容易了，以像素为单位那就将所有像素值加在一起，然后处以像素数
#方差也是同样的计算思路，但是要注意到底是以什么为单位。
root_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
numpy_path = os.path.join(root_path,'data/20K/numpy/no_aug/train.npy')

numpy_data = np.load(numpy_path,allow_pickle=True)
image_list,teaching_label_list,predict_label_list = list(numpy_data[0]),numpy_data[1],numpy_data[2]

mean_list = []
var_list = []
for image in image_list:
    mean_tmp = np.mean(image)
    mean_list.append(mean_tmp)

mean_dataset = np.mean(mean_list)/255.0

for image in image_list:
    var_tmp = np.mean(np.power(image-mean_dataset,2))
    var_list.append(var_tmp)

sqrt_dataset = np.sqrt(np.mean(var_list))/255.0
print(sqrt_dataset)




