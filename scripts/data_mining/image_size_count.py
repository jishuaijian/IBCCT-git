import os
import cv2
import numpy as np

root_path = '/'.join(os.path.abspath(__file__).split('/')[:-3])
numpy_path = os.path.join(root_path,'data/98K/numpy/no_aug/train.npy')

numpy_data = np.load(numpy_path,allow_pickle=True)
image_list,teaching_label_list,predict_label_list = numpy_data[0],numpy_data[1],numpy_data[2]

image_shape_dic = {}
for image,teaching_label,predict_label in zip(image_list,teaching_label_list,predict_label_list):
    (h,w) = image.shape
    if (h,w) in image_shape_dic:
        image_shape_dic[(h,w)] += 1
    else:
        image_shape_dic[(h,w)] = 1

image_shape_dic = sorted(image_shape_dic.items(),key=lambda x:x[1],reverse=True)

print(image_shape_dic)
