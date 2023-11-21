import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

abs_path = os.path.abspath(__file__)
root_path = ('/').join(abs_path.split('/')[:-3])

def delete_Padding(imgNp):
    imgNp[imgNp > 220] == 255
    binary = imgNp != 255
    sum_0 = np.sum(binary, axis=0)
    sum_1 = np.sum(binary, axis=1)
    sum_0_left = min(np.argwhere(sum_0 != 0).tolist())[0]
    sum_0_right = max(np.argwhere(sum_0 != 0).tolist())[0]
    sum_1_left = min(np.argwhere(sum_1 != 0).tolist())[0]
    sum_1_right = max(np.argwhere(sum_1 != 0).tolist())[0]
    # delImg = imgNp[sum_0_left:sum_0_right,sum_1_left:sum_1_right]
    delImg = imgNp[sum_1_left:sum_1_right+1,sum_0_left:sum_0_right+1]
    delImg = np.pad(delImg, ((10,10),(20,20)), 'constant', constant_values=(255,255))
    return delImg

if __name__ =='__main__':
    input_path = root_path+ '/data/image_processed_data/no_aug/20K/images_processed/'
    output_path = root_path + '/data/image_processed_data/no_aug/20K/images_deleted/'

    img_list = os.listdir(input_path)
    print(len(img_list))
    for img_item in img_list:
        im = Image.open(input_path+img_item).convert('L')
        img_array = np.array(im)
        img_array = delete_Padding(img_array)

        im = Image.fromarray(img_array)

        w = im.size[0]
        h = im.size[1]
        if h / w < 0.5:
            pad_len = (w // 2 - h) // 2
            im_pad = np.pad(im, ((pad_len, pad_len), (0, 0)), 'constant', constant_values=(255, 255))
        else:
            if h / w > 0.5:
                pad_len = (h * 2 - w) // 2
                im_pad = np.pad(im, ((0, 0), (pad_len, pad_len)), 'constant', constant_values=(255, 255))

        #注意这里不进行resize
        cv2.imwrite(output_path+ img_item,im_pad)
    print('save {} deleted img to {}'.format(len(img_list),output_path))

