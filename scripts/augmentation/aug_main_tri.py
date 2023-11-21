
# -*- coding:utf-8 -*-
import os

import cv2
import imageio
from augment import distort, stretch, perspective

abs_path = os.path.abspath(__file__)
root_path = ('/').join(abs_path.split('/')[:-3])

def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

if __name__ == '__main__':

    #the param to aug
    segment = 4

    input_path = os.path.join(root_path,'data/98K/image/no_aug')
    output_path = os.path.join(root_path,'data/98k/image/tri_aug')
    images_list = os.listdir(input_path)
    #images_processed = os.listdir('D:\Project\gongshishibie\Text-Image-Augmentation-python\\20K\images_augmentation')
    print(len(images_list))
    for image_name in images_list:
        index = image_name.split('.')[0].strip()

        distort_name = index + 'd.png'
        stretch_name = index + 's.png'
        perspective_name = index + 'p.png'

        image_file = cv2.imread(os.path.join(input_path,image_name),cv2.IMREAD_GRAYSCALE)

        distort_img = distort(image_file, segment)
        stretch_img = stretch(image_file, segment)
        perspective_img = perspective(image_file)

        cv2.imwrite(os.path.join(output_path,image_name), image_file)
        cv2.imwrite(os.path.join(output_path,distort_name), distort_img)
        cv2.imwrite(os.path.join(output_path,stretch_name), stretch_img)
        cv2.imwrite(os.path.join(output_path,perspective_name), perspective_img)
    print('save {} images to {}'.format(len(images_list)*4,output_path))
