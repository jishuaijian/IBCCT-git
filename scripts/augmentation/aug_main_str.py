
# -*- coding:utf-8 -*-
import os

import cv2
from tqdm import tqdm
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
    input_path = root_path + '/data/image_processed_data/no_aug/98K/images_processed/'
    output_path = root_path + '/data/image_processed_data/str_aug/98K/images_processed/'
    images_list = os.listdir(input_path)
    #images_processed = os.listdir('D:\Project\gongshishibie\Text-Image-Augmentation-python\\98K\images_augmentation')
    print(len(images_list))
    for i in tqdm(images_list):
        index = i.split('.')[0].strip()

        #cv2.destroyAllWindows()


        im = cv2.imread(input_path+i)
        #origin_list.append(im)
        #im = cv2.resize(im, (200, 64))
        #cv2.imshow("im_CV", im)

        cv2.imwrite(output_path + index + '.png', im)
        for i in range(4):
            #
            distort_img = distort(im, 4)
            #distort_list.append(distort_img)
            #cv2.imshow("distort_img", distort_img)

            stretch_img = stretch(im, 4)
            #cv2.imshow("stretch_img", stretch_img)
            #stretch_list.append(stretch_img)

            perspective_img = perspective(im)
            #cv2.imshow("perspective_img", perspective_img)
            #perspective_list.append(perspective_img)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            #root = 'D:\Project\gongshishibie\Text-Image-Augmentation-python\\98K\images_augmentation/'
            cv2.imwrite(output_path+index+'d_'+str(i)+'.png',distort_img)
            cv2.imwrite(output_path+index+'s_'+str(i)+'.png',stretch_img)
            cv2.imwrite(output_path+index+'p_'+str(i)+'.png',perspective_img)
    print('save {} imges to {}'.format(len(images_list)*13,output_path))
