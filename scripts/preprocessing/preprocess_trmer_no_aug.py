import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
from PIL import Image
import torchvision.transforms as transforms

import torch
import argparse
import sys
import os
import cv2


cur_path = os.getcwd()
root_path = '/'.join(cur_path.split('/')[:-2])
sys.path.append(root_path)
warnings.filterwarnings('ignore')
which_sets = ['train','val','test']

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
def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')


    parser.add_argument('--formulas', dest='formulas_file_path',
                        default=root_path+'/data/98K/annotations/formulas.txt',
                        type=str,
                        help= 'Input formulas.txt path')

    parser.add_argument('--index', dest='txt_file',
                        default=root_path + '/data/98K/annotations/' + which_set + '.txt',
                        type=str,
                        help='Input one specific index.txt path')

    parser.add_argument('--vocab', dest='vocab_file_path',
                        default=root_path + '/data/98K/annotations/latex_vocab.txt',
                        type=str,
                        help='Input latex_vocab.txt path')

    parser.add_argument('--img', dest='image_path',
                        default=root_path + '/data/98K/image/no_aug/',
                        type=str,
                        help='Input image path')
    parameters = parser.parse_args()
    return parameters


#整个程序的逻辑是有问题的，存在label和image根本不一一对应的问题，
#我将所有的formula排列成一个list，但是这个list的下标无法索引到对应的图片
if __name__ == '__main__':
    for which_set in which_sets:
        parameters = process_args()
        f_formula = open(parameters.formulas_file_path, encoding='utf-8').readlines()
        labelIndexDic = {}
        for item_f in f_formula: #remove the start and end space and get the GT
            labelIndexDic[item_f.strip().split('\t')[0]] = item_f.strip().split('\t')[1]
        # print(labelIndexDic['10247'])
        f_list = open(parameters.txt_file,
                            encoding='utf-8').readlines()#train image and formula index

        labellist = []
        for item_f_list in f_list:
            if len(item_f_list) > 0:
                labellist.append(labelIndexDic[item_f_list.strip()])

        #这里的MAXLENGTH不起任何作用，所以edsl设置这个干嘛呢,也是一样的冗余
        MAXLENGTH = 150
        f_vocab = open(parameters.vocab_file_path, encoding='utf-8').readlines()

        PAD = 0
        START = 1
        END = 2
        index_label_dic = {}
        label_index_dic = {}

        i = 3
        for item_f_vocab in f_vocab:
            word = item_f_vocab.strip()
            if len(word) > 0:
                label_index_dic[word] = i
                index_label_dic[i] = word
                i += 1
        label_index_dic['unk'] = i
        index_label_dic[i] = 'unk'
        i += 1

        labelEmbed_teaching = [] #Have 1'start' in the begining

        labelEmbed_predict = []  #have 2 end in the end

        for item_l in labellist:
            tmp = [1]
            words = item_l.strip().split(' ')
            for item_w in words:
                if len(item_w) > 0:
                    if item_w in label_index_dic:
                        tmp.append(label_index_dic[item_w])
                    else:
                        tmp.append(label_index_dic['unk'])

            labelEmbed_teaching.append(tmp)

            tmp = []
            words = item_l.strip().split(' ')
            for item_w in words:
                if len(item_w) > 0:
                    if item_w in label_index_dic:
                        tmp.append(label_index_dic[item_w])
                    else:
                        tmp.append(label_index_dic['unk'])
            tmp.append(2)
            labelEmbed_predict.append(tmp)

        Label_len_list = [len(item) for item in labelEmbed_teaching]

        #根据长度进行索引
        labelEmbed_order_by_len = np.argsort(np.array(Label_len_list)).tolist()
        labelEmbed_teaching_ordered_array = np.array(labelEmbed_teaching)[labelEmbed_order_by_len]
        labelEmbed_predict_ordered_array = np.array(labelEmbed_predict)[labelEmbed_order_by_len]

        imgList = []
        all_data = []
        for i in tqdm(range(len(labelEmbed_order_by_len))):
            item_f_list = labelEmbed_order_by_len[i] #Fixing index incompleteness

            cv2_img = cv2.imread(parameters.image_path +
                             f_list[item_f_list].strip() + '.png',cv2.IMREAD_GRAYSCALE)

            Image_img = np.array(Image.open(parameters.image_path +
                             f_list[item_f_list].strip() + '.png').convert('L'))

            #print((cv2_img==Image_img).all())
            imgList.append(cv2_img)

            #img_np_array = delete_Padding(img_np_array)
            #plt.imshow(img_np_array,cmap='gray')
            #plt.show()
            #将scaled img转化为Img对象并进行resize
            #tmp_img = Image.fromarray(img_np_array)
            #resize = transforms.Resize([112, 224])
            #resized_img = resize(tmp_img)
            #img_array = np.array(resized_img)
            #imgList.append(img_array)

        all_data = (imgList, labelEmbed_teaching_ordered_array, labelEmbed_predict_ordered_array)

        out_file_path = os.path.join(root_path,'data/98K/numpy/no_aug/' + which_set + '.npy')
        np.save(out_file_path, np.array(all_data))
        print('Save {} dataset to {}, the length is {}'.format(which_set, out_file_path,len(all_data[0])))

