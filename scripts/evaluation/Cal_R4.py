from tqdm import tqdm

import argparse

import sys
import os

from loguru import logger

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


def formula_to_index(formula, dic):
    tmp = []
    for word in formula.strip().split(' '):
        if len(word.strip()) > 0:
            if word.strip() in dic:
                tmp.append(dic[word.strip()])
            else:
                tmp.append(dic['UNK'])
    return tmp


def getRouge_N(predict, label, n):
    predictCanList = []
    labelCanList = []

    for i in range(len(predict)):
        tmp = predict[i:i + n]
        if len(tmp) == n:
            predictCanList.append(tmp)

    for i in range(len(label)):
        tmp = label[i:i + n]
        if len(tmp) == n:
            labelCanList.append(tmp)

    len_labenCanList = len(labelCanList)

    if len_labenCanList == 0:
        return None
    else:
        countList = []

        while len(predictCanList) > 0:
            try:
                index = labelCanList.index(predictCanList[0])
                countList.append(predictCanList[0])
                del labelCanList[index]
            except:
                pass

            del predictCanList[0]

        rouge_n = len(countList) / len_labenCanList
        return rouge_n





if __name__ == '__main__':

    logger.add(root_path+'/CTNN_output/tri_aug/small/20K/R4.txt')
    f2 = open(root_path + '/CTNN_output/tri_aug/small/20K/result.txt', encoding='utf-8').readlines()
    logger.info('load result.txt from {}',root_path+'/CTNN_output/tri_aug/small/20K/result.txt')

    roughList = []

    for item in tqdm(f2):
        predict_label = item.split('     ')
        predict = predict_label[0].strip()[8:]
        label = predict_label[1].strip()[6:]

        rougeN = getRouge_N(predict, label, 4)

        if rougeN == None:
            logger.info('{} R4:None',predict_label)
            pass
        else:
            roughList.append(rougeN)
            logger.info('{} R4:{}',predict_label,rougeN)

    logger.info('total valid number:{}',len(roughList))
    logger.info('ROUGE-4:{}',sum(roughList)/len(roughList))
