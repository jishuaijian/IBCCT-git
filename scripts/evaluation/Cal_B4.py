from nltk.translate.bleu_score import sentence_bleu
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





if __name__ == '__main__':

    logger.add(root_path + '/CTNN_output/tri_aug/small/20K/B4.txt')
    f2 = open(root_path + '/CTNN_output/tri_aug/small/20K/result.txt', encoding='utf-8').readlines()
    logger.info('load result.txt from {}',root_path+'/CTNN_output/tri_aug/small/20K/result.txt')

    bleuList = []
    for item in tqdm(f2):
        predict_label = item.split('     ')
        predict = predict_label[0].strip()[8:]
        label = predict_label[1].strip()[6:]

        if len(label) >= 4:
            tmpBleu1 = 0.
            if len(predict) < 4:
                bleuList.append(tmpBleu1)
            else:
                tmpBleu1 = sentence_bleu([label], predict, weights=(0, 0, 0, 1))
                bleuList.append(tmpBleu1)
            logger.info('{} current bleu-4:{:.2}',predict_label,tmpBleu1)

    logger.info('total valid number:{}',len(bleuList))
    logger.info('BLEU-4:{}',sum(bleuList)/len(bleuList))
