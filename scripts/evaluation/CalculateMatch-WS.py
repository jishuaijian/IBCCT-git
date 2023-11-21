import subprocess

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np

import argparse
from utils import ImgCandidate

import sys
import os

from loguru import logger

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)



if __name__ == '__main__':

    logger.add(root_path + '/CTNN_output/tri_aug/small/20K/Match-ws.txt')
    f = open(root_path + '/CTNN_output/tri_aug/small/20K/result.txt').readlines()
    logger.info('load result.txt from {}', root_path + '/CTNN_output/tri_aug/small/20K/result.txt')

    accList = []
    index = 0
    for item in tqdm(f):
        predict_label = item.split('     ')
        predict = predict_label[0].strip()[8:]
        label = predict_label[1].strip()[6:]

        if predict == label:
            accList.append(1)
        else:

            pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                      r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + predict + \
                      r'\end{equation*}' + '\n' + '\end{document}'
            f3 = open('predict.tex', mode='w')
            f3.write(pdfText)
            f3.close()
            sub = subprocess.Popen("pdflatex -halt-on-error " + "predict.tex", shell=True, stdout=subprocess.PIPE)
            sub.wait()

            pdfFiles = []
            for _, _, pf in os.walk(os.getcwd()):
                pdfFiles = pf
                break

            if 'predict.pdf' in pdfFiles:
                try:
                    pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                              r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + label + \
                              r'\end{equation*}' + '\n' + '\end{document}'
                    f3 = open('label.tex', mode='w')
                    f3.write(pdfText)
                    f3.close()
                    sub = subprocess.Popen("pdflatex -halt-on-error " + "label.tex", shell=True, stdout=subprocess.PIPE)
                    sub.wait()

                    os.system(
                        'convert -background white -density 200  -quality 100 -strip ' + 'label.pdf ' + 'label.png')
                    os.system(
                        'convert -background white -density 200  -quality 100 -strip ' + 'predict.pdf ' + 'predict.png')
                    labelImg = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('label.png').convert('L')))).tolist()
                    predictImg = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('predict.png').convert('L')))).tolist()

                    if labelImg == predictImg:
                        accList.append(1)
                    else:
                        logger.info('predict img and label img is not equal')
                        tmpImg = Image.open('predict.png').convert('L')
                        tmpImg.save(root_path+'/CTNN_output/tri_aug/small/20K/match-ws_img/'+str(index) + '_predict'+'.png')
                        tmpImg = Image.open('label.png').convert('L')
                        tmpImg.save(root_path + '/CTNN_output/tri_aug/small/20K/match-ws_img/' +str(index) + '_label' + '.png')
                        accList.append(0)


                except Exception as e:
                    logger.info('catch Exception {}',e)
                    accList.append(0)

            else:
                logger.info('current img is not rendered')
                accList.append(0)

        logger.info('current match-ws:{}',sum(accList)/len(accList))
        os.system('rm -rf *.aux')
        os.system('rm -rf *.log')
        os.system('rm -rf *.tex')
        os.system('rm -rf *.pdf')
        os.system('rm -rf *.png')
        index += 1


    logger.info('Match-ws:{}',sum(accList)/len(accList))
