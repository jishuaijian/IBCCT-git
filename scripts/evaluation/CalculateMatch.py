import subprocess
from PIL import Image
from tqdm import tqdm
from utils import ImgCandidate
import argparse
import numpy as np
import sys
import os

from loguru import logger

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


if __name__ == '__main__':

    f = open(root_path + '/result_two_model/result_98K_new.txt').readlines()

    accList = []
    index = 0
    for item in tqdm(f):
        formulas = item.strip().split('\t')
        label = formulas[0]
        cct = formulas[1]
        edsl  = formulas[2]

        if label==cct==edsl:
            accList.append(1)
        else:
            pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                      r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + label + \
                      r'\end{equation*}' + '\n' + '\end{document}'
            f3 = open('label.tex', mode='w')
            f3.write(pdfText)
            f3.close()
            sub = subprocess.Popen("pdflatex -halt-on-error " + "label.tex", shell=True, stdout=subprocess.PIPE)
            sub.wait()

            pdfFiles = []
            for _, _, pf in os.walk(os.getcwd()):
                pdfFiles = pf
                break

            if 'label.pdf' in pdfFiles:
                try:
                    pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                              r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + cct + \
                              r'\end{equation*}' + '\n' + '\end{document}'
                    f3 = open('cct.tex', mode='w')
                    f3.write(pdfText)
                    f3.close()
                    sub = subprocess.Popen("pdflatex -halt-on-error " + "cct.tex", shell=True, stdout=subprocess.PIPE)
                    sub.wait()

                    pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                              r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + edsl + \
                              r'\end{equation*}' + '\n' + '\end{document}'
                    f3 = open('edsl.tex', mode='w')
                    f3.write(pdfText)
                    f3.close()
                    sub = subprocess.Popen("pdflatex -halt-on-error " + "edsl.tex", shell=True, stdout=subprocess.PIPE)
                    sub.wait()

                    os.system('convert -background white -density 200  -quality 100 -strip ' + 'label.pdf ' + 'label.png')
                    os.system('convert -background white -density 200  -quality 100 -strip ' + 'cct.pdf ' + 'cct.png')
                    os.system('convert -background white -density 200  -quality 100 -strip ' + 'edsl.pdf ' + 'edsl.png')

                    label = Image.open('label.png').convert('L')
                    cct = Image.open('cct.png').convert('L')
                    edsl = Image.open('edsl.png').convert('L')
                    '''
                    label = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('label.png').convert('L'))))
                    cct = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('cct.png').convert('L'))))
                    edsl = ImgCandidate.deleSpace(
                        ImgCandidate.deletePadding(np.array(Image.open('edsl.png').convert('L'))))
                    label = Image.fromarray(label)
                    cct = Image.fromarray(cct)
                    edsl = Image.fromarray(edsl)
                    '''
                    if label == cct == edsl:
                        accList.append(1)
                    else:
                        #logger.info('png images are not equal')
                        label.save(root_path+'/image_cons_98K/'+str(index)+'_label.png')
                        cct.save(root_path+'/image_cons_98K/'+str(index)+'_cct.png')
                        edsl.save(root_path + '/image_cons_98K/' + str(index) + '_edsl.png')
                        accList.append(0)

                except Exception as e:
                    logger.info('catch exception {}',e)
                    accList.append(0)

            else:
                logger.info('predict.pdf is not existed')

                accList.append(0)
        logger.info('current match:{}',sum(accList)/len(accList))
        os.system('rm -rf *.aux')
        os.system('rm -rf *.log')
        os.system('rm -rf *.tex')
        os.system('rm -rf *.pdf')
        os.system('rm -rf *.png')
        index += 1


    logger.info('Match:{}',sum(accList)/len(accList))
