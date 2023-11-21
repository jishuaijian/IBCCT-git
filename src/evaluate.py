import sys
import os

from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import argparse
import subprocess
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


from loguru import logger

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
if root_path not in sys.path:
    sys.path.append(root_path)

from scripts.evaluation.utils import ImgCandidate

def formula_to_index(formula, dic):
    tmp = []
    for word in formula.strip().split(' '):
        if len(word.strip()) > 0:
            if word.strip() in dic:
                tmp.append(dic[word.strip()])
            else:
                tmp.append(dic['UNK'])
    return tmp


def process_args():
    parser = argparse.ArgumentParser(description='Get args')

    parser.add_argument('--dataset', dest='dataset',
                        default='98K', type=str)

    parser.add_argument('--aug', dest='aug',
                        default='tri_aug', type=str)

    parser.add_argument('--imgSize', dest='imgSize',
                        default=[112, 224], type=list)
    parser.add_argument('--num_encoder', dest='encoderN',
                        default=20, type=int, required=False,
                        help='the number of encoder blocks')
    parser.add_argument('--num_decoder', dest='decoderN',
                        default=8, type=int, required=False,
                        help='the number of decoder blocks')
    parser.add_argument('--d_model', dest='d_model',
                        default=256, type=int,
                        help='the dimension of hidden d_model')
    parser.add_argument('--kernel_size', dest='kernel_size',
                        default=9, type=int,
                        help='the kernel size of encoder conv')

    args = parser.parse_args()
    return args


def computeB4(result_file):
    bleu4_list = []

    intervel_list = list(range(10,51,10))
    bleu4_length_list = [[] for _ in range(len(intervel_list)+1)]

    for line in result_file_lines:
        predict_formula = line.strip().split('\t')[0][8:].split(' ')
        label_formula = line.strip().split('\t')[1][6:].split(' ')

        if len(label_formula) >= 4:
            cur_bleu = 0.
            if len(predict_formula) < 4:
                cur_bleu = 0.
            else:
                cur_bleu = sentence_bleu([label_formula], predict_formula, weights=(0, 0, 0, 1))
            bleu4_list.append(cur_bleu)

            length = len(label_formula)
            if length<=intervel_list[0]:
                bleu4_length_list[0].append(cur_bleu)
            elif length<=intervel_list[1]:
                bleu4_length_list[1].append(cur_bleu)
            elif length <= intervel_list[2]:
                bleu4_length_list[2].append(cur_bleu)
            elif length <= intervel_list[3]:
                bleu4_length_list[3].append(cur_bleu)
            elif length <= intervel_list[4]:
                bleu4_length_list[4].append(cur_bleu)
            else:
                bleu4_length_list[5].append(cur_bleu)

    result = [sum(item_list)/len(item_list) for item_list in bleu4_length_list]
    for item_list in bleu4_length_list:
        print(len(item_list))

    print(sum([len(item_list) for item_list in bleu4_length_list]))

    return sum(bleu4_list) / len(bleu4_list),result


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


def computeR4(result_file_lines):

    intervel_list = list(range(10,51,10))
    rough4_length_list = [[] for _ in range(len(intervel_list)+1)]

    rough4_list = []
    for line in result_file_lines:
        predict_formula = line.strip().split('\t')[0][8:].split(' ')
        label_formula = line.strip().split('\t')[1][6:].split(' ')

        cur_rough = getRouge_N(predict_formula, label_formula, 4)
        if cur_rough == None:
            pass
        else:
            rough4_list.append(cur_rough)

            length = len(label_formula)
            if length <= intervel_list[0]:
                rough4_length_list[0].append(cur_rough)
            elif length <= intervel_list[1]:
                rough4_length_list[1].append(cur_rough)
            elif length <= intervel_list[2]:
                rough4_length_list[2].append(cur_rough)
            elif length <= intervel_list[3]:
                rough4_length_list[3].append(cur_rough)
            elif length <= intervel_list[4]:
                rough4_length_list[4].append(cur_rough)
            else:
                rough4_length_list[5].append(cur_rough)


    result = [sum(item_list) / len(item_list) for item_list in rough4_length_list]
    print(sum([len(item_list) for item_list in rough4_length_list]))

    return sum(rough4_list) / len(rough4_list),result


def computeMatch(result_file_lines):

    intervel_list = list(range(10,51,10))
    match_length_list = [[] for _ in range(len(intervel_list) + 1)]

    accList = []
    for line in result_file_lines:
        cur_match = 0
        predict_formula = line.strip().split('\t')[0][8:]
        label_formula = line.strip().split('\t')[1][6:]

        if predict_formula == label_formula:
            cur_match = 1
            accList.append(1)
        else:
            pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                      r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + predict_formula + \
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
                              r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + label_formula + \
                              r'\end{equation*}' + '\n' + '\end{document}'
                    f3 = open('label.tex', mode='w')
                    f3.write(pdfText)
                    f3.close()
                    sub = subprocess.Popen("pdflatex -halt-on-error " + "label.tex", shell=True, stdout=subprocess.PIPE)
                    sub.wait()

                    os.system('convert -strip ' + 'label.pdf ' + 'label.png')
                    os.system('convert -strip ' + 'predict.pdf ' + 'predict.png')
                    label = Image.open('label.png').convert('L')

                    predict = Image.open('predict.png').convert('L')

                    if label == predict:
                        cur_match = 1
                        accList.append(1)
                    else:
                        cur_match = 0
                        accList.append(0)
                except Exception as e:
                    cur_match = 0
                    accList.append(0)
            else:
                cur_match = 0
                accList.append(0)
        os.system('rm -rf *.aux')
        os.system('rm -rf *.log')
        os.system('rm -rf *.tex')
        os.system('rm -rf *.pdf')
        os.system('rm -rf *.png')

        length = len(label_formula)
        if length <= intervel_list[0]:
            match_length_list[0].append(cur_match)
        elif length <= intervel_list[1]:
            match_length_list[1].append(cur_match)
        elif length <= intervel_list[2]:
            match_length_list[2].append(cur_match)
        elif length <= intervel_list[3]:
            match_length_list[3].append(cur_match)
        elif length <= intervel_list[4]:
            match_length_list[4].append(cur_match)
        else:
            match_length_list[5].append(cur_match)

    result = [sum(item_list) / len(item_list) for item_list in match_length_list]
    print(sum([len(item_list) for item_list in match_length_list]))

    return sum(accList) / len(accList),result


def computeMatch_WS(result_file_lines):

    intervel_list = list(range(10,51,10))
    match_ws_length_list = [[] for _ in range(len(intervel_list) + 1)]

    accList = []
    for line in result_file_lines:
        predict_formula = line.strip().split('\t')[0][8:]
        label_formula = line.strip().split('\t')[1][6:]

        cur_match_ws = 0
        if predict_formula == label_formula:
            cur_match_ws = 1
            accList.append(1)
        else:
            pdfText = r'\documentclass{article}' + '\n' + r'\usepackage{amsmath,amssymb}' + '\n' + '\pagestyle{empty}' + '\n' + \
                      r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + predict_formula + \
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
                              r'\thispagestyle{empty}' + '\n' + r'\begin{document}' + '\n' + r'\begin{equation*}' + '\n' + label_formula + \
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
                        cur_match_ws = 1
                        accList.append(1)
                    else:
                        cur_match_ws = 0
                        accList.append(0)
                except Exception as e:
                    cur_match_ws = 0
                    accList.append(0)

            else:
                cur_match_ws = 0
                accList.append(0)

        os.system('rm -rf *.aux')
        os.system('rm -rf *.log')
        os.system('rm -rf *.tex')
        os.system('rm -rf *.pdf')
        os.system('rm -rf *.png')

        length = len(label_formula)
        if length <= intervel_list[0]:
            match_ws_length_list[0].append(cur_match_ws)
        elif length <= intervel_list[1]:
            match_ws_length_list[1].append(cur_match_ws)
        elif length <= intervel_list[2]:
            match_ws_length_list[2].append(cur_match_ws)
        elif length <= intervel_list[3]:
            match_ws_length_list[3].append(cur_match_ws)
        elif length <= intervel_list[4]:
            match_ws_length_list[4].append(cur_match_ws)
        else:
            match_ws_length_list[5].append(cur_match_ws)

    result = [sum(item_list) / len(item_list) for item_list in match_ws_length_list]
    print(sum([len(item_list) for item_list in match_ws_length_list]))

    return sum(accList) / len(accList),result


if __name__ == "__main__":
    args = process_args()

    result_path = os.path.join(root_path,
                               f'result/result_{args.dataset}_{args.aug}_{str(args.imgSize)}_encoder={args.encoderN}_decoder={args.decoderN}_d_model={args.d_model}_kernel={args.kernel_size}.txt')
    #result_path = os.path.join(root_path,'result_edsl','predict_98K.txt')
    result_file_lines = open(result_path, encoding='utf-8').readlines()

    Bleu4,Bleu4_list = computeB4(result_file_lines)
    Rough4,Rough4_list = computeR4(result_file_lines)
    print(Bleu4)
    print(Rough4)
    Match,Match_list = computeMatch(result_file_lines)
    Match_WS,Match_WS_list = computeMatch_WS(result_file_lines)
    print(Match)
    print(Match_WS)

    metric_path = os.path.join(root_path,
                               f'result/result_{args.dataset}_{args.aug}_{str(args.imgSize)}_encoder={args.encoderN}_decoder={args.decoderN}_d_model={args.d_model}_kernel={args.kernel_size}_metric_length.txt')
    #metric_path = os.path.join(root_path,'result_edsl','predict_98K_length.txt')
    metric_txt = open(metric_path,mode='w')
    metric_txt.write('length\tB4\tR4\tMatch\tMatch-Ws\n')

    #intervel_list = list(range(10,51,10))
    intervel_list = list(range(20,101,20))
    for index,length in enumerate(intervel_list):
        metric_txt.write('<={}\t{}\t{}\t{}\t{}\t\n'.format(length,Bleu4_list[index],Rough4_list[index],Match_list[index],Match_WS_list[index]))
    metric_txt.write('> {}\t{}\t{}\t{}\t{}\t\n'.format(intervel_list[-1],Bleu4_list[-1],Rough4_list[-1],Match_list[-1],Match_WS_list[-1]))
    #print(Bleu4,Rough4,Match,Match_WS)
    #metric_txt.write('{}\t{}\t{}\t{}\n'.format(Bleu4, Rough4, Match, Match_WS))

