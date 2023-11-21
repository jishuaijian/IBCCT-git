import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import argparse
import random
from torch.autograd import Variable
from tqdm import tqdm
from torch import optim
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics import edit_distance
import warnings
import sys
from loguru import logger
import torch.utils.data
import cv2
from PIL import Image
from torchvision.transforms import transforms

#import from this project
from dataset import trmerdataset,paired_collate_fn
import CTNNnetwork

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
warnings.filterwarnings('ignore')

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_model(tgt_vocab,encoderN=20,decoderN=8,
               d_model=256,d_ff=1024,h=8,dropout=0.2,patch_size=4,img_h=112,img_w=224,kernel_size=5):
    'Helper:Construct a model from hyperparameters.'
    c = copy.deepcopy
    attn = CTNNnetwork.MultiHeadedAttention(h,d_model,dropout)
    ff = CTNNnetwork.PositionwiseFeedForward(d_model,d_ff,dropout)
    model = CTNNnetwork.EncoderDecoder(
        CTNNnetwork.Encoder(CTNNnetwork.EncoderLayer(d_model=d_model,kernel_size=kernel_size,patch_size=patch_size),N=encoderN),
        CTNNnetwork.Decoder(CTNNnetwork.DecoderLayer(d_model, c(attn), c(attn),
                                                     c(ff), dropout), N=decoderN),
        CTNNnetwork.Embeddings(d_model,tgt_vocab),
        CTNNnetwork.Generator(d_model,tgt_vocab),
        patch_size=patch_size,img_h=img_h,img_w=img_w,d_model=d_model
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_normal(p)
    return model

def greedy_decode(model, img, max_len):
    memory = model.encode(img)
    src_mask = torch.ones(memory.size(0), model.num_patches)
    src_mask = src_mask.unsqueeze(1).to(device)

    lastWord = torch.ones(len(memory), 1).long().to(device)
    for i in range(max_len):
        tgt_mask = Variable(subsequent_mask(lastWord.size(1)).type_as(memory.data))
        tgt_mask = tgt_mask.repeat(memory.size(0), 1, 1)
        out = model.decode(memory, src_mask, Variable(lastWord), tgt_mask)
        tmp_put = out[:, -1, :]
        tmp_out = tmp_put.squeeze(0)
        prob = model.generator(out[:, -1, :]).unsqueeze(1)
        _, predictTmp = prob.max(dim=-1)
        lastWord = torch.cat((lastWord, predictTmp), dim=-1)
    prob = model.generator.proj(out)

    return prob

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

def pre_process(img):
    img = delete_Padding(img)
    h = img.shape[0]
    w = img.shape[1]
    if h / w < 0.5:
        pad_len = (w // 2 - h) // 2
        im_pad = np.pad(img, ((pad_len, pad_len), (0, 0)), 'constant', constant_values=(255, 255))
    else:
        if h / w > 0.5:
            pad_len = (h * 2 - w) // 2
            im_pad = np.pad(img, ((0, 0), (pad_len, pad_len)), 'constant', constant_values=(255, 255))

    resize = transforms.Resize([112,224])
    resized_img = resize(Image.fromarray(im_pad))
    return np.array(resized_img)

def process_args():
    parser = argparse.ArgumentParser(description='Get parameters')
    parser.add_argument('--vocab',dest='vocab_file_path',
                        default=root_path+'/data/annotations/latex_vocab.txt')
    parser.add_argument('--image',dest='image_file',
                        default='../data/asserts/demo.png')
    return parser.parse_args()
if __name__=='__main__':

    parameters = process_args()
    MAXLENGTH = 200
    vocab_file = open(parameters.vocab_file_path,encoding='utf-8').readlines()

    PAD = 0
    START = 1
    END = 2
    index_label_dic = {}
    label_index_dic = {}
    i = 3
    for item_f5 in vocab_file:
        word = item_f5.strip()
        if len(word) > 0:
            label_index_dic[word] = i
            index_label_dic[i] = word
            i += 1
    label_index_dic['unk'] = i
    index_label_dic[i] = 'unk'
    i += 1
    index_label_dic[0] = ''
    index_label_dic[1] = ''
    index_label_dic[2] = ''

    img = Image.open(parameters.image_file).convert('L')

    #first question: why need a float()
    #so in test.py这里已经是一个float了，对应于batch和channnel的unsqueeze
    img = pre_process(np.array(img))
    img = torch.from_numpy(img).float().to(device)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)

    model = make_model((len(index_label_dic)))
    state_dict = torch.load(root_path+'/CTNN_output/no_aug/small/20K/model.pkl')
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        out = greedy_decode(model,img,MAXLENGTH)
        _, latexPredict = out.max(dim=-1)
        if 2 in latexPredict[0].tolist():
            endIndex = latexPredict[0].tolist().index(2)
        else:
            endIndex = MAXLENGTH-1

        predictStr = ' '.join([index_label_dic[item] for item in latexPredict[0].tolist()]).strip()
        print(predictStr)




