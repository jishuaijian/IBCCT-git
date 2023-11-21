import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

import datetime

from PIL import Image
import matplotlib.pyplot as plt

import torch.utils.data
from dataset import trmerdataset, paired_collate_fn
import CTNNnetwork
from loguru import logger
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
if root_path not in sys.path:
    sys.path.append((root_path))
sys.path.append(root_path)
warnings.filterwarnings('ignore')
writer = SummaryWriter(logdir='../CTNN_output/no_aug/small/98K/tensorboard',filename_suffix='_loss')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_dataloader(train_data, val_data, batch_size,imgSize):
    train_loader = torch.utils.data.DataLoader(
        dataset=trmerdataset(train_data,imgSize),
        num_workers=8,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=trmerdataset(val_data,imgSize),
        num_workers=8,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    return train_loader, val_loader




def make_tgt_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        return loss


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


def make_model(tgt_vocab, encoderN=6, decoderN=6,
               d_model=256, d_ff=1024, h=8, dropout=0.0, patch_size=16, img_h=56, img_w=224,kernel_size=9):
    "Helper: Construct a model from hyperargs."
    c = copy.deepcopy
    attn = CTNNnetwork.MultiHeadedAttention(h, d_model, dropout)
    ff = CTNNnetwork.PositionwiseFeedForward(d_model, d_ff, dropout)
    model = CTNNnetwork.EncoderDecoder(
        CTNNnetwork.Encoder(CTNNnetwork.EncoderLayer(d_model=d_model, kernel_size=kernel_size, patch_size=patch_size), N=encoderN),
        CTNNnetwork.Decoder(CTNNnetwork.DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N=decoderN),
        CTNNnetwork.Embeddings(d_model, tgt_vocab),
        CTNNnetwork.Generator(d_model, tgt_vocab),
        patch_size=patch_size, img_h=img_h, img_w=img_w, d_model=d_model
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # init the para
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
        prob = model.generator(out[:, -1, :].squeeze(0)).unsqueeze(1)
        _, predictTmp = prob.max(dim=-1)
        lastWord = torch.cat((lastWord, predictTmp), dim=-1)
    prob = model.generator.proj(out)

    return prob


def process_args():
    parser = argparse.ArgumentParser(description='Get args')

    parser.add_argument('--num_encoder',dest='encoderN',
                         default=20,type=int,required=False,
                        help='the number of encoder blocks')
    parser.add_argument('--num_decoder', dest='decoderN',
                        default=8, type=int, required=False,
                        help='the number of decoder blocks')
    parser.add_argument('--d_model',dest='d_model',
                        default=256,type=int,
                        help='the dimension of hidden d_model')
    parser.add_argument('--kernel_size',dest='kernel_size',
                        default=9,type=int,
                        help='the kernel size of encoder conv')

    parser.add_argument('--print_intevel',dest='print_interval',
                        default=10,type=int)
    parser.add_argument('--val_interval',dest='val_interval',
                        default=3,type=int)

    parser.add_argument('--dataset', dest='dataset',
                        default='98K', type=str)

    parser.add_argument('--aug',dest='aug',
                        default='no_aug',type=str)
    
    parser.add_argument('--imgSize',dest='imgSize',
                        default=[112,224],type=list)

    parser.add_argument('--batch_size',dest='batch_size',
                        default=12,type=int)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    date_time = datetime.datetime.now().strftime("%Y%m_%d_%I_%M_%S")
    
    args = process_args()
    logger_path = os.path.join(root_path,f'log/log_{args.dataset}_{args.aug}_{str(args.imgSize)}_encoder={args.encoderN}_decoder={args.decoderN}_d_model={args.d_model}_kernel={args.kernel_size}_{date_time}.txt')
    logger.add(logger_path,mode='a')
    logger.info('save log to {}',logger_path)
    logger.info('args:{}',args)

    MAXLENGTH = 200

    vocab_file_path = os.path.join(root_path,'data',args.dataset,'annotations/latex_vocab.txt')
    f5 = open(vocab_file_path, encoding='utf-8').readlines()

    PAD = 0
    START = 1
    END = 2

    index_label_dic = {}
    label_index_dic = {}

    i = 3
    for item_f5 in f5:
        word = item_f5.strip()
        if len(word) > 0:
            label_index_dic[word] = i
            index_label_dic[i] = word
            i += 1
    label_index_dic['unk'] = i
    index_label_dic[i] = 'unk'
    i += 1

    BATCH_SIZE =args.batch_size

    train_file = os.path.join(root_path,'data',args.dataset,'numpy',args.aug,'train.npy')
    val_file = os.path.join(root_path, 'data', args.dataset, 'numpy', args.aug, 'val.npy')

    train_data = np.load(train_file, allow_pickle=True)
    val_data = np.load(val_file, allow_pickle=True)

    logger.info('load numpy file of train from {}',train_file)
    logger.info('load numpy file of val from {}',val_file)

    train_loader, val_loader = prepare_dataloader(train_data, val_data, BATCH_SIZE,args.imgSize)
    #####Regularization args####
    dropout = 0.2
    l2 = 1e-4
    #################

    '''585-952-876
    Init Model
    '''
#########################################################
    model = make_model(len(index_label_dic) + 3, encoderN=args.encoderN, decoderN=args.decoderN,
                       d_model=args.d_model, d_ff=1024, dropout=dropout, patch_size=4, img_h=args.imgSize[0], img_w=args.imgSize[1],kernel_size=args.kernel_size).to(device)
    logger.info('encoder={},decoder={},d_model={},kernel_size={}'.format(args.encoderN,args.decoderN,args.d_model,args.kernel_size))
##########################################################
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_parameters = 0
    decoder_parameters = 0
    other_parameters = 0
    for name,param  in model.named_parameters():
        tmp_num = param.numel()
        if name.startswith('encoder'):
            encoder_parameters += tmp_num
        elif name.startswith('decoder'):
            decoder_parameters += tmp_num
        else:
            other_parameters += tmp_num

    logger.info('number of params:{:.3f}M',n_parameters/(1024*1024))
    logger.info('number of encoder params:{:.3f}',encoder_parameters/(1024*1024))
    logger.info('number of decoder params:{:.3f}',decoder_parameters/(1024*1024))
    logger.info('\n{}',model)


    criterion = LabelSmoothing(size=len(label_index_dic) + 3, padding_idx=0, smoothing=0.1)
    lossComput = SimpleLossCompute(model.generator, criterion)

    learningRate = 3e-4
    totalCount = 0
    exit_count = 0
    bestVal = 0
    bestTrainList = 0
    criterionVal = nn.CrossEntropyLoss(ignore_index=0, size_average=True).to(device)
    #batch_train_data = iter(train_loader)
    #batch_val_data = iter(val_loader)

    while True:
        """
        train
        """
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=l2)


        lossListTrain = []
        latexAccListTrain = []
        bleuListTrain = []
        editdistListTrain = []
        #for batch_i in tqdm(range(5)):
        max_iter = len(train_loader)
        for num_batch, data in enumerate(train_loader):

            #img, tgt_teaching, tgt_predict, _ = batch_train_data.next()
            img, tgt_teaching, tgt_predict, max_batch_length = data

            img, tgt_teaching, tgt_predict = img.to(device), tgt_teaching.to(device), tgt_predict.to(device)
            if max_batch_length > MAXLENGTH:
                tgt_teaching = tgt_teaching[:,:MAXLENGTH]
                tgt_predict = tgt_predict[:,:MAXLENGTH]
            out = model.forward(img, tgt_teaching)
            _, latexPredict = model.generator(out).max(dim=-1)

            for i in range(len(latexPredict)):

                if 2 in tgt_predict[i].tolist():
                    endIndex = tgt_predict[i].tolist().index(2)
                else:
                    endIndex = MAXLENGTH - 1

                predictTmp = latexPredict[i].tolist()[:endIndex + 1]
                labelTmp = tgt_predict[i].tolist()[:endIndex + 1]
                if predictTmp == labelTmp:
                    latexAccListTrain.append(1)
                else:
                    latexAccListTrain.append(0)

                bleuScore = sentence_bleu([labelTmp], predictTmp)
                editdist = edit_distance(labelTmp, predictTmp)
                editdistScore = 1 - editdist/max(len(labelTmp), len(predictTmp))
                editdistListTrain.append(editdistScore)
                bleuListTrain.append(bleuScore)

            loss = lossComput(out, tgt_predict, len(tgt_predict))

            lossListTrain.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if num_batch % args.print_interval == 0:
                logger.info('epoch:{} iter:{}/{} loss={:.3f}',totalCount,num_batch,max_iter,loss.item())

        trainLoss = sum(lossListTrain) / len(lossListTrain)
        trainAcc = sum(latexAccListTrain) / len(latexAccListTrain)
        trainBleu = sum(bleuListTrain) / len(bleuListTrain)
        trainEditDist = sum(editdistListTrain) / len(editdistListTrain)
        #writer.add_scalar('train_loss',trainLoss,totalCount)
        #writer.add_scalar('train_acc',trainAcc,totalCount)



        """
        Val
        
        """
        logger.info('start eval')
        model.eval()
        latexAccListVal = []
        latexLossListVal = []
        bleuListval = []
        editdistListval = []
        with torch.no_grad():
            # for batch_i in tqdm(range(len(latex_batch_index))):
            # for batch_i in tqdm(range(5)):
            for num_batch, data in enumerate(tqdm(val_loader)):

                img_val, tgt_teaching_val, tgt_predict_val, tgtMaxBatch = data

                img_val, tgt_teaching_val, tgt_predict_val = \
                    img_val.to(device), tgt_teaching_val.to(device), tgt_predict_val.to(device)

                if tgtMaxBatch>MAXLENGTH:
                    tgt_teaching_val = tgt_teaching_val[:, :MAXLENGTH]
                    tgt_predict_val = tgt_predict_val[:, :MAXLENGTH]
                    tgtMaxBatch = MAXLENGTH

                out = greedy_decode(model, img_val, tgtMaxBatch)
                _, latexPredict_val = out.max(dim=-1)

                for i in range(len(latexPredict_val)):
                    if 2 in latexPredict_val[i].tolist():
                        endIndex = latexPredict_val[i].tolist().index(2)
                    else:
                        endIndex = MAXLENGTH
                    predictTmp = latexPredict_val[i].tolist()[:endIndex]
                    endIndex = tgt_predict_val[i].tolist().index(2) if len(tgt_predict_val[i]) < MAXLENGTH else MAXLENGTH
                    labelTmp = tgt_predict_val[i].tolist()[:endIndex]

                    if predictTmp == labelTmp:
                        latexAccListVal.append(1)
                    else:
                        latexAccListVal.append(0)
                    bleuScore = sentence_bleu([labelTmp], predictTmp)
                    editdist = edit_distance(labelTmp, predictTmp)
                    editdistScore = 1 - editdist / max(len(labelTmp), len(predictTmp), 1)
                    editdistListval.append(editdistScore)
                    bleuListval.append(bleuScore)
                out = out.contiguous().view(-1, out.size(-1))
                targets = tgt_predict_val.contiguous().view(-1)
                loss = criterionVal(out, targets)
                latexLossListVal.append(loss.item())

        valAcc = sum(latexAccListVal) / len(latexAccListVal)
        valLoss = sum(latexLossListVal) / len(latexLossListVal)
        valBleu = sum(bleuListval) / len(bleuListval)
        valEditDist = sum(editdistListval) / len(editdistListval)
        #writer.add_scalar('val_loss',valLoss,totalCount)
        #writer.add_scalar(('val_accc'),valAcc,totalCount)


        if valAcc > bestVal:
            model_path = os.path.join(root_path,f'model/model_{args.dataset}_{args.aug}_{str(args.imgSize)}_encoder={args.encoderN}_decoder={args.decoderN}_d_model={args.d_model}_kernel={args.kernel_size}.pkl')
            torch.save(model.state_dict(), model_path)
            logger.info('save mdoel_state_dict to {}'.format(model_path))
            bestVal = valAcc
            exit_count = 0
        else:
            exit_count += 1
            if exit_count > 0 and exit_count % 3 == 0:
                learningRate *= 0.5
        if exit_count == 10:
            logger.info('the best resutl is {}',bestVal)
            exit()

        logger.info(
            'Epoch:{} TrainingSetLoss:{:.3f} Acc:{:.3f} BLEU:{:.3f} EditDist:{:.3f}'.format(totalCount, trainLoss,
                                                                                            trainAcc, trainBleu,
                                                                                            trainEditDist))
        logger.info(
            'Epoch:{} ValSetLoss:{:.3f} ACC:{:.3f} BLEU:{:.3f} EditDist:{:.3f}'.format(totalCount, valLoss, valAcc,
                                                                                       valBleu, valEditDist))
        logger.info('Epoch:{} Learning_rate:{} exit_count:{}'.format(totalCount, learningRate, exit_count))

        totalCount += 1
