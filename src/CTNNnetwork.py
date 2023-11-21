import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy
import seaborn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
seaborn.set_context(context="talk")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:1")
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_tgt_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, tgt_embed, generator, patch_size, img_h, img_w, d_model):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

        self.image_height, self.image_width = img_h, img_w
        self.patch_height, self.patch_width = patch_size, patch_size
        self.num_patches = (img_h // patch_size) * (img_w // patch_size)

        self.patch_embedding = nn.Conv2d(1,d_model,kernel_size=patch_size,stride=patch_size)
        self.gelu = nn.GELU()
        self.batchNorm = nn.BatchNorm2d(d_model)

    def forward(self, img, target_latex):
        memory = self.encode(img)
        #这里的mask的维度需要注意，这里为什么也需要一个unsqueeze
        src_mask = torch.ones(memory.size(0), self.num_patches)
        src_mask = src_mask.unsqueeze(1).to(device)

        tgt_teachingMask = make_tgt_mask(target_latex, 0)
        decode_embeded = self.decode(memory, src_mask, target_latex, tgt_teachingMask)
        return decode_embeded

    def encode(self,x):
        x = self.patch_embedding(x)
        x = self.gelu(x)
        x = self.batchNorm(x)
        x = self.encoder(x)
        return x

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_embed = self.tgt_embed(tgt)
        return self.decoder(tgt_embed, memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Encoder(nn.Module):
    def __init__(self,layer,N=6):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)

    def forward(self,x):
        B,C = x.size(0),x.size(1)
        for layer in self.layers:
            x = layer(x)

        return x.view(B,C,-1).transpose(1,2)

class EncoderLayer(nn.Module):
    def __init__(self,d_model=256,kernel_size=9,patch_size=7):
        super(EncoderLayer,self).__init__()
        self.depthConv = Residual(
            nn.Sequential(
                nn.Conv2d(d_model,d_model,kernel_size=kernel_size,groups=d_model,padding=kernel_size//2),
                nn.GELU(),
                nn.BatchNorm2d(d_model)
            )
        )
        self.pointConv = nn.Conv2d(d_model,d_model,kernel_size=1)
        self.gelu = nn.GELU()
        self.batchNorm = nn.BatchNorm2d(d_model)
    def forward(self,x):
        x = self.depthConv(x)
        x = self.pointConv(x)
        x = self.gelu(x)
        x = self.batchNorm(x)
        return x

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

        self.count = nn.Embedding(200, 256)

        # self.pos_pj = nn.Linear(512,256)

    def forward(self, x, memory, src_mask, tgt_mask):
        # src_count = torch.sum(src_mask,d_model=-1).squeeze(-1).long().cuda() - 1
        # src_count = self.count(src_count).unsqueeze(1).repeat(1,x.size(1),1)

        #get the embed of x
        list_range_x = torch.Tensor(list(range(x.size(1))))
        list_range_x = list_range_x.unsqueeze(0)
        list_range_x = list_range_x.repeat(x.size(0), 1)
        x_pos = torch.Tensor(list(range(x.size(1)))).unsqueeze(0).repeat(x.size(0), 1).long().to(device)
        # print(x_pos[0])
        x_pos = self.count(x_pos)

        # x_pos = self.pos_pj(torch.cat((src_count, x_pos),d_model=-1))
        x = x + x_pos

        # x_cat = torch.empty(x.size(0), 0, x.size(1), x.size(2)).cuda()
        # x_cat = torch.cat((x_cat, x.unsqueeze(1)), d_model=1)

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # x_cat = torch.cat((x_cat, x.unsqueeze(1)), d_model=1)

        # x_cat = torch.mean(x_cat, d_model=1)

        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

##############################################################################
def attention(query, key, value, mask=None, dropout=None, posr=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if posr is not None:
        posr = posr.unsqueeze(1)

        scores = scores + posr

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, posr=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout, posr=posr)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(F.relu(self.dropout(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x.long()) * math.sqrt(self.d_model)
        return embed



class EncoderPositionalEmbedding(nn.Module):
    def __init__(self, dmodel):
        super(EncoderPositionalEmbedding, self).__init__()

        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

    def forward(self, encoder_position):
        e = F.relu((self.fc1(encoder_position)))
        e = F.relu((self.fc2(e)))
        e = F.relu((self.fc3(e)))

        return e

if __name__ == '__main__':
    d_model = 256
    h=8
    dropout=0.2
    tgt_vocab = 193
    c = copy.deepcopy
    patch_size =4
    img_h = 112
    img_w = 224
    attn = MultiHeadedAttention(h=8, d_model=256, dropout=0.2)
    ff = PositionwiseFeedForward(d_model=256, d_ff=1024, dropout=0.2)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model=256,kernel_size=9,patch_size=4),depth=6),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                                     c(ff), dropout), N=6),
        Embeddings(d_model, tgt_vocab),
        Generator(d_model, tgt_vocab),
        patch_size=patch_size, img_h=img_h, img_w=img_w, d_model=d_model
    )
    img = torch.rand(16,1,112,224)
    tgt_latex = torch.rand(16,192)
    result = model(img,tgt_latex)
    print(result)
