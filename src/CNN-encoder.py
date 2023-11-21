import torch.nn as nn
import copy
import torch

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class CNN_encoder(nn.Module):
    def __init__(self,dim,depth,kernel_size=9,patch_size=7):
        super(CNN_encoder,self).__init__()
        self.patch_embedding = nn.Conv2d(3,dim,kernel_size=patch_size,stride=patch_size)
        self.gelu = nn.GELU()
        self.batchNorm = nn.BatchNorm2d(dim)
        self.layers = clones(CNN_layer(dim=dim,kernel_size=kernel_size),depth)
        self.adaptivaAvg = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.gelu(x)
        x = self.batchNorm(x)
        for layer in self.layers:
            x = layer(x)
        x = self.adaptivaAvg(x)
        return x
class CNN_layer(nn.Module):
    def __init__(self,dim,kernel_size):
        super(CNN_layer,self).__init__()
        self.depth_conv = Residual(nn.Conv2d(dim,dim,kernel_size=kernel_size,groups=dim,padding='same'))
        self.point_conv = nn.Conv2d(dim,dim,kernel_size=1)
        self.gelu = nn.GELU()
        self.batchNorm = nn.BatchNorm2d(dim)
    def forward(self,x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.gelu(x)
        x = self.batchNorm(x)
        return x

if __name__ == '__main__':
    img = torch.rand(16,3,256,256)
    model = CNN_encoder(dim=256,depth=5,kernel_size=9,patch_size=7)
    result = model(img)
    print(result)