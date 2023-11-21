import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
import pickle
from PIL import Image
mask_padding = 0


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

def keep_ratio_padding(img,ratio):
    h = img.shape[0]
    w = img.shape[1]
    if h / w < ratio:
        pad_len = int((w * ratio - h) // 2)
        img = np.pad(img, ((pad_len, pad_len), (0, 0)), 'constant', constant_values=(255, 255))
    else:
        pad_len = int((h / ratio - w) // 2)
        img = np.pad(img, ((0, 0), (pad_len, pad_len)), 'constant', constant_values=(255, 255))
    return img

class trmerdataset(torch.utils.data.Dataset):
    def __init__(self, data,imgSize):
        self.img = data[0]
        self.ratio = imgSize[0]/imgSize[1]
        self.target_teaching_label = data[1]
        self.target_predict_label = data[2]
        self.transform = Compose([
            Resize(imgSize),
            ToTensor()
        ])
    def __getitem__(self, index):
        img = delete_Padding(self.img[index])
        img = keep_ratio_padding(img,self.ratio)
        return self.transform(Image.fromarray(img)), \
               self.target_teaching_label[index], \
               self.target_predict_label[index]

    def __len__(self):
        return len(self.img)

def collate_fn(labels):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(label) for label in labels)

    batch_target_label = np.array([
            label + [mask_padding] * (max_len - len(label))
            for label in labels])
    batch_target_label = torch.LongTensor(batch_target_label)

    return batch_target_label, max_len

def paired_collate_fn(data):
    img, teaching_label, predict_label = list(zip(*data))
    teaching_label, max_len = collate_fn(teaching_label)
    predict_label, _ = collate_fn(predict_label)
    return torch.stack(img,0), teaching_label, predict_label, max_len

if __name__ == '__main__':

    file = '/home/zhou/EDSL-master/data/new_processed_data/train.npy'
    data = np.load(file)
    # sample_img = data[0][0]
    # sample_teach = data[0][1]
    # img = data[0]
    # target_teaching_label = data[1]
    # target_predict_label = data[2]
    # print(target_predict_label[200])
    # print(target_teaching_label[200])
    train_loader = torch.utils.data.DataLoader(
        trmerdataset(
            data),
        num_workers=2,
        batch_size=16,
        collate_fn=paired_collate_fn,
        shuffle=True
    )

    # for x, y in enumerate(train_loader):
    #     img, teach, predict = y
        # print(z)
    batch_data = iter(train_loader)

    img, teach_label, predict_label, max_len = batch_data.next()
    print('test good')

