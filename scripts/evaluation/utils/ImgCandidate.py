import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import copy
import math
from skimage import measure



#传入的参数是一个二维矩阵
def deletePadding(imgNp):

    #result = (imgNp > 220)
    #result1 = (imgNp[imgNp>220] = 255)
    imgNp[imgNp > 220] = 255
    #使用plt.imshow不加cmap参数可能出现错误
    #plt.imshow(imgNp,cmap="gray")
    #plt.show()

    binary = imgNp != 255
    #plt.imshow(binary,cmap='gray')
    #plt.show()
    #axis代表跨行，down，所以就是按列计算，
    sum_0 = np.sum(binary,axis=0)
    sum_1 = np.sum(binary,axis=1)
    #所以到这儿大概都是明白的
    sum_0_left = min(np.argwhere(sum_0 != 0).tolist())[0]
    sum_0_right = max(np.argwhere(sum_0 != 0).tolist())[0]
    sum_1_left = min(np.argwhere(sum_1 != 0).tolist())[0]
    sum_1_right = max(np.argwhere(sum_1 != 0).tolist())[0]
    delImg = imgNp[sum_1_left:sum_1_right + 1,sum_0_left :sum_0_right+ 1]

    return  delImg

def deleSpace(imgNp):
    binary = imgNp != 255
    sum_0 = np.sum(binary, axis=0).tolist()
    index_0 = []

    for i in range(len(sum_0)):
        if sum_0[i] != 0:
            index_0.append(i)

    imgNp = imgNp[:,index_0]

    return imgNp


def imgResize(arr,square=30):

    a_height = arr.shape[0]
    a_weight = arr.shape[1]


    ratio = square / max(a_height,a_weight)
    arr = np.array(Image.fromarray(arr).resize((math.ceil(a_weight*ratio),math.ceil(a_height*ratio)),Image.ANTIALIAS))

    a_height = arr.shape[0]
    a_weight = arr.shape[1]

    if a_height < square:
        h_1 = int((square - a_height)/2)
        h_2 = square - a_height- h_1
        if h_2!=0:
            arr = np.vstack((np.ones((h_1,a_weight)) * 255,arr,np.ones((h_2,a_weight)) * 255))
        else:
            arr = np.vstack((np.ones((h_1, a_weight)) * 255, arr))
    if a_weight <square:
        w_1 = int((square - a_weight)/2)
        w_2 = square-a_weight-w_1
        if w_2!=0:
            arr = np.hstack((np.ones((arr.shape[0],w_1)) * 255,arr,np.ones((arr.shape[0],w_2)) * 255))
        else:
            arr = np.hstack((np.ones((square,w_1)) * 255, arr))
    arr = arr[:square,:square].astype(np.int16)


    return arr

#这是一个连通域算法，
def getAllCandidate(img, THREASHOLD):
    img = np.array(img)
    binary = img < 220
    #plt.imshow(binary,cmap='gray')
    #plt.show()
    #分割点，这是啥意思呀
    splitPoint = (img > THREASHOLD) & (img < 255)
    #plt.imshow(splitPoint,cmap='gray')
    #plt.show()

    binary = binary * (splitPoint == False)
    #measure.label(input,connectivity=None),当input是二维是connectivity只有两个取值0和1，分别是四联通和八联通
    label =  (measure.label(binary, connectivity=2))
    #plt.imshow(label,cmap='gray')
    #plt.show()

    #label_add_spilt是一个拷贝
    label_add_split = copy.copy(label)
    times = 0
    while True in splitPoint and times < 10 :
        #找到对应条件元素的下标
        for item in np.argwhere(splitPoint == True):
            top = item[0] - 1 if item[0] - 1 > 0 else 0
            down = item[0] + 2 if item[0] + 2 < label.shape[0] else label.shape[0]
            left = item[1] - 1 if item[1] - 1 > 0 else 0
            right = item[1] + 2 if item[1] + 2 < label.shape[1] else label.shape[1]

            #reshape（-1）,进行展开
            area = label[top:down,left:right]
            area = label[top:down,left:right].reshape(-1)

            #bin的数量比area中最大的元素大一，每个bin给出了其索引值在area中出现的次数
            count = np.bincount(area)
            if len(count) > 1:
                count[0] = 0
                label_add_split[item[0],item[1]] = np.argmax(count)
                splitPoint[item[0],item[1]] = False
        label = label_add_split
        times += 1

    if True in splitPoint:
        for item in np.argwhere(splitPoint == True):
            label_add_split[item[0], item[1]] = 0


    labelCount = np.max(label_add_split)


    labelPosition = []

    for i in range(labelCount):

        tmpLeft = np.min(np.where(label_add_split == i + 1)[1])
        tmpRight = np.max(np.where(label_add_split == i + 1)[1])
        tmpTop = np.min(np.where(label_add_split == i + 1)[0])
        tmpDown = np.max(np.where(label_add_split == i + 1)[0])

        imgTmp = copy.copy(img)
        imgTmp[label_add_split != i+1] = 255
        imgTmp = deletePadding(imgTmp)
        imgTmp = imgResize(imgTmp,24)

        pad1 = np.ones((3, imgTmp.shape[1])) * 255
        imgTmp = np.concatenate((pad1,imgTmp),axis=0)
        imgTmp = np.concatenate((imgTmp,pad1),axis=0)

        pad2 = np.ones((imgTmp.shape[0],3)) * 255
        imgTmp = np.concatenate((pad2, imgTmp), axis=1)
        imgTmp = np.concatenate((imgTmp, pad2), axis=1)


        labelPosition.append((imgTmp,(tmpTop, tmpDown+1, tmpLeft, tmpRight+1)))


    return labelPosition
