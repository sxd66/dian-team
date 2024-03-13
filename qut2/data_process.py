from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import numpy as np

path='../data/fashion-mnist/t10k-images-idx3-ubyte'


with open(path,'rb') as f:
    x=f.read()
x = int.from_bytes(x[2], "big")
y=np.array(x[16:]).astype(np.uint8)
print(x.shape[0:])


#np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)


