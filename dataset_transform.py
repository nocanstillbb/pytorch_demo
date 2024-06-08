import torch
import numpy as np
import torchvision
from  torch.utils.data import Dataset
import math

class WineDataset(Dataset):
    def __init__(self,transform = None) :
        #加载数据
        xy = np.loadtxt("./data/wine/wine.csv",skiprows=1,delimiter=",",dtype=np.float32) 
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_simples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index) -> any:
        #索引器  dataset[index]
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        #返回数据集长度
        return self.n_simples


class ToTensort:
    def __call__(self,sameple) :
        feature,label = sameple
        return torch.from_numpy(feature),torch.from_numpy(label)
class MultTrans:
    def __init__(self,factor) :
        self.factor = factor

    def __call__(self,sameple) :
        feature,label = sameple
        feature *= self.factor
        return feature,label

transform = ToTensort()
dataset = WineDataset(transform=MultTrans(2))
features,labels = dataset[0]
print(features)
print(type(features))


compose = torchvision.transforms.Compose([ToTensort(),MultTrans(2)])
dataset = WineDataset(transform=compose)
features,labels = dataset[0]
print(features)
print(type(features))


