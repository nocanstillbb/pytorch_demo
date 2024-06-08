# epoch 一次所有训练样本向前传播&向后传播的周期
# batch_numer 一次向前传播&向后传播的训练样本数量
# number of iteraters 迭代完所有样本的次数
# 例如有100个样本, 批次号为20, 迭代次数为100/20 = 5 ,epoch = 2 的话需要把所有数据训练两轮

import torch
import numpy as np
import torchvision
from  torch.utils.data import Dataset,DataLoader
import math


class WineDataset(Dataset):
    def __init__(self) -> None:
        #加载数据
        xy = np.loadtxt("./data/wine/wine.csv",skiprows=1,delimiter=",",dtype=np.float32) 
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_simples = xy.shape[0]


    def __getitem__(self, index) -> any:
        #索引器  dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        #返回数据集长度
        return self.n_simples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=4,num_workers=0)

#if __name__ == "__main__" : #如果num_workers大于1,则需要加上这一行,防止递归创建进程
epoch_num = 2
total_simples = len(dataset)
print(f"total simples:{total_simples}")
n_iteraters = math.ceil(total_simples/4)
for epoch in range(epoch_num):
    for i,(features,labels) in enumerate(dataloader) :
        if((i+1)%5 == 0):
            print(f"epoch:{epoch + 1 }  step:{i + 1}/{n_iteraters}   features:{features[0][0]} labels:{labels.shape}")
