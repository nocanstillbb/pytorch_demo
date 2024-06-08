#softmax是上古时期逻辑回归时首先被创建出来的激活函数,其作用是把一个线性张量转换为独热编码
#因在多层神经网络中经常有梯度消失的问题,后经常被ReUl激活函数取代


import torch.nn as nn
import numpy as np
import torch

def softmax(inputs):
    return  np.exp(inputs)  / np.sum(np.exp(inputs),axis = 0)

x = np.array([3 , 5.2 , 6 , 3])
y = softmax(x)
print(f"y : {y}")

x = torch.tensor([3 , 5.2 , 6 , 3])
y =  torch.softmax(x,dim=0)

#softmax激活函数通常使用交叉熵函数作为损失函数
#根据最大似然估计,softmax的所有概率项积乘取负对数,把积乘转为对数项积分,再求导函数为0时的解作为最优参数,再根据最优参数计算损失值,作为损失函数,以用于反向传播以梯度下降
def crossEntropyloss(onehot_y,predicted_y):
    loss = - np.sum(onehot_y * np.log(predicted_y))
    return loss

lable_onehost_y = np.array([0,0,1])

bad_prefict_y = np.array([.5 , .2 , .3]) #概率积乘需要为1
good_prefict_y =np.array([.2 , .3 , .5])

bad_loss = crossEntropyloss(lable_onehost_y,bad_prefict_y)
good_loss = crossEntropyloss(lable_onehost_y,good_prefict_y)
print(f"\n\n=======================自实现交叉熵")
print(f"bad_loss:{bad_loss:0.4f}   good_loss:{good_loss:0.4f}")

#注意 nn.CrossEntropyLoss 已经包含 LogSoftmax和NLLLoss,所以使用nn.CrossEntropyLoss时不需要自己实现logsoftmax的逻辑了
loss = nn.CrossEntropyLoss()
y_bad_prefict =   torch.tensor([[5 , 2 , 3]],dtype=torch.float32) 
y_good_prefict = torch.tensor([[2 , 3 , 5]],dtype=torch.float32)

_,y_bad_prefict_max =  torch.max(y_bad_prefict,1) 
_,y_good_prefict_max= torch.max(y_good_prefict,1)
print(f"y_bad_prefict_max_index:{y_bad_prefict_max} \ny_good_prefict_max_index:{y_good_prefict_max} ")
y = torch.tensor([2],dtype=torch.long) #第几个值是正确的类别,2表示第3个

bad_loss = loss(y_bad_prefict,y)
good_loss = loss(y_good_prefict,y)
print(f"\n\n=======================以第2项为正确值运算nn.crossEntropyLoss")
print(f"bad_loss:{bad_loss:0.4f}   good_loss:{good_loss:0.4f}")

# 当分类大于2时使用nn.crossEntropyLoss,不需要自己用softmax处理y
class NeunalNet1(nn.Module):
    def __init__(self,n_input,n_hidden,n_classes):
        super(NeunalNet1,self).__init__()
        self.linear1 = nn.Linear(n_input,n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden,n_classes)
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #不需要自己使用softmax计算,这个步骤已经在 nn.crossEntropyLoss中
        return out
model = NeunalNet1(28*28,n_hidden=5,n_classes=3)
loss = nn.CrossEntropyLoss()

# 当分类为2时(是否),使用nn.BCELoss(binary cross entropy loss)时,需要自己使用sigmoix处理y  
class NeunalNet2(nn.Module):
    def __init__(self,n_input,n_hidden):
        super(NeunalNet2,self).__init__()
        self.linear1 = nn.Linear(n_input,n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden,1)
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #二进制交叉熵需要自己使用sigmoid映射y
        out = nn.Sigmoid(out)
        return out
model = NeunalNet2(28*28,n_hidden=5)
loss = nn.BCELoss()