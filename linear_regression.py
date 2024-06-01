#1.设计模型,  输入,输出,大小,传播
#2.设计损失函数,优化函数
#3.循环训练
#   向前传播 计算预测值
#   向后传播 计算梯度
#   更新权重 损失函数+优化函数

import torch
import torch.nn as nn


X = torch.tensor([1,2,3,4] ,dtype=torch.float32)
Y = torch.tensor([2,4,6,8] ,dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

def forward(x): 
    return  w * x

def loss(y,predict):  
    return ((y - predict)**2).mean()

print(f"训练之前 f(5):{forward(5):.3f}")

n_iters = 100
learning_rate = 1/n_iters


for  epoch in range(n_iters):
    y_pre = forward(X)

    l = loss(Y,y_pre)

    l.backward()
    
    with torch.no_grad():
        w -= w.grad * learning_rate
    
    w.grad.zero_()

    if epoch%10 == 0 :
        print(f"epoch:{ epoch + 1 },  w:{w:.3f}  loss:{l:.8f}")

print(f"训练之后 f(5):{forward(5):.3f}")