#1.设计模型,  输入,输出,大小,传播
#2.设计损失函数,优化函数
#3.循环训练
#   向前传播 计算预测值
#   向后传播 计算梯度
#   更新权重 损失函数+优化函数

#使用torch内置的损失函数和优化函数

import torch
import torch.nn as nn



X = torch.tensor([[1],[2],[3],[4]] ,dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]] ,dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)

n_samples, n_features= X.shape
print(f"n_samples={n_samples}, n_features= {n_features}")

input_size = n_features
output_size = n_features

model = nn.Linear(input_size,output_size)
print(f"训练之前 f(5):{model(X_test).item():.3f}")

n_iters = 10000
learning_rate = 1/n_iters

loss = nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)


for  epoch in range(n_iters):
    w,b = model.parameters()

    y_pre = model(X)

    l = loss(Y,y_pre)

    l.backward()  #dl / dw
    
    optimizer.step()
    optimizer.zero_grad()
    

    if epoch%10 == 0 :
        print(f"epoch:{ epoch + 1 },  w:{w[0][0].item():.3f}  b:{b[0].item():.3f}  loss:{l:.8f}")

print(f"训练之后 f(5):{model(X_test).item():.3f}")