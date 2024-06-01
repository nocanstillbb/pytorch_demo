# 添加绘图
import torch
import torch.nn as nn
import numpy
from sklearn  import datasets  
import matplotlib.pyplot as plt


#生成回归测试数据
x_numpy ,y_numpy = datasets.make_regression(n_samples=1000,n_features=1,noise=20,random_state=1)
X = torch.from_numpy(x_numpy.astype(numpy.float32))
Y = torch.from_numpy(y_numpy.astype(numpy.float32))
Y = Y.view(Y.shape[0],1) #view和reshape功能类似,但效率更高,因为内存是连续的, 如果内存不是连续的会报错

#定义mode
n_samples,n_featurs = X.shape
print(f"n_sameples:{n_samples}   n_featurs:{n_featurs}")
n_input = n_featurs
n_output = 1
model = nn.Linear(n_input,n_output)

#定义损失函数
loss = nn.MSELoss()

#优化函数
n_iters = 1000
learning_rate = 1/n_iters
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate)

#循环训练
for epoch in range(n_iters):
    #计算预测
    y_predicted = model(X)
    #计算损失
    l = loss(y_predicted,Y)
    #反向传播
    l.backward()
    #优化参数
    optimizer.step()
    optimizer.zero_grad()
    
    w,b = model.parameters()
    
    if (epoch+1) %10 == 0:
        print(f"epoch:{epoch+1} w:{w[0][0]:.4f}   b:{b[0]:0.4f}  l:{l:0.8f}")


#绘制图表
predicteds = model(X).detach().numpy()
plt.plot(x_numpy,y_numpy,"r*")
plt.plot(x_numpy,predicteds,"b")
plt.show()