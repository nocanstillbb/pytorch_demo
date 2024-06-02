#逻辑回归
import torch
import torch.nn as nn
import numpy as np
from  sklearn import datasets
from  sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import train_test_split

#准备训练数据
bc = datasets.load_breast_cancer()
X,y = bc.data ,bc.target
#569个样本,30种特征
n_samples,n_features = X.shape

#按比例分割训练数据和测试数据集
X,X_test,y,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

#逻辑回归之前对特征归一化 
# z = (x - u) / s   u为样本平均值,s为样本标准差
sc = StandardScaler() 
X = sc.fit_transform(X) 
X_test = sc.fit_transform(X_test) 

#numpy 转为 torch tensort
X_train = torch.from_numpy(X.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#y数据集一维数组转换为 二维数据, 行数等于原来一维数组长度,列数为1
y_train = y_train.view(y_train.shape[0],1) 
y_test = y_test.view(y_test.shape[0],1) 

#model 设计
#逻辑回归 仍然使用线性模型,只是在最后使用激活函数sigmoid把输出映射为(0,1)的结果,sigmoid可以视为softmax中只有两种结果的特例
#其他激活函数有线性修正单元ReLU rectifield linear unit ,以及tanh(与simoid相似,只是y向上偏移,图像关于(0,0)对象)
class LogisticRegression(nn.Module):
    def __init__(self,n_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_features,1)

    def forward(self,X_t):
        y_predicted =  torch.sigmoid(self.linear(X_t))
        return y_predicted

model = LogisticRegression(n_features=n_features)

#损失函数
#二进制交叉熵损失函数,softmax的损失函数也是这个, sigmoid激活函数作为softmax的特例,也是使用这个损失函数
#softmax或sigmoid 都是把分量概率相乘得出总概率 两边同时取对数可推导出,损失函数为 f_(i)^(N) y^i - log\hat(Y)
loss = nn.BCELoss() 
#优化方法 仍然使用随机梯度下降
learning_rate =0.03
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
n_iters = 2000
#循环训练
for epoch in range(n_iters):
    #向前传播, 计算损失
    y_predicted = model(X_train)
    l = loss(y_predicted,y_train)
    #向后传播
    l.backward()
    #更新训练参数,清空梯度
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) %10 == 0:
        print(f"epoch:{epoch+1} loss:{l:.4f} ")


# 校验测试数据
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_clas = y_predicted.round()
    acc = y_predicted_clas.eq(y_test).sum() / y_predicted_clas.shape[0]
    print(f"预测结果准确率:{acc}")

        
