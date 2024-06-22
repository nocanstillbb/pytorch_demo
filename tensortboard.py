# 安装
# python -m pip install tensorboard

# 启动web disboard
# tensorboard --logdir=runs


# 使用卷积网络做实验
# 0. 设置tensorboard writer属性,主要是设置保存路径
# 1. 添加 1 batch 输入图片
# 2. 添加 model graph
# 3. 添加 计算过程中的损失函数和优化器
# 4. 添加 绘制曲线

import torch
import torch.nn as nn
import torch.optim
import torch.utils
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.utils.data
import torchvision
from torchvision  import datasets,models,transforms
import matplotlib.pyplot as plt

import torchvision.transforms as transform
import sys

"""
setp 1 ====================================== tensorboard writer
"""
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/fine-tuning") 

data_dir = "/Users/hbb/Pictures/machine_learning/"
sets = ['train','val']


#选用device,如果有加速优先使用加速
device = None
if torch.cuda.is_available:
    device = torch.device("cuda")
if torch.backends.mps.is_available:
    device = torch.device("mps")
else :
    device = torch.device("cpu")
#device = torch.device("cpu")
print(f"device is {device}")


#超参数
n_epoch = 1
n_batch = 5
learning_rate = 0.001

# Normalize([(通道1平均值,通道2平均值,通道3平均值),(通道1标准差,通道2标准差,通道3标准差)])
transforms = transform.Compose([ transform.ToTensor(), transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ])



#加载训练数据和测试数据
#CIFAR10 : Canadian Institute for Advanced Research 10  包含一个10个分类的图片集
train_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR10",download=True,transform=transforms,train=True)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=n_batch,shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root="./data/CIFAR10",download=False,transform=transforms,train=False)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=n_batch,shuffle=False)

#simples,labels = iter(train_dataloader)._next_data()
#print(f"simple shape:{simples.shape}, lable shape:{labels.shape}")
#
##显示前n_batch张图看看
#for i in range(n_batch) :
#     plt.subplot(2,3,i+1)
#     plt.imshow(simples[i][0])
#plt.show()


#显示一个批次的图片到tensorboard
simples,labels = iter(test_dataloader)._next_data()
print(f"simple shape:{simples.shape}, lable shape:{labels.shape}")
imgs = torchvision.utils.make_grid(simples)

"""
setp 2 ====================================== 输入图像
"""
writer.add_image("one batch of test data",imgs)
#writer.close()
#sys.exit()



classes = ["飞机","轿车","鸟","猫", "鹿","狗","蛙","马","鱼","卡车"]


#定义卷积网络
class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet,self).__init__()
        #卷积后的宽高为  (输入 - 卷积核 + 2*填充 / 步长) +1
        self.conv1 = nn.Conv2d(3,6,5) # 输入3通道,输出6通道,卷积核大小5*5 
        self.pool = nn.MaxPool2d(2,2) # width=height=2, stride = 2 
        self.conv2 = nn.Conv2d(6,16,5) 
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x) :
        #x 为 3 * 32 * 32 的图像集
        out = self.conv1(x) # 卷积1 输入3通道,输出6通道,卷积核5, 输出 6通道,宽=高=32-5+1 = 28
        out = torch.relu(out)  
        out = self.pool(out) #聚合 w=h =2 ,stride = 2 , 输出6通道, w=h=28/2 = 14

        out = self.conv2(out) #卷积2   输入通道6,输出通道16,卷积核5 , 输出 16通道,宽=高=14-5 = 10
        out = torch.relu(out)  
        out = self.pool(out) #继续聚合  输出 16*5*5

        out = out.view(-1,16*5*5)

        out = self.fc1(out) #全连接层1  输入16*5*5 输出120
        out = torch.relu(out)  

        out = self.fc2(out) #全连接层2  输入120 输出84
        out = torch.relu(out)  

        out = self.fc3(out) # 输入84 ,输出10, 分类完成

        return out


#模型
model = ConvolutionalNeuralNet().to(device) # 2层卷积,3层线性全连接层
#损失函数
loss = nn.CrossEntropyLoss()
#优化函数
opt = torch.optim.SGD(model.parameters(),lr=learning_rate)

"""
setp 3 ====================================== model的计算图
"""
writer.add_graph(model, simples.to(device)) #3通道 32*32的图像
#sys.exit()

#循环训练
runing_loss = 0;
runing_correct = 0;
n_total_simples = len(train_dataset)
for epoch in range(n_epoch):
    for i,(simples,labels) in  enumerate(train_dataloader):
        simples = simples.to(device)
        labels = labels.to(device)

        #forward
        y = model(simples)
        #backward
        l = loss(y,labels)
        l.backward()
        #optimizer
        opt.step()
        opt.zero_grad()


        runing_loss += l
        _,predicted = torch.max(y,1)
        runing_correct += (predicted == labels).sum()

        if (i+1)%1000 == 0 :
            print(f"epoch:{epoch+1}/{n_epoch}  step:{i+1}/{n_total_simples/n_batch:.0f}  loss:{l:.4f}")
            """
            setp 4.1 ====================================== 绘制准确率和损失 曲线
            """
            # 由于每1000个样本绘制一个刻度,所以把加起来的损失值和准确率 /1000
            writer.add_scalar("train loss", runing_loss / 1000, epoch * n_total_simples+i )
            runing_loss = 0 
            writer.add_scalar("train acc", runing_correct / 1000/ labels.shape[0], epoch * n_total_simples+i )
            runing_correct = 0

PATH = "./cnn.pat"
torch.save(model.state_dict(),PATH)

class_labels = []
class_preds = []
with torch.no_grad():
    n_simple = 0
    n_crrect = 0

    n_class_simple = [0] * 10
    n_class_crrect = [0] * 10

    for i,(simples,labels) in enumerate(test_dataloader):
        simples = simples.to(device)
        labels = labels.to(device)

        y = model(simples)

        _,predicteds = torch.max(y,1)
        
        n_simple +=  simples.shape[0]
        n_crrect += (labels == predicteds).sum()


        class_labels.append(labels)
        #因为softmax在交叉熵损失函数中,所以y还没有经过激活函数,需要加工一下
        class_probs_batch = [torch.softmax(output, dim=0) for output in y]
        class_preds.append(class_probs_batch) 

        for j in range(simples.shape[0]) :
            lab = labels[j]
            predicted = predicteds[j]
            if(lab == predicted):
                n_class_crrect[lab] += 1
            
            n_class_simple[lab] += 1
    #合并
    #stack dim=0 , 升维
    #cat dim = 0 , 维度不变, 合并多个集
    # class_preds 是独热向量,堆叠起来方便后面取第n列的值
    class_preds = torch.cat([torch.stack(output,dim=0) for output in class_preds])
    class_labels = torch.cat(class_labels)

    acc_total = 100.0 * n_crrect / n_simple
    print(f"总体准确率:{acc_total}%")
    for i in range(10):
        print(f"{classes[i]} 准确率 : {100.0 * n_class_crrect[i] / n_class_simple[i]}%")
        """
        setp 4.2 ====================================== 
        pr_curve 准确度/召回率曲线,用于衡量分类问题的质量
        """
        i_lables = class_labels == i
        i_preds =  class_preds[:,i] # 切片第i列所有行
        writer.add_pr_curve(classes[i],i_lables,i_preds,global_step=0)
        writer.close()



    ##测试前n_batch张图
    #simples,labels = iter(test_dataloader)._next_data()
    #simples2 = simples[:n_batch]
    ##simples2 = copy.deepcopy(simples)
    #simples = simples.to(device)
    #out = model(simples)
    #predicteds_value,predicteds_idx = torch.max(out,1)
    #for i in range(n_batch) :
    #    print(f"第{i+1}张图实际为{classes[labels[i]]}  预测为{classes[predicteds_idx[i]]}")
    #    plt.subplot(2,3,i+1)
    #    plt.imshow(simples2[i][0],cmap="gray")
    #plt.show()

    #为每个分类绘制曲线



