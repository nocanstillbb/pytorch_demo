import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import copy


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

#将要把28*28的单灰度图展开平
n_input = 28*28 
#隐藏层输入100
n_hidden = 100
#最终输出10个分类
n_classes = 10
#所有数据训练2轮
n_epoch = 2
#每个批次训练100个样本
n_batch = 100
#训练率
learning_rate = 0.001

#加载训练数据和测试数据
# MNIST(Mixed National Institute of Standards and Technology database) 
train_dataset = torchvision.datasets.MNIST(root="./data/MNIST",download=True,transform=transform.ToTensor(),train=True)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=n_batch,shuffle=True)

test_dataset = torchvision.datasets.MNIST(root="./data/MNIST",download=False,transform=transform.ToTensor(),train=False)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=n_batch,shuffle=False)

simples,labels = iter(train_dataloader)._next_data()
print(f"simple shape:{simples.shape}, lable shape:{labels.shape}")

#显示前6张图看看
for i in range(6) :
     plt.subplot(2,3,i+1)
     plt.imshow(simples[i][0],cmap="gray")
#plt.show()

#两层神网络(输入层不算)
class NeuralNet(nn.Module):
    def __init__(self,n_simple,n_hidden,n_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(n_simple,n_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(n_hidden,n_classes)

    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model  = NeuralNet(n_input,n_hidden, n_classes).to(device)
loss = nn.CrossEntropyLoss()
#op = torch.optim.SGD(model.parameters(),lr=learning_rate)
op = torch.optim.Adam(model.parameters(),lr=learning_rate) #自适应学习率,学习较快,有时会太快导致训练衰减过快训练不佳

total_steps = len(train_dataset) /n_batch
for epoch in range(n_epoch):
    for i,(simples,labels) in enumerate(train_dataloader):
        #展平图片把数据放入硬件加整设备
        simples = simples.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        #向前传播
        y_predicted = model(simples)
        #计算损失
        l = loss(y_predicted,labels)
        #向前传播
        l.backward()
        #优化
        op.step()
        op.zero_grad()

        if (i+1)%100 == 0:
            print(f"epoch:{epoch+1}/{n_epoch}  step:{i+1}/{total_steps}   loss:{l:.4f}")


#测试
with torch.no_grad():
    n_test_simple_totle = 0
    n_test_correct = 0
    for simples,labels in test_dataloader:
        simples = simples.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        out = model(simples)

        #value,index
        _,predictes = torch.max(out,1) #[index] 指示哪一个分类
        n_test_simple_totle += labels.shape[0]
        n_test_correct += (predictes == labels).sum()

    print(f" acc = 100.0 * {n_test_correct} / {n_test_simple_totle}")
    acc = 100.0 * n_test_correct / n_test_simple_totle
    print(f"准确率:{acc:.4f}%")

    #测试前6张图
    simples,labels = iter(train_dataloader)._next_data()
    simples = simples[:6]
    simples2 = copy.deepcopy(simples)
    simples = simples.reshape(-1,28*28).to(device)
    out = model(simples)
    predicteds_value,predicteds_idx = torch.max(out,1)
    for i in range(6) :
        print(f"第{i+1}张图,真实值为{labels[i]}  预测值为{predicteds_idx[i]}")
        plt.subplot(2,3,i+1)
        plt.imshow(simples2[i][0],cmap="gray")
    plt.show()
