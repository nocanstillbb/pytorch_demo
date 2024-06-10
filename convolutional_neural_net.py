import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt


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
n_epoch = 4
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

simples,labels = iter(train_dataloader)._next_data()
print(f"simple shape:{simples.shape}, lable shape:{labels.shape}")

#显示前n_batch张图看看
#for i in range(n_batch) :
#     plt.subplot(2,3,i+1)
#     plt.imshow(simples[i][0])
#plt.show()



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

""" 稠密连接卷积
#def conv_block(input_channels, num_channels):
#    return nn.Sequential(
#        nn.BatchNorm2d(input_channels), nn.ReLU(),
#        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
#
#class DenseBlock(nn.Module):
#    def __init__(self, num_convs, input_channels, num_channels):
#        super(DenseBlock, self).__init__()
#        layer = []
#        for i in range(num_convs):
#            layer.append(conv_block(
#                num_channels * i + input_channels, num_channels))
#        self.net = nn.Sequential(*layer)
#
#    def forward(self, X):
#        for blk in self.net:
#            Y = blk(X)
#            # 连接通道维度上每个块的输入和输出
#            X = torch.cat((X, Y), dim=1)
#        return X
#def transition_block(input_channels, num_channels):
#    return nn.Sequential(
#        nn.BatchNorm2d(input_channels), nn.ReLU(),
#        nn.Conv2d(input_channels, num_channels, kernel_size=1),
#        nn.AvgPool2d(kernel_size=2, stride=2))
#
#b1 = nn.Sequential(
#    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
#    nn.BatchNorm2d(64), nn.ReLU(),
#    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
## num_channels为当前的通道数
#num_channels, growth_rate = 64, 32
#num_convs_in_dense_blocks = [4, 4, 4, 4]
#blks = []
#for i, num_convs in enumerate(num_convs_in_dense_blocks):
#    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
#    # 上一个稠密块的输出通道数
#    num_channels += num_convs * growth_rate
#    # 在稠密块之间添加一个转换层，使通道数量减半
#    if i != len(num_convs_in_dense_blocks) - 1:
#        blks.append(transition_block(num_channels, num_channels // 2))
#        num_channels = num_channels // 2
#
#model = nn.Sequential(
#    b1, *blks,
#    nn.BatchNorm2d(num_channels), nn.ReLU(),
#    nn.AdaptiveAvgPool2d((1, 1)),
#    nn.Flatten(),
#    nn.Linear(num_channels, 10)).to(device)
"""

#模型
model = ConvolutionalNeuralNet().to(device) # 2层卷积,3层线性全连接层
#损失函数
loss = nn.CrossEntropyLoss()
#优化函数
opt = torch.optim.SGD(model.parameters(),lr=learning_rate)

#循环训练
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
        if (i+1)%1000 == 0 :
            print(f"epoch:{epoch+1}/{n_epoch}  step:{i+1}/{n_total_simples/n_batch:.0f}  loss:{l:.4f}")

PATH = "./cnn.pat"
torch.save(model.state_dict(),PATH)
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

        for j in range(simples.shape[0]) :
            lab = labels[j]
            predicted = predicteds[j]
            if(lab == predicted):
                n_class_crrect[lab] += 1
            
            n_class_simple[lab] += 1

    acc_total = 100.0 * n_crrect / n_simple
    print(f"总体准确率:{acc_total}%")
    for i in range(10):
        print(f"{classes[i]} 准确率 : {100.0 * n_class_crrect[i] / n_class_simple[i]}%")

    #测试前n_batch张图
    simples,labels = iter(train_dataloader)._next_data()
    simples2 = simples[:n_batch]
    #simples2 = copy.deepcopy(simples)
    simples = simples.to(device)
    out = model(simples)
    predicteds_value,predicteds_idx = torch.max(out,1)
    for i in range(n_batch) :
        print(f"第{i+1}张图实际为{classes[labels[i]]}  预测为{classes[predicteds_idx[i]]}")
        plt.subplot(2,3,i+1)
        plt.imshow(simples2[i][0],cmap="gray")
    plt.show()



