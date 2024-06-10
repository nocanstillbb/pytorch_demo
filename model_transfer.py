#模型微调demo
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
import time
import os
import copy

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


#"""
## 定义数据转换
#transform = transforms.Compose([
#    transforms.ToTensor()
#])
#
## 加载数据集
#dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#
## 创建数据加载器
#loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#
## 初始化平均值和标准差
#mean = 0.0
#std = 0.0
#
## 计算平均值和标准差
#for images, _ in loader:
#    batch_samples = images.size(0)  # 批量大小
#    images = images.view(batch_samples, images.size(1), -1)  # 将图像展平为向量
#
#    mean += images.mean(2).sum(0)  # 计算每个通道的平均值
#    std += images.std(2).sum(0)  # 计算每个通道的标准差
#
#mean /= len(loader.dataset)  # 计算整个数据集的平均值
#std /= len(loader.dataset)  # 计算整个数据集的标准差
#
#print("Mean:", mean)
#print("Std:", std)
#"""

##平均值
mean = np.array([0.6434, 0.6074, 0.5470])
##标准差
std = np.array([0.2115, 0.2113, 0.2208])

data_transform = {
    'train':torchvision.transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)]),
    'val':torchvision.transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)])
}


image_datasets = { x:datasets.ImageFolder(os.path.join(data_dir,x), data_transform[x]) for x in sets}
image_dataloader = { x:DataLoader(dataset=image_datasets[x],batch_size=20,shuffle=True,num_workers=0) for x in sets}

class_names = image_datasets['train'].classes
print(f"class names : {class_names}")

def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()



def trains_model(model,loss,optimizer,scheduler,n_epoch):
   since = time.time()
   best_model_wtf = copy.deepcopy(model.state_dict())
   best_acc = .0
   
   for epoch in range(n_epoch):
       print(f"epoch:{epoch + 1}/{n_epoch} :")
       print("-"*10)

       for phase in sets:
           if phase == "train":
              model.train()
           else:
              model.eval()
           running_loss = .0
           running_corrects = 0

           dl = image_dataloader[phase]

           for simples,labels in dl:
              simples = simples.to(device)
              labels = labels.to(device)

              with torch.set_grad_enabled(phase == "train"):
                  outputs = model(simples)
                  _,predicted = torch.max(outputs,1)
                  #计算损失函数
                  l = loss(outputs,labels)
                  #如果是训练模式,优化
                  if phase == "train" :
                      optimizer.zero_grad()
                      l.backward()
                      optimizer.step()
              #统计损失和准确率
              running_corrects += (predicted == labels).sum()
              running_loss += l.item() * simples.size(0)
           #每个批次训练完更新学习率
           if phase == "train" :
              scheduler.step()

       epoch_lose = running_loss / len(dl.dataset)
       epoch_acc = running_corrects*100.0 / len(dl.dataset)
       print(f"epoch 准确率{epoch_acc:.4f}%  epoch loss:{epoch_lose:.4f}")

       # 挑选最佳的准确率的模型
       if phase == "val" and  epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_model_wtf = copy.deepcopy(model.state_dict())
           
       print()
   elapsed = time.time() - since
   print(f"trainning complete in :{elapsed//60:.0f}m:{elapsed%60:.0f}s")
   print(f"the best acc is :{best_acc}%")
   
   model.load_state_dict(best_model_wtf)
   return model



#torch < 0.13
#model = models.resnet18(pretrained = True).to(device)
#torch version >= 0.13
model = models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT).to(device)

#如果只想训练最后一层,可以把模型的参数保持住,再重置fc层,这样会提升训练效率,牺牲部分准确率
#for param in model.parameters() :
#    param.requires_grad = False

num_features = model.fc.in_features

#fc: full connection layer
#替换模型最后的全连接层,改变分类结果为2类
model.fc = nn.Linear(num_features,2)
model.to(device)

loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=0.0001)

n_epoch = 1000
#定义学习率调度器,每5个epoch调整学习率为当前的10分之1
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_epoch,eta_min=0.00001) #比较常用的还有cos余弦scheduler
model = trains_model(model,loss,opt,lr_scheduler,n_epoch)


# Get a batch of training data
inputs, classes = next(iter(image_dataloader['val']))
inputs_device = inputs.to(device)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
y = model(inputs_device)
_,predicteds = torch.max(y,1)

for i in range(inputs_device.shape[0]) :
    print(f"第{i}图,预测为:{class_names[predicteds[i]]}   实际为:{class_names[classes[i]]}")

imshow(out, title=[class_names[x] for x in classes])