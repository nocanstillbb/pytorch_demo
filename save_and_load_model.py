import torch
import torch.nn as nn
import torchvision.models as models

model =  nn.Module() #应该为具体类型的model

PATH = "demo.pth"

#方法一, 保存整个模型,以二进制的形式,不方便做版本控制
#torch.save(model,PATH)
#torch.load(PATH)
#model.eval()#切换到推理模式

#方法二,推荐方式 ,仅保存权重, 更小,且可以替换优化,损失等结构
#可以在训练过程中保存各个阶段的权重, 替换结构反复实验
#且内容可读,可以进行版本控制

torch.save(model.state_dict(),PATH)
model =  nn.Module() #应该为具体类型的model
model.load_state_dict(torch.load(PATH))
#model.load_state_dict(torch.load(PATH,map_location=device)) 
    #如果保存时的device和想要加载到的device不同,则需要指定map_localtion
    #model.to(device=device)
model.eval()#切换到推理模式

for param in model.parameters():
    print(param)


opt = torch.optim.SGD(model.parameters(),lr=0.001)
#方法三, 在方法二的基础上保存更多参数
checkpoint = {
    "epoch": 90,
    "opt":  opt.state_dict(),
    "model" : model.state_dict()
}
torch.save(checkpoint,PATH)

load_checkpoint = torch.load(PATH)
epoch = load_checkpoint["epoch"]
load_opt = torch.optim.SGD.load_state_dict(load_checkpoint["opt"]) 
load_model =  nn.Module() #应该为具体类型的model
load_model.load_state_dict(load_checkpoint['model'])