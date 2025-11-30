import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from torchvision import models
import os
import numpy as np
from utilities import *
from typing import Union


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50',model_path=None, normalize=True):
        super(ResNetFc, self).__init__()
        self.model_resnet = resnet_dict[model_name](pretrained=False)
        if not os.path.exists(model_path):
            model_path = None
            print('invalid model path!')
        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):#继承自torch.nn.Module
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256,  temp=0.05):#in_dim:特征提取后的维度，outdim:类别数，bottle_neck_dim:瓶颈层维度
        super(CLS, self).__init__()#调用父类的初始化方法，继承参数管理、自动求导等
        self.temp = 1#nn.Parameter(torch.ones(1,device='cuda'), requires_grad=True)温度系数，概率分布平滑度
        if bottle_neck_dim:#如果指定了瓶颈层维度
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)#线性变换层，将输入特征映射到瓶颈维度降维，从in_dim到bottle_neck_dim
            self.weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad = True)#学习的参数，微调瓶颈层输出的特征尺度
            self.fc = nn.Linear(bottle_neck_dim, out_dim, bias = False)#线性变换层，将瓶颈特征映射到类别数维度
            
            self.main = nn.Sequential(
                self.bottleneck,#瓶颈层，最终维度256
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),#批量归一化，对256维（batchsize x 256）的特征进行归一化
                    nn.LeakyReLU(0.2, inplace=True),#LeakReLU激活函数，允许小于0的输入有非零输出，inplace=True表示直接在输入上修改
                    self.fc#线性层，最终输出类别数维度
                ),
                nn.Softmax(dim=-1)#对线性层输出的类别进行softmax转换为概率
            )
        else:#没有瓶颈层
            self.fc = nn.Linear(in_dim, out_dim)#直接从输入特征映射到类别数维度
            # if fc_init is not None:
            #     nn.init.constant_(self.fc.weight, fc_init)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):#forward propagation
        out = [x]#初始化一个list，先把特征维度放进去
        
        for i, module in enumerate(self.main.children()):#遍历 self.main 中的所有子模块（按顺序：瓶颈层→BN→LeakyReLU→fc→Softmax）
            if i==0:#第一个模块是瓶颈层
                x = module(x)
                x = x/torch.norm(x, dim =-1,keepdim=True)#对瓶颈层输出的特征进行L2归一化，除以模长，保持特征向量的方向不变，长度为1
            else:
                x = module(x)#其他层直接前向传播
            out.append(x)#把每一层的输出都存到out列表中
       
        out[-2] = out[-2]/ self.temp#softmax之前的线性层做调整，除以温度系数，由于后续是softmax，当温度系数较大时，概率分布更平滑
        out[-1] = nn.Softmax(dim=-1)(out[-2])#重新计算softmax概率分布
        return out#返回所有层的输出列表
    
    def virt_forward(self, K, feature_source, logits: torch.Tensor, target: Union[torch.Tensor, None] = None,  ) -> torch.Tensor:#虚拟类，论文第一大模块：reserve space for unknown classes
        if self.training:#训练模式 feature_source:源域特征 K:目标域特征 logits:线性层输出的类别得分 target:源域标签
            with torch.no_grad():#不计算梯度
                W_yi = torch.gather(self.fc.weight, 0, target.unsqueeze(1).expand(target.size(0), self.fc.weight.size(1)))#按源域真实标签target，从fc层（softmax前的线性层）self.fc.weight取出对应的weights,对于第i个样本，j类取第j行，放到W_yi中第i行
                W_virt = torch.norm(W_yi,dim=1).unsqueeze(-1).unsqueeze(-1) * ((K / torch.norm(K, dim =1).unsqueeze(-1)).unsqueeze(0))#生成虚拟类的权重，并对已知类权重做归一化
            vir = torch.bmm(W_virt, feature_source.unsqueeze(-1)).squeeze(-1)#计算特征和虚拟类权重的点积，得到虚拟类的的得分
            logits = torch.cat([logits, vir], dim=-1)#将虚拟类得分添加到原始类别得分中
            x = nn.Softmax(-1)(logits)#对扩展后的得分做softmax，得到最终类别概率分布
        return x
    

class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x

class LargeAdversarialNetwork(AdversarialNetwork):#对抗网络
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()#调用父类初始化方法
        self.ad_layer1 = nn.Linear(in_feature, 1024)#线性层，将输入特征映射到1024维
        self.ad_layer2 = nn.Linear(1024, 1024)#线性层，1024维到1024维
        self.ad_layer3 = nn.Linear(1024, 1)#线性层，1024维到1维，输出对抗判别结果
        self.sigmoid = nn.Sigmoid()#sigmoid激活函数，将输出映射到0-1之间，表示概率

        self.main = nn.Sequential(
            self.ad_layer1,#线性层，输入特征到1024维
            nn.BatchNorm1d(1024),#批量归一化,对1024维特征进行归一化
            nn.LeakyReLU(0.2, inplace=True),#LeakReLU激活函数
            self.ad_layer2,#线性层，1024维到1024维
            nn.BatchNorm1d(1024),#批量归一化
            nn.LeakyReLU(0.2, inplace=True),#LeakReLU激活函数
            self.ad_layer3,#线性层，1024维到1维
            self.sigmoid#sigmoid激活函数，输出概率
        )#对原来的main进行重新定义，前两个线性层加入了BN和LeakyReLU激活函数


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs
    
class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)
