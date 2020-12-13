# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : 汪逢生
# @FILE     : main.py
# @Time     : 2020-12-13 下午 1:47
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
import timeit
from PIL import Image
import os
import numpy as np
import scipy.io
import torchvision.models.inception as inception

# data load
# pytorch dataloader 作为数据的加载器，该加载器能够根据设定在每一次请求时自动加载一批训练数据
# 自主实现多线程加载，快速加载的同时节省内存开销

# dataloader 加载的数据必须是pytorch 的Dataset类，第一步，将数据封装成Dataset类

label_mat = scipy.io.loadmat('./data/q3_2_data.mat')
label_train = label_mat['trLb']
label_val = label_mat['valLb']

class ActionDataset(Dataset):
    def __init__(self,root_dir,labels=[],transformer=None):
        self.root_dir = root_dir
        self.length = len(os.listdir(root_dir))
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return self.length*3

    def __getitem__(self, idx):
        folder = idx//3+1
        imidx = idx%3+1
        # 输出宽度至少为5
        folder = format(folder,'05d')
        imgname = str(imidx)+'.jpg'
        img_path = os.path.join(self.root_dir,folder,imgname)
        image = Image.open(img_path)
        if len(self.labels)!=0:
            Label = self.labels[idx//3][0]-1
        if self.transformer:
            image =self.transformer(image)
        if len(self.labels)!=0:
            sample = {'image':image,'img_path':img_path,'Label':Label}
        else:
            sample = {'image': image, 'img_path': img_path}
        return sample

image_dataset = ActionDataset(root_dir='./data/trainClips/',labels=label_train,transformer=T.ToTensor())
# for i in range(3):
#     sample = image_dataset[i]
#     print(sample['image'].shape)
#     print(type(sample['Label']))
#     print(sample['img_path'])


# dataloader实现了
# 1次加载batchsize大小的数据 按顺序从里面取
# 打乱数据的顺序
# 多线程加载数据

# image_dataloader = DataLoader(image_dataset,batch_size=4,shuffle=True,num_workers=4)
# def data():
#     for i,sample in enumerate(image_dataloader):
#         print(len(sample))
#         sample['image'] = sample['image']
#         print(i,sample['image'].shape)
#         if i>5:
#             break

image_dataset_train = ActionDataset(root_dir='./data/trainClips/',labels=label_train,transformer=T.ToTensor())
image_dataloader_train = DataLoader(image_dataset_train,batch_size=32,shuffle=True,num_workers=4)
image_dataset_val = ActionDataset(root_dir='./data/valClips/',labels=label_val,transformer=T.ToTensor())
image_dataloader_val = DataLoader(image_dataset_val,batch_size=32,shuffle=True,num_workers=4)
image_dataset_test = ActionDataset(root_dir='./data/testClips/',labels=[],transformer=T.ToTensor())
image_dataloader_test = DataLoader(image_dataset_test,batch_size=32,shuffle=True,num_workers=4)

#pytorch支持CPU数据类型的浮点数类型
dtype = torch.FloatTensor
print_every = 100
# 控制loss 的打印频率

def reset(m):#参数的初始化
    if  hasattr(m,'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x.view(N,-1) #将除N之外的维度全部展开

fixed_model_base = nn.Sequential(
    nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2,stride=2),
    nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2,stride=2),
    Flatten(),
    nn.ReLU(inplace=True),
    nn.Linear(32*16*16,10)
)


fixed_model = fixed_model_base.type(dtype)
#
def test():
    x = torch.randn(1,3,64,64).type(dtype)
    x_var = Variable(x.type(dtype))
    ans = fixed_model(x_var)
    print(ans.shape)

# train
# 将训练数据输入模型开始前向传播
# 计算loss值
# 根据loss值反向传播,使用优化器更新模型参数

# 学习率，动量，学习率变化
# pytorch 基于动态图模型，具有自动求导的过程，前向传播的过程保留各个变量的梯度简化手动求导的过程
# 数据输入模型的到输出
# 根据输出和标签计算loss
# optimizer.zero_grad()优化器梯度归零
# loss.backward()反向传播
# optimizer.step() 更新参数

def check_accuracy(model,loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for t,sample in enumerate(loader):
        x_var = Variable(sample['image'])
        y_var = sample['Label']
        scores = model(x_var)
        _,preds = scores.data.max(1)

        num_correct += (preds.numpy() == y_var.numpy()).sum()
        num_samples += preds.size(0)
    acc = float(num_correct)/num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct,num_samples,acc))


def train(model,loss_fn,optimizer,dataloader,num_epochs=1):
    for epoch in range(num_epochs):
        print('starting epoch %d / %d' % (epoch+1,num_epochs))
        check_accuracy(model,image_dataloader_val)
        model.train()
        for t,sample in enumerate(dataloader):
            x_var = Variable(sample['image'])
            y_var = Variable(sample['Label'].long())

            score = model(x_var)
            loss = loss_fn(score,y_var)
            if (t+1) % print_every ==0:
                print_every('t=%d,loss=%.4f' % (t+1,loss.item))
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    # data()
    # test()
    optimizer = torch.optim.RMSprop(fixed_model_base.parameters(),lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    torch.random.manual_seed(5441)
    fixed_model.cpu()
    fixed_model.apply(reset)
    fixed_model.train()
    train(fixed_model,loss_fn,optimizer,image_dataloader_train,1)
    # pass













