# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 14:11:01 2021

@author: 11200
"""

import glob
import json
import os
import datetime
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import time
import pprint

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class dataset(Dataset):
    def __init__(self, path, lbl, transform):
        self.img_path = path
        self.img_lbl = lbl
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        if self.transform is not None:
            img = self.transform(img)
        lbl = self.img_lbl[index]
        lbl.extend([10]*(6-len(self.img_lbl[index])))
        lbl = torch.tensor(lbl, dtype=torch.long)
        return img, lbl

    def __len__(self):
        return len(self.img_path)


def makeLoader(file):
    # 保证path和label里的数据一一对应
    data_path = glob.glob('data/{}/{}/*.png'.format(file, file))
    data_path.sort()
    data_json = json.load(open('data/{}.json'.format(file)))
    data_json = sorted(data_json.items(), key=lambda x: x[0])
    data_labels = [v['label'] for k, v in data_json]

    data_dataset = dataset(data_path, data_labels,
                           transform=transforms.Compose([
                               # 缩放到固定尺⼨寸
                               transforms.Resize((64, 128)),
                               # 随机颜⾊色变换
                               transforms.ColorJitter(0.2, 0.2, 0.2),
                               # 加⼊入随机旋转
                               transforms.RandomRotation(5),
                               # 将图⽚片转换为pytorch 的tesntor
                               transforms.ToTensor(),
                               # 对图像像素进⾏行行归⼀一化
                               transforms.Normalize([0.485, 0.456, 0.406], [
                                                    0.229, 0.224, 0.225])
                           ]))

    data_loader = DataLoader(data_dataset,
                             batch_size=10,
                             shuffle=True,
                             num_workers=2)
    return data_loader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #输入torch.Size([10, 3, 64, 128])
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6


def train(epoch):
    global train_loader, model, criterion, optimizer, device,schedule
    start = time.time()
    cost = 0.0
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = 0.0
        for i in range(6):
            loss += criterion(outputs[i], labels[:, i])
        loss /= 6
        '''
        if idx==2500:
            #print(lbl)
            #print(lbl[0])
            #print(outputs[0])
            #print(criterion(outputs[5],lbl[:,5]))
            for i in range(6):
                print(lbl[:,i])
                _,predicted=torch.max(outputs[i].data,dim=1)
                print(predicted)
                print()
            break
        '''
        cost += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if idx%500==0:
        # print(epoch,idx,loss.item())
    end = time.time()
    diary = "epoch:{:2d}\tcost:{:.2f}\ttime:{:.2f}\tlr:{:.10f}".format(
        epoch, cost, end-start, optimizer.state_dict()['param_groups'][0]['lr'])
    diary_list.append(diary+'\n')
    print(diary)


def val(epoch):
    global val_loader, model, criterion, optimizer, device
    start = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # print(outputs[0].shape)
            predicted = [torch.max(outputs[i].data, dim=1)[1].view(1, 10) for i in range(6)]
            predicted = torch.cat(tuple(predicted), 0).t()
            # print(predicted)
            # 张量之间的比较运算
            for i in range(10):
                if (predicted[i, :] == labels[i, :]).sum().item() == 6:
                    correct += 1
            total += labels.shape[0]
        end = time.time()
        diary = "val:{:2d}\tcorrect:{:d}\ttime:{:.2f}".format(
            epoch, correct, end-start)
        diary_list.append(diary+'\n')
        print(diary)


if __name__ == '__main__':

    train_loader = makeLoader('mchar_train')
    val_loader = makeLoader('mchar_val')

    '''
    #train_loader的每一个是一个list
    #[torch.Size([10, 3, 64, 128]),torch.Size([10, 6])
    #第一个是图，(batch_size,channels,h,w)，
    #第二个s是标签，(batch_size,个数)。
    
    for data in train_loader:
        print(data)
        print(type(data[0]))
        print(data[0].shape)
        
        print(data[1].shape)
        break
    '''

    model_vision = 'model_0'
    model_pt = f'{model_vision}.pt'
    model = Model()
    if os.path.exists(model_pt):
        model.load_state_dict(torch.load(model_pt))
    lr = 0.002
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(
        #optimizer, step_size=6, gamma=0.25)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)

    diary_list = [str(datetime.datetime.now())+'\n']
    for epoch in range(15):
        # break
        train(epoch)
        #scheduler.step()
        if epoch % 3 == 2:
            val(epoch+1)

    with open('{}_diary.txt'.format(model_vision), 'a') as f:
        diary_list.append('\n')
        f.writelines(diary_list)
    torch.save(model.state_dict(), model_pt)
#
