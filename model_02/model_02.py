# -*- coding: utf-8 -*-
"""
Created on  2021/6/14 16:16

@author: shengrihui
"""

import glob
import json
import os
import datetime
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from model_02_net import *
from resnet import *
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
        lbl.extend([10] * (6 - len(self.img_lbl[index])))
        lbl = torch.tensor(lbl, dtype=torch.long)
        return img, lbl

    def __len__(self):
        return len(self.img_path)


def makeLoader(file):
    # 保证path和label里的数据一一对应
    data_path = glob.glob('../data/{}/{}/*.png'.format(file, file))
    data_path.sort()
    data_json = json.load(open('../data/{}.json'.format(file)))
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


def train(epoch):
    global train_loader, model, criterion, optimizer, device
    start = time.time()
    cost = 0.0
    lenth=len(train_loader)-1

    for idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = 0.0
        for i in range(6):
            loss += criterion(outputs[i], labels[:, i])
        loss /= 6

        cost += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\repoch:{:d} {:.2f}% |{: <100}|".format(epoch,100*idx/lenth,'*'*int(100*idx/lenth)),end='')

    end = time.time()
    diary = "epoch:{:2d}\tcost:{:.2f}\ttime:{:.2f}\tlr:{:.10f}".format(
        epoch, cost, end - start, optimizer.state_dict()['param_groups'][0]['lr'])
    diary_list.append(diary + '\n')
    print()
    print(diary)


def val(epoch):
    global val_loader, model, criterion, optimizer, device, best
    start = time.time()
    correct = 0
    lenth=len(val_loader)-1
    with torch.no_grad():
        for idx ,(inputs, labels) in enumerate( val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = [torch.max(outputs[i].data, dim=1)[
                             1].view(1, 10) for i in range(6)]
            predicted = torch.cat(tuple(predicted), 0).t()
            # print(predicted)
            # 张量之间的比较运算
            for i in range(10):
                if (predicted[i, :] == labels[i, :]).sum().item() == 6:
                    correct += 1
            print("\rval:{:d} {:.2f}% |{: <100}|".format(epoch, 100 * idx / lenth, '*' * int(100 * idx / lenth)),end='')

        end = time.time()
        diary = "val:{:2d}\tcorrect:{:d}\ttime:{:.2f}".format(
            epoch, correct, end - start)
        diary_list.append(diary + '\n')
        print()
        print(diary)
        if correct > best:
            best = correct
            torch.save(model.state_dict(), model_pt)
            print('save')
            diary_list.append('save\n')



if __name__ == '__main__':

    train_loader = makeLoader('mchar_train')
    val_loader = makeLoader('mchar_val')

    model_vision = 'model_2_resnet50'
    model_pt = f'{model_vision}.pt'
    model = Model()
    if os.path.exists(model_pt):
        model.load_state_dict(torch.load(model_pt))

    lr = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best = 4706
    diary_list = [str(datetime.datetime.now()) + '\n']
    for epoch in range(6):
        # break
        train(epoch)
        scheduler.step()
        if epoch % 3 == 2:
            val(epoch //3)

    with open('{}_diary.txt'.format(model_vision), 'a') as f:
        diary_list.append('\n')
        f.writelines(diary_list)

#