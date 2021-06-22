# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 20:40:23 2021

@author: 11200
"""

import numpy as np
import glob,json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print(torch.cuda.is_initialized())
print(torch.cuda.device_count())
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            # 原始SVHN中类别10为数字0
            lbl = np.array(self.img_label[index], dtype=np.int64)
            lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))
    def __len__(self):
        return len(self.img_path)
train_path = glob.glob('../data/mchar_train/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('../data/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]



train_loader = DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=10, # 每批样本个数
        shuffle=False, # 是否打乱顺序
        num_workers=2, # 读取的线程个数
        )




class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
        #
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6


if __name__ == '__main__':
    model = SVHN_Model1()
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器器
    optimizer = torch.optim.Adam(model.parameters(), 0.005)
    loss_plot, c0_plot = [], []
    # 迭代10个Epoch
    t1=time.time()
    for epoch in range(10):
        for idx,data in enumerate(train_loader):
            data[0],data[1]=data[0].to(device),data[1].to(device)
            c0, c1, c2, c3, c4, c5 = model(data[0])
            loss = criterion(c0, data[1][:, 0]) + \
            criterion(c1, data[1][:, 1]) + \
            criterion(c2, data[1][:, 2]) + \
            criterion(c3, data[1][:, 3]) + \
            criterion(c4, data[1][:, 4]) 
            loss /= 6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_plot.append(loss.item())
            c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item()*1.0 / c0.shape[0])
            if idx%500==0:
                print(epoch,idx,loss.item())
        t2=time.time()
        print(t2-t1)
        