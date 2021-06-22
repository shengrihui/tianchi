# -*- coding: utf-8 -*-
"""
Created on  2021/6/16 20:48
    
@author: shengrihui
"""

from torchvision.models.resnet import resnet50, resnet34,resnet18
from torch import nn as nn
from torch.nn import functional as F

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc0=nn.Linear(1000,512)
        self.fc1=nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3=nn.Linear(128,32)
        self.fc4 = nn.Linear(32, 11)
    def forward(self,x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net=resnet50()
        self.fc1 = fc()
        self.fc2 = fc()
        self.fc3 = fc()
        self.fc4 = fc()
        self.fc5 = fc()
        self.fc6 = fc()

    def forward(self, x):
        x=self.net(x)
        c1 = self.fc1(x)
        c2 = self.fc2(x)
        c3 = self.fc3(x)
        c4 = self.fc4(x)
        c5 = self.fc5(x)
        c6 = self.fc6(x)
        return c1, c2, c3, c4, c5, c5
if __name__=='__main__':
    import model_02
    train_loader = model_02.makeLoader('mchar_train')

    model=Model()
    for inputs,labels in train_loader:
        #c0=model(inputs)
        c0,c1,c2,c3,c4,c5=model(inputs)
        print(c0.shape)
        break