# -*- coding: utf-8 -*-
"""
Created on  2021/6/14 16:28

@author: shengrihui
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch5x5_1 =nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2=nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch_pool=nn.Conv2d(in_channels,16,kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch1x1,branch3x3,branch5x5,branch_pool]
        return torch.cat(outputs,dim=1)
    def __len__(self):
        return self.branch1x1.out_channels+\
                self.branch_pool.out_channels+\
                self.branch3x3_3.out_channels+\
                self.branch5x5_2.out_channels

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc1=nn.Linear(525,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3=nn.Linear(128,32)
        self.fc4 = nn.Linear(32, 11)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model,self,).__init__()
        self.covn1=nn.Conv2d(3,10,kernel_size=5)
        self.incep1=InceptionA(in_channels=10)
        self.covn2=nn.Conv2d(len(self.incep1),20,kernel_size=3)
        self.incep2=InceptionA(in_channels=20)

        self.covn3=nn.Conv2d(in_channels=len(self.incep2),out_channels=5,kernel_size=1)
        self.mp=nn.MaxPool2d(2)

        self.fc1 = fc()
        self.fc2 = fc()
        self.fc3=fc()
        self.fc4 = fc()
        self.fc5 = fc()
        self.fc6 = fc()

    def forward(self,x):
        in_size=x.size(0)   #相当于x.shape[0]
        x=F.relu(self.mp(self.covn1(x)))
        x=self.incep1(x)
        x = F.relu(self.mp(self.covn2(x)))
        x = self.incep2(x)
        x=self.mp(self.covn3(x))
        x_shape=x.shape[1]*x.shape[2]*x.shape[3]
        #print(x.shape) #torch.Size([10, 5, 7, 15]) #torch.Size([10, 80, 14, 30])
        #print(x_shape) #525     #33600
        x=x.view(in_size,x_shape)
        c1=self.fc1(x)
        c2 = self.fc2(x)
        c3 = self.fc3(x)
        c4 = self.fc4(x)
        c5 = self.fc5(x)
        c6 = self.fc6(x)
        return c1,c2,c3,c4,c5,c5

'''
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.net1=Net()
        self.net2 = Net()
        self.net3 = Net()
        self.net4 = Net()
        self.net5 = Net()
        self.net6=Net()
    def forward(self,x):
        return self.net1(x),self.net2(x),self.net3(x),\
               self.net4(x),self.net5(x),self.net6(x)
'''

if __name__=='__main__':
    import model_02
    train_loader = model_02.makeLoader('mchar_train')

    model=Model()

    for inputs,labels in train_loader:

        c0,c1,c2,c3,c4,c5=model(inputs)
        print(c0.shape)
        break








