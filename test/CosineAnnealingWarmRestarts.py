# -*- coding: utf-8 -*-
"""
Created on  2021/6/18 20:16
    
@author: shengrihui
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import itertools

import matplotlib.pyplot as plt

#初始化学习率
initial_lr = 0.1
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
    def forward(self, x):
        pass

net_1 = model()
optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)

#T_0 用多少轮将学习率调到eta_min
#到eta_min后重新从初始学习率开始，经过T_0*T_mult轮调到eta_min
scheduler_1 = CosineAnnealingWarmRestarts(optimizer_1, T_0=10,T_mult=2,eta_min=0)

print("初始化的学习率：", optimizer_1.defaults['lr'])

lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化

for epoch in range(1, 101):
    # train

    optimizer_1.zero_grad()
    optimizer_1.step()
    if optimizer_1.param_groups[0]['lr']==initial_lr:
        print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    lr_list.append(optimizer_1.param_groups[0]['lr'])
    scheduler_1.step()

# 画出lr的变化
plt.plot(list(range(1, 101)), lr_list)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.title("learning rate's curve changes as epoch goes on!")
plt.show()

