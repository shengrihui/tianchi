# -*- coding: utf-8 -*-
"""
Created on  2021/6/15 13:40
    
@author: shengrihui
"""

import glob,csv
import torch
#from model_02 import *
#from model_02_net import Model
from resnet import *
from PIL import  Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import  time

class test_Dataset(Dataset):
    def __init__(self,img_path, transform):
        self.img_path=img_path

        if transform is not None:
            self.transform=transform
        else:
            self.transform=None
    def __getitem__(self, index):
        img=Image.open(self.img_path[index])
        if self.transform is not None:
            img=self.transform(img)
        name=self.img_path[index][-10:]
        return img,name
    def __len__(self):
        return len(self.img_path)

test_path=glob.glob('../data/mchar_test_a/mchar_test_a/*.png')
test_dataset = test_Dataset(test_path,
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


test_loader=DataLoader(test_dataset,
                       batch_size=1,
                       shuffle=False
                       )

f=open('result_resnet50.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['file_name','file_code'])

a=0
model=Model()
model.load_state_dict(torch.load('model_2_resnet50.pt'))

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
t1=time.time()
results=[]
with torch.no_grad():
    for img,file_name in test_loader:
        img=img.to(device)
        res =model(img)
        pred=[  torch.max( res[i].data ,dim=1 )[1] for i in range(6)     ]
        #print(pred)
        res_str=''
        for i in pred:
            if i.item()!=10:
                res_str+=str(i.item())


        #print(res_str,type(file_name))

        result=[file_name[0],res_str]
        results.append(result)
        a+=1
        print('\r{}'.format(100*a/len(test_dataset)),end='')

results.sort(key=lambda x:x[0])
csv_writer.writerows(results)
f.close()
t2=time.time()
print()
print(t2-t1)


