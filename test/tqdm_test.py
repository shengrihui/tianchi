# -*- coding: utf-8 -*-
"""
Created on  2021/6/21 15:40
    
@author: shengrihui
"""

from tqdm import tqdm
from tqdm import trange
s=1
for i in tqdm(range(1,100000)):
   s*=i

for i in trange(100000000):
   pass