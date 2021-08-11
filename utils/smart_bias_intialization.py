#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:32:54 2021

@author: danapalgokulesh
"""

import torch
import math
na = 3
nc = 4
strides = [8,16,32]
weight_path = '/home/danapalgokulesh/dataset/dense/yolo_pre_4c.pt'
ckpt =  torch.load(weight_path)
a = ckpt['model']
l = list(a.keys())

biases = []
for param in l:
    if 'final' in param and '.bias' in param:
        biases.append(param)

    
def smart_bias(bias_,stride):
    bias = bias_.view(na, -1)  # shape(3,85)
    #bias[:, 4] += -4.5  # obj
    bias[:, 4] += math.log(8 / (640 / stride) ** 2)  # obj (8 objects per 640 image)
    bias[:, 5:] += math.log(0.6 / (nc - 0.99))  # cls (sigmoid(p) = 1/nc)
    return bias_


for key,stride in zip(biases,strides):
    a[key] =  smart_bias(a[key],stride)
    
ckpt['model']=a
torch.save(ckpt,weight_path)