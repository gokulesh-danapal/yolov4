#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:07:20 2021

@author: danapalgokulesh
"""
import numpy as np
import torch
import torchvision as tv
image =  tv.transforms.ToPILImage()
import cv2,os

root = '/home/danapalgokulesh/dataset/nuscenes/images'
weights = '/home/danapalgokulesh/dataset/nuscenes/runs/train_rgb/exp_cam/weights/best.pt'

anchors_g= np.array([[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]])
from yolo_backend import letterbox
from models import Darknet
model = Darknet(10,anchors_g,channels = 3)
model.load_state_dict(torch.load(weights)['model'])
model.eval()
backbone = model.backbone
neck =  model.neck

splits = torch.load('/home/danapalgokulesh/dataset/nuscenes/splits.pytorch')
rgb_img = letterbox(cv2.imread(os.path.join(root,splits['train'][0])),auto=False)[0]
img =  rgb_img.transpose(2, 0, 1)
input_tensor = torch.from_numpy(img).unsqueeze(0).float()/255.0

#%%
out = neck(backbone(input_tensor))
