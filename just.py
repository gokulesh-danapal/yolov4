#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:45:14 2021

@author: danapalgokulesh
"""

import torch
import cv2, os
from utils.foggify import fogify
import matplotlib.pyplot as plt
root = '/home/danapalgokulesh/dataset/nuscenes/images'
splits =  torch.load('/home/danapalgokulesh/dataset/nuscenes/splits.pytorch')['val']
troots = ['hazing/ten','hazing/fifteen','hazing/twenty','hazing/twenty_five','hazing/thirty','hazing/thirty_five','hazing/forty']
betas = [0.01,0.015,0.02,0.025,0.03,0.035,0.04]
for troot,beta in zip(troots,betas):
    for file in splits:
        img =  cv2.imread(os.path.join(root,file))
        depth =  cv2.imread(os.path.join(root.replace('images','depth'),file.replace('.png','.jpg')))[:,:,0]
        img = fogify(img,depth,beta)
        cv2.imwrite(os.path.join(root.replace('images',troot),file), img)
        #plt.imshow(img)