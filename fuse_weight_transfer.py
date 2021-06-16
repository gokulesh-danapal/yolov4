# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:24:33 2021

@author: TK6YNZ7
"""
import torch
import numpy as np

anchors_g= np.array([[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]])

from yolo_backend import MAFnet
model = MAFnet(4,anchors_g)
pre = {}
pre['model'] = model.state_dict()
torch.save(pre,'/home/danapalgokulesh/dataset/dense/yolo_fusion2.pt')
        
fuse = torch.load('/home/danapalgokulesh/dataset/dense/yolo_fusion2.pt')
dest = fuse['model']
yolo =  torch.load('/home/danapalgokulesh/dataset/dense/runs_rgb_320/weights/last.pt')['model']
keys = list(dest.keys())
rad = torch.load('/home/danapalgokulesh/dataset/dense/radar_tiny.pt')['model']
#%%
vision = []; detector = []; changes = []; radar = [];
for key in keys:
    if 'backbone.main' in key and key[14] == 'v':
        vision.append(key)
    elif 'neck' in key or 'head' in key:
        detector.append(key)
    if 'backbone.main' in key and key[14] == 'r':
        radar.append(key)
        
for i, key in enumerate(vision):
    pre_key = list(yolo.keys())[i]
    if torch.numel(dest[key]) == torch.numel(yolo[pre_key]):
        dest[key] = yolo[pre_key]
        changes.append([key,pre_key])
for key in detector:
    if torch.numel(dest[key]) == torch.numel(yolo[key]):
        dest[key] =  yolo[key]
        changes.append([key,key])
for i,key in enumerate(radar):
    pre_key = list(rad.keys())[i]
    if torch.numel(dest[key]) == torch.numel(rad[pre_key]):
        dest[key] = rad[pre_key]
        changes.append([key,pre_key])

changes =  np.array(changes)
print('Number of weights transferred',len(changes))
fuse['epoch'] = 0
fuse['optimizer'] =  None
fuse['best_fitness'] = 0
fuse['training_results'] = ' '
fuse['model'] = dest
torch.save(fuse,'/home/danapalgokulesh/dataset/dense/yolo_fusion2.pt')

