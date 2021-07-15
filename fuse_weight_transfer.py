# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:24:33 2021
@author: TK6YNZ7
"""
import torch
import numpy as np

anchors_g= np.array([[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]])

from models import Darknet
model = Darknet(10,anchors_g,channels = 6)
pre = {}
pre['model'] = model.state_dict()
torch.save(pre,'/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt')
        
fuse = torch.load('/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt')
dest = fuse['model']
yolo =  torch.load('/home/danapalgokulesh/dataset/nuscenes/runs/train_rgb/exp_cam/weights/best.pt')['model']
keys = list(dest.keys())
rad = torch.load('/home/danapalgokulesh/dataset/nuscenes/runs/train_fusion/exp_from_scratch/weights/best.pt')['model']
#%%
vision = []; detector = []; changes = []; radar = [];
for key in keys:
    if 'backbone.main' in key and key[14] == 'v':
        vision.append(key)
    elif 'neck' in key or 'head' in key:
        detector.append(key)
    elif 'backbone.main' in key and key[14] == 'r':
        radar.append(key)
    elif 'backbone.tam' in key:
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
    #pre_key = list(rad.keys())[i]
    if torch.numel(dest[key]) == torch.numel(rad[key]):
        dest[key] = rad[key]
        changes.append([key,key])

changes =  np.array(changes)
print('Number of weights transferred',len(changes))
fuse['epoch'] = -1
fuse['optimizer'] =  None
fuse['best_fitness'] = 0
fuse['training_results'] = ' '
fuse['model'] = dest
torch.save(fuse,'/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt')
