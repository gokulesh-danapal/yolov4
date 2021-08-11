# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:24:33 2021
@author: TK6YNZ7
"""
import torch
import numpy as np

anchors_g= np.array([[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]])

from models import Darknet
model = Darknet(10,anchors_g,channels = 5)
pred = model(torch.zeros((1,7,320,320)))
pre = {}
pre['model'] = model.state_dict()
torch.save(pre,'/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt')
        
fuse = torch.load('/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt')
dest = fuse['model']
yolo =  torch.load('/home/danapalgokulesh/dataset/nuscenes/yolo_pre_4c.pt')['model'] #runs/train_rgb/exp_pre_320/weights/best.pt')['model']
keys = list(dest.keys())
rad = torch.load('/home/danapalgokulesh/dataset/nuscenes/runs/train_radar/exp_cam_coordinates/weights/best.pt')['model']
#%%
vision = []; detector = []; changes = []; radar = [];
for key in keys:
    if 'backbone.main' in key and key[14] == 'v':
       vision.append(key)
    elif 'neck' in key or 'head' in key:
        detector.append(key)
    elif 'backbone.main' in key and key[14] == 'r':
        radar.append(key)
    #elif 'backbone.half' in key:
        #radar.append(key)
        
print('vision' ,len(vision),'detector',len(detector),'radar',len(radar))
for i, key in enumerate(vision):
    pre_key = list(yolo.keys())[i]
    if torch.numel(dest[key]) == torch.numel(yolo[pre_key]):
        dest[key] = yolo[pre_key]
        changes.append([key,pre_key])
print('vision',len(changes))
for key in detector:
    if torch.numel(dest[key]) == torch.numel(yolo[key]):
        dest[key] =  yolo[key]
        changes.append([key,key])
print('detector',len(changes))
for i,key in enumerate(radar):
    #pre_key = list(rad.keys())[i]
    if torch.numel(dest[key]) == torch.numel(rad[key]):
        dest[key] = rad[key]
        changes.append([key,key])
print('radar',len(changes))

changes =  np.array(changes)

print('Number of weights transferred',len(changes))
fuse['epoch'] = -1
fuse['optimizer'] =  None
fuse['best_fitness'] = 0
fuse['training_results'] = ' '
fuse['model'] = dest
torch.save(fuse,'/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt')
