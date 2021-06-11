# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:24:33 2021

@author: TK6YNZ7
"""

import torch
import numpy as np
fuse = torch.load(r"C:\Users\TK6YNZ7\Desktop\Datasets\Dense\yolo_fusion.pt")
yolo_pre =  torch.load(r"C:\Users\TK6YNZ7\Desktop\Datasets\Dense\runs\dehb2\last.pt")#['model']
with open('radar_keys.txt') as f:
    radar_keys = f.readlines() 

with open('backbone_keys.txt') as f:
    bone_keys = f.readlines() 

notkeys = []
for key in radar_keys + bone_keys:
    notkeys.append(str(key).split('\n')[0])
#%%
i = 0
for key in fuse.keys():
    if key not in notkeys:
        if torch.numel(fuse[key]) == torch.numel(yolo_pre[key]):
            fuse[key] = yolo_pre[key]
    elif '3v.' in key:
        if torch.numel(fuse[key]) == torch.numel(yolo_pre[key.replace('3v.','3.')]):
            fuse[key] = yolo_pre[key.replace('3v.','3.')]
    elif '4v.' in key:
        if torch.numel(fuse[key]) == torch.numel(yolo_pre[key.replace('4v.','4.')]):
            fuse[key] = yolo_pre[key.replace('4v.','4.')]
    elif '5v.' in key:
        if torch.numel(fuse[key]) == torch.numel(yolo_pre[key.replace('5v.','5.')]):
            fuse[key] = yolo_pre[key.replace('5v.','5.')]
#%%
import torch
import numpy as np
fuse = torch.load(r"C:\Users\TK6YNZ7\Desktop\Datasets\Dense\yolo_fusion_2.pt")
dest = fuse['model']
yolo =  torch.load(r"C:\Users\TK6YNZ7\Desktop\Datasets\Dense\runs\weights_rgb\best.pt")['model']
keys = list(dest.keys())
#%%
vision = []; detector = []; changes = [];
for key in keys:
    if 'backbone.main' in key and key[14] == 'v':
        vision.append(key)
    elif 'neck' in key or 'head' in key:
        detector.append(key)
        
for i, key in enumerate(vision):
    pre_key = list(yolo.keys())[i]
    if torch.numel(dest[key]) == torch.numel(yolo[pre_key]):
            dest[key] = yolo[pre_key]
            changes.append([key,pre_key])
for key in detector:
    if torch.numel(dest[key]) == torch.numel(yolo[key]):
        dest[key] =  yolo[key]
        changes.append([key,key])
changes =  np.array(changes)
print('Number of weights transferred',len(changes))
fuse['epoch'] = 0
fuse['optimizer'] =  None
fuse['best_fitness'] = 0
fuse['training_results'] = ' '
fuse['model'] = dest
torch.save(fuse,r"C:\Users\TK6YNZ7\Desktop\Datasets\Dense\yolo_fusion_2.pt")

