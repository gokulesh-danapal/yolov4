#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:15:28 2021

@author: danapalgokulesh
"""

import torch
from yolo_backend import create_dataloader, non_max_suppression
from models import Darknet
class options:
    def __init__(self):
        self.channels = 5  #fusion control: 3 for rgb, 1 for radar_cam, 2 for radar_bev, 5 for fusion, 6 for radar with bev, 7 for radar as points
        self.weights = '/home/danapalgokulesh/dataset/nuscenes/runs/train_fusion/exp_square_320/weights/best.pt'
        #self.weights = '/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt'
        self.splits =  '/home/danapalgokulesh/dataset/nuscenes/splits.pytorch'
        self.root = '/home/danapalgokulesh/dataset/nuscenes/images'
        self.names = ['car','pedestrian','barrier','truck','traffic_cone','trailer','bus','construction_vehicle','motorcycle','bicycle']
        self.nc = 10
        self.batch_size = 1
        self.img_size = 320
        self.device = 'cuda:0'
        self.augment = False
        self.rect = False
        self.rect_test = True
        self.cache_images = True
        self.workers = 8
        self.single_cls = False
opt = options()

hyp = { 'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'iou_t': 0.6,  # IoU training threshold
        'conf_t':0.001, # Confidence training threshold
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0.1,  # image translation (+/- fraction)
        'scale': 0.9,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mixup': 0.0, #mix up probability
        'mosaic': 1.0 }

testloader = create_dataloader(opt.root, opt.img_size, opt.batch_size, 64, opt, split = 'train_10', hyp=hyp, cache=opt.cache_images, rect=opt.rect_test,
                                       rank=-1, world_size=1, workers=opt.workers,pad=0.5)[0]  # testloader

device = opt.device
model = Darknet(opt.nc,hyp['anchors_g'],channels = opt.channels).to(device)
model.load_state_dict(torch.load(opt.weights)['model'], strict=False)
model.eval()
for i, (img, targets, paths, shapes,rads) in enumerate(testloader):
    img = img.to(device, non_blocking=True)
    img = img.float()/255.0
    with torch.no_grad():
          inf_out, train_out = model(img)
          output = non_max_suppression(inf_out, conf_thres=hyp['conf_t'], iou_thres=hyp['iou_t'])