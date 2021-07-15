#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:57:35 2021

@author: danapalgokulesh
"""

import torch
import os
import torch.distributed as dist
import logging
from models import Darknet
from yolo_backend import create_dataloader, select_device


hyp = { 'lr0': 0.01,#0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.2,
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
     
        'box':0.05,# 0.05,  # GIoU loss gain
        'cls': 0.3,#0.3  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 0.7, #1.0,  # obj loss gain (scale with pixels)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
        'iou_t': 0.6,  # IoU training threshold
        'conf_t':0.001, # Confidence training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
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
        'mosaic': 0.0
     }


class options:
    def __init__(self):
        self.channels = 6  #fusion control: 3 for rgb, 2 for radar, 5 for fusion, 6 for radar as points
        #self.weights = '/home/danapalgokulesh/dataset/nuscenes/runs/train_fusion/exp_from_scratch1/weights/last.pt'
        self.weights = '/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt'
        self.splits =  '/home/danapalgokulesh/dataset/nuscenes/splits.pytorch'
        self.root = '/home/danapalgokulesh/dataset/nuscenes/images'
        self.names = ['car','pedestrian','barrier','truck','traffic_cone','trailer','bus','construction_vehicle','motorcycle','bicycle']
        self.freeze = []#['backbone.main1v','backbone.main2v','backbone.main3v']#,'backbone.main4v','backbone.main5v']
        self.nc = 10
        self.epochs = 30
        self.batch_size = 1
        self.img_size = 640
        self.device = 'cuda'
        self.device_num = '1'
        self.augment = True
        self.rect = False
        self.resume = False
        self.nosave = False
        self.noautoanchor = False
        self.cache_images = True
        self.multi_scale = False
        self.adam = False
        self.workers = 8
        self.project = '/home/danapalgokulesh/dataset/nuscenes/runs/train_fusion'
        self.name = 'exp_from_scratch2'
        self.evolve = False
        self.notest = False
        self.gradscale = False
        
        self.save_json =  False
        self.save_txt = False
        self.save_conf = False
        
        self.image_weights = False
        self.single_cls = False
        self.sync_bn = False
        self.log_imgs = 16
        self.exist_ok = True
        self.bucket = None

opt = options()
opt.gs = 64
logger = logging.getLogger(__name__)
opt.local_rank = -1
opt.total_batch_size = opt.batch_size
opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
# DDP mode
device = select_device(opt.device_num, batch_size=opt.batch_size, logger = logger)
if opt.local_rank != -1:
    assert torch.cuda.device_count() > opt.local_rank
    torch.cuda.set_device(opt.local_rank)
    device = torch.device('cuda', opt.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    opt.batch_size = opt.total_batch_size // opt.world_size
    
dataloader, dataset = create_dataloader(opt.root, opt.img_size, opt.batch_size, opt.gs, opt, split = 'train_10',
                                            hyp=hyp, augment=opt.augment, cache=opt.cache_images, rect=opt.rect,
                                            rank=opt.global_rank, world_size=opt.world_size, workers=opt.workers)
ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
model = Darknet(opt.nc,hyp['anchors_g'],channels = opt.channels, return_att=True).to(device) 
state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
model.load_state_dict(state_dict, strict=False)
model.eval()
print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.weights))  # report
for i, (imgs, targets, paths, _, rads) in enumerate(dataloader):
    if i > 0:
        break;
    else:
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        rads = rads.to(device)
        train,inf,att = model((imgs,rads))
        
    