# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:12 2021
@author: Gokulesh Danapal (GX6)
Confidentiality: Internal
"""
import torch
from yolo_backend import train, test, create_dataloader
import os
import time 
from datetime import datetime

def delay():
    cond =  True
    while (cond):
        memory = 0
        a = torch.cuda.list_gpu_processes(1)
        lines = a.splitlines()
        for line in lines:
            words = line.split(' ')
            if len(words) > 4:
                if words[-3] == 'MB':
                    memory += float(words[-4])
        if memory < (49152 - 6543 - 30000):
            cond = False
            print('memory free',49152-6543-memory,'Starting training at',datetime.now().strftime('%H:%M'))
        else:
            #print('memory free',49152-6543-memory, 'Trying again in 10 minutes')
            time.sleep(600)
            
hyp = { 'lr0': 0.01,#0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.2,
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
     
        'box':0.05,  # GIoU loss gain
        'cls': 0.3 , # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 1.0,  # obj loss gain (scale with pixels)
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
        'mosaic': 1.0
     }


class options:
    def __init__(self):
        self.channels = 5  #fusion control: 3 for rgb, 1 for radar_cam, 2 for radar_bev, 5 for fusion, 6 for radar with bev, 7 for radar as points
        #self.weights = '/home/danapalgokulesh/dataset/nuscenes/runs/train_rgb/exp_square_320/weights/best.pt'
        self.weights = '/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt'
        self.splits =  '/home/danapalgokulesh/dataset/nuscenes/splits.pytorch'
        self.root = '/home/danapalgokulesh/dataset/nuscenes/hazed'
        self.names = ['car','pedestrian','barrier','truck','traffic_cone','trailer','bus','construction_vehicle','motorcycle','bicycle']
        self.freeze = []#['neck','head']#['backbone.main1r','backbone.main2r','backbone.main3r','head','neck']#,'backbone.main4v','backbone.main5v']
        self.nc = 10
        self.epochs = 30
        self.batch_size = 32
        self.img_size = 320
        self.device = 'cuda'
        self.device_num = '1'
        self.augment = True
        self.rect = False
        self.rect_test = False
        self.resume = False
        self.nosave = False
        self.noautoanchor = False
        self.cache_images = True
        self.multi_scale = False
        self.adam = False
        self.workers = 8
        self.project = '/home/danapalgokulesh/dataset/nuscenes/runs/train_fusion'
        self.name = 'exp_fog'
        self.evolve = False
        self.notest = False
        self.gradscale = True
        self.distance =  120.1
        self.foggify = 0.0
        
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
if not opt.evolve: 
    #delay()
    results = train(hyp = hyp, opt=opt, train_case = 'train')
    
    # opt.weights = '/home/danapalgokulesh/dataset/nuscenes/runs/train_rgb/exp_cam2/weights/best.pt'
    # opt.project = '/home/danapalgokulesh/dataset/nuscenes/runs/train_rgb'
    # opt.name = 'exp_cam3'
    # opt.channels = 3
    #results = train(hyp = hyp, opt=opt)
    
    #results = test(hyp=hyp, opt=opt)#, test_case =  'train_10')
else:
    import numpy as np
    import warnings
    import sys
    sys.path.append("../")
    from dehb import DEHB  
    seed = 123
    np.random.seed(seed)
    warnings.filterwarnings('ignore')
    min_budget, max_budget = 20 , 60
    import ConfigSpace as CS
    
    def create_search_space(seed=123):
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)
    
        cs.add_hyperparameters([
            #CS.UniformFloatHyperparameter('lr0', lower=1e-5, upper=1e-1),
            #CS.UniformFloatHyperparameter('momentum', lower=0.6, upper=0.98),
            #CS.UniformFloatHyperparameter('weight_decay', lower=0.0, upper=0.001),
            CS.UniformFloatHyperparameter('box', lower=0.02, upper=0.2),
            CS.UniformFloatHyperparameter('cls', lower=0.2, upper=4.0),
            #CS.UniformFloatHyperparameter('cls_pw', lower=0.5, upper=2.0),
            CS.UniformFloatHyperparameter('obj', lower=0.2, upper=4.0),
            # CS.UniformFloatHyperparameter('obj_pw', lower=0.5, upper=2.0),
            # CS.UniformFloatHyperparameter('fl_gamma', lower=0.0, upper=2.0),
            # CS.UniformFloatHyperparameter('hsv_h', lower=0.0, upper=0.1),
            # CS.UniformFloatHyperparameter('hsv_s', lower=0.0, upper=0.9),
            # CS.UniformFloatHyperparameter('hsv_v', lower=0.0, upper=0.9),
            # CS.UniformFloatHyperparameter('degrees', lower=0.0, upper=45),
            # CS.UniformFloatHyperparameter('translate', lower=0.0, upper=0.9),
            # CS.UniformFloatHyperparameter('scale', lower=0.0, upper=0.9),
            # CS.UniformFloatHyperparameter('shear', lower=0.0, upper=10.0),
            # CS.UniformFloatHyperparameter('perspective', lower=0.0, upper=0.001),
            # CS.UniformFloatHyperparameter('flipud', lower=0.0, upper=1.0),
            # CS.UniformFloatHyperparameter('fliplr', lower=0.0, upper=1.0),
            # CS.UniformFloatHyperparameter('mixup', lower=0.0, upper=1.0),
        ])
        return cs
    
    cs = create_search_space(seed)
    dimensions = len(cs.get_hyperparameters())
    dehb = DEHB(f=train,cs=cs,dimensions=dimensions,min_budget=min_budget,max_budget=max_budget,n_workers=1,output_path="./temp")
    
    trajectory, runtime, history = dehb.run(
        brackets=8,
        verbose=True,
        save_intermediate=True,
        # parameters expected as **kwargs in target_function is passed here
        hyp = hyp, opt=opt)
    print(dehb.get_incumbents())
