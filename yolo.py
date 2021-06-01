# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:12 2021
@author: Gokulesh Danapal (GX6)
Confidentiality: Internal
"""
import torch
import os
from yolo_backend import Dataset, Darknet, train, test, train_hpo, cache
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune.schedulers.pb2 import PB2


hyp = { 'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda    
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 4, #Number of classes
        'names' : ['person', 'RidableVehicle', 'car', 'LargeVehicle'],
        'gs': 32, #Image size multiples
        'img_size': 640, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 30, #number of epochs
        'batch_size': 16, #train batch size
        'test_size': 1, #test batch size
        'use_adam': False, #Bool to use Adam optimiser
        'use_ema': True, #Exponential moving average control
        'multi_scale': False, #Bool to do multi-scale training
        'test_all': True, #Run test after end of each epoch
        'save_all': True, #Save checkpoints after every epoch
        'auto_anchor': False, #create new anchors by Kmeans clustering
        
        'giou': 0.05,  # GIoU loss gain
        'cls': 0.025,  # cls loss gain
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
        'translate': 0.0,  # image translation (+/- fraction)
        'scale': 0.5,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mixup': 0.0, #mix up probability
        
        # 'anchors_g': [[9,82],
        #             [16,146],
        #             [30, 82],
        #             [ 24,224],
        #             [ 53,136],
        #             [ 42,363],
        #             [96,237],
        #             [108,885],
        #             [219,549]]
     }

# #%%   
config={'momentum': 0.865,  # SGD momentum/Adam beta1
        'weight_decay': 0.00063,  # optimizer weight decay}
        'giou': 0.01,  # GIoU loss gain
        'cls': 0.9,  # cls loss gain
        'cls_pw': 1.4,  # cls BCELoss positive_weight
        'obj': 3.26,  # obj loss gain (scale with pixels)
        'obj_pw': 0.74,  # obj BCELoss positive_weight
        'anchor_t': 3.9,  # anchor-multiple threshold 
        'fl_gamma': 0.53,  # focal loss gamma (efficientDet default gamma=1.5)
        }

# for key in list(config.keys()):
#     hyp[key] = config[key]
    
#weight_path = '/home/danapalgokulesh/dataset/dense/hpo/'
weight_path = '/home/danapalgokulesh/dataset/dense/runs/weights/last_025.pt'
imroot = '/home/danapalgokulesh/dataset/dense/images_640'
lroot = '/home/danapalgokulesh/dataset/dense/labels'
logdir = '/home/danapalgokulesh/dataset/dense/runs'
split_path = '/home/danapalgokulesh/dataset/dense/splits.pytorch'
splits = torch.load(split_path)


# test_clear_simple = []
# for image in splits['test_clear']:
#     f = open(os.path.join(lroot,image.replace('.png','.txt')))
#     if len(f.readlines()) < 4:
#         test_clear_simple.append(image)
        
# weight_path = r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\yolo_pre.pt"
# imroot = r'E:\Datasets\Dense\cam_stereo_left_lut'
# logdir = r'E:\Datasets\Dense\runs'
# lroot = r'E:\Datasets\Dense\labels_4'
# splits = torch.load(r'E:\Datasets\Dense\splits.pytorch')

#dicts = cache(imroot,lroot,splits['train_10'],hyp['img_size'])
# dicts = cache(imroot,lroot,splits['train']+splits['val'],hyp['img_size'])
# train_set = Dataset(hyp,dicts, splits['train'],augment=True)#, splits =  splits['image_weights'])
# val_set = Dataset(hyp,dicts, splits['val'], augment= False)
# tb_writer = SummaryWriter(log_dir = logdir)
# results = train(hyp,tb_writer, train_set, weight_path, val_set)

dicts = cache(imroot,lroot,splits['test'],hyp['img_size'])
test_set = Dataset(hyp,dicts, splits['test'], augment= False)
results = test(test_set,hyp,weight_path,plot_all = False)

# import numpy as np
# import warnings
# import sys
# sys.path.append("../")
# from dehb import DEHB

# seed = 123
# np.random.seed(seed)
# warnings.filterwarnings('ignore')
# min_budget, max_budget = 5, 50
# import ConfigSpace as CS


# def create_search_space(seed=123):
#     """Parameter space to be optimized --- contains the hyperparameters
#     """
#     cs = CS.ConfigurationSpace(seed=seed)

#     cs.add_hyperparameters([
#         #CS.UniformFloatHyperparameter('lr0', lower=1e-5, upper=1e-1),
#         CS.UniformFloatHyperparameter('momentum', lower=0.6, upper=0.98),
#         CS.UniformFloatHyperparameter('weight_decay', lower=0.0, upper=0.001),
#         CS.UniformFloatHyperparameter('giou', lower=0.02, upper=0.2),
#         CS.UniformFloatHyperparameter('cls', lower=0.2, upper=4.0),
#         CS.UniformFloatHyperparameter('cls_pw', lower=0.5, upper=2.0),
#         CS.UniformFloatHyperparameter('obj', lower=0.2, upper=4.0),
#         CS.UniformFloatHyperparameter('obj_pw', lower=0.5, upper=2.0),
#         CS.UniformFloatHyperparameter('anchor_t', lower=2.0, upper=8.0),
#         CS.UniformFloatHyperparameter('fl_gamma', lower=0.0, upper=2.0),
#         # CS.UniformFloatHyperparameter('hsv_h', lower=0.0, upper=0.1),
#         # CS.UniformFloatHyperparameter('hsv_s', lower=0.0, upper=0.9),
#         # CS.UniformFloatHyperparameter('hsv_v', lower=0.0, upper=0.9),
#         # CS.UniformFloatHyperparameter('degrees', lower=0.0, upper=45),
#         # CS.UniformFloatHyperparameter('translate', lower=0.0, upper=0.9),
#         # CS.UniformFloatHyperparameter('scale', lower=0.0, upper=0.9),
#         # CS.UniformFloatHyperparameter('shear', lower=0.0, upper=10.0),
#         # CS.UniformFloatHyperparameter('perspective', lower=0.0, upper=0.001),
#         # CS.UniformFloatHyperparameter('flipud', lower=0.0, upper=1.0),
#         # CS.UniformFloatHyperparameter('fliplr', lower=0.0, upper=1.0),
#         # CS.UniformFloatHyperparameter('mixup', lower=0.0, upper=1.0),
#     ])
#     return cs

# cs = create_search_space(seed)
# dimensions = len(cs.get_hyperparameters())
# dehb = DEHB(f=train,cs=cs,dimensions=dimensions,min_budget=min_budget,max_budget=max_budget,n_workers=1,output_path="./temp")
# dicts = cache(imroot,lroot,splits['train']+splits['val'],hyp['img_size'])
# #dicts = cache(imroot,lroot,splits['train_10'],hyp['img_size'])
# train_set = Dataset(hyp,dicts, splits['train'],augment=False)#, splits =  splits['image_weights'])
# val_set = Dataset(hyp,dicts, splits['val'], augment= False)
# tb_writer = SummaryWriter(log_dir = logdir)

# trajectory, runtime, history = dehb.run(
#     total_cost=86400*3,
#     verbose=True,
#     save_intermediate=True,
#     # parameters expected as **kwargs in target_function is passed here
#     tb_writer = tb_writer,
#     hyp = hyp,
#     dataset = train_set,
#     test_set = val_set,
#     checkpoint_dir = weight_path)
# print(dehb.get_incumbents())


# # #%%   
# config={'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
#         'momentum': 0.937,  # SGD momentum/Adam beta1
#         'weight_decay': 0.0005,  # optimizer weight decay}
#         'giou': 0.05,  # GIoU loss gain
#         'cls': 0.025,  # cls loss gain
#         'cls_pw': 1.0,  # cls BCELoss positive_weight
#         'obj': 1.0,  # obj loss gain (scale with pixels)
#         'obj_pw': 1.0,  # obj BCELoss positive_weight
#         'anchor_t': 4.0,  # anchor-multiple threshold 
#         'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
#         'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
#         'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
#         'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
#         'degrees': 0.0,  # image rotation (+/- deg)
#         'translate': 0.0,  # image translation (+/- fraction)
#         'scale': 0.5,  # image scale (+/- gain)
#         'shear': 0.0,  # image shear (+/- deg)
#         'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
#         'flipud': 0.0,  # image flip up-down (probability)
#         'fliplr': 0.5,  # image flip left-right (probability)
#         'mixup': 0.0, #mix up probability 
#         }

# pbt = PB2(time_attr = "training_iteration", perturbation_interval=5, log_config = True,
#         hyperparam_bounds={ 'lr0': [1e-5, 1e-1],
#                             'momentum': [0.6, 0.98],
#                             'weight_decay': [ 0.0, 0.001],
#                             'giou': [ 0.02, 0.2],  
#                             'cls': [ 0.2, 4.0],  
#                             'cls_pw':[ 0.5, 2.0],  
#                             'obj': [0.2, 4.0],  
#                             'obj_pw': [0.5, 2.0], 
#                             'anchor_t': [2.0, 8.0], 
#                             'fl_gamma': [0.0, 2.0],  
#                             'hsv_h': [0.0, 0.1],  
#                             'hsv_s': [0.0, 0.9], 
#                             'hsv_v': [0.0, 0.9],  
#                             'degrees': [0.0, 45.0], 
#                             'translate': [0.0, 0.9],  
#                             'scale': [0.0, 0.9],  
#                             'shear': [0.0, 10.0],  
#                             'perspective': [0.0, 0.001],
#                             'flipud': [0.0, 1.0],  
#                             'fliplr': [0.0, 1.0],
#                             'mixup': [0.0, 1.0]          })

# dicts = cache(imroot,lroot,splits['train']+splits['val'],hyp['img_size'])
# #cache(imroot,lroot,splits['train_10'],hyp['img_size'])
# analysis = tune.run(
#         tune.with_parameters(train_hpo, dicts =  dicts, hyp =  hyp, splits = splits),
#         name="pbt_test",
#         scheduler=pbt,
#         metric="AP50",
#         mode="max",
#         verbose=True,
#         stop={"training_iteration": 300},
#         num_samples=4,
#         fail_fast=True,
#         config= config,
#         keep_checkpoints_num=1,
#         checkpoint_score_attr="AP50",
#         resources_per_trial = {"gpu": 1/4},
#         resume=True)

# print("Best hyperparameters found were: ", analysis.best_config)
