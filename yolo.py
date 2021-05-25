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
        #'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 4, #Number of classes
        'names' : ['person', 'RidableVehicle', 'car', 'LargeVehicle'],
        'gs': 32, #Image size multiples
        'img_size': 320, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 40, #number of epochs
        'batch_size': 2, #train batch size
        'test_size': 2, #test batch size
        'use_adam': False, #Bool to use Adam optimiser
        'use_ema': True, #Exponential moving average control
        'multi_scale': False, #Bool to do multi-scale training
        'test_all': True, #Run test after end of each epoch
        'save_all': True, #Save checkpoints after every epoch
        'auto_anchor': True, #create new anchors by Kmeans clustering
        
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
        
        'anchors_g': [[9.72318281, 89.90291182],
                    [27.38753188, 76.11763602],
                    [ 17.16083467, 163.93638637],
                    [ 48.64984549, 127.39453151],
                    [ 32.2540104 , 286.52858829],
                    [ 86.4122153 , 222.36901603],
                    [100.69202021, 821.01962072],
                    [155.38462465, 351.55357456],
                    [253.37522873, 768.02296055]]
     }


#weight_path = '/home/danapalgokulesh/dataset/dense/hpo/'
weight_path = '/home/danapalgokulesh/dataset/dense/yolo_pre_4c.pt'
imroot = '/home/danapalgokulesh/dataset/dense/images'
lroot = '/home/danapalgokulesh/dataset/dense/labels'
logdir = '/home/danapalgokulesh/dataset/dense/runs_hpo'
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

dicts = cache(imroot,lroot,splits['train']+splits['val'],hyp['img_size'])
train_set = Dataset(hyp,dicts, splits['train'],augment=True)#, splits =  splits['image_weights'])
val_set = Dataset(hyp,dicts, splits['val'], augment= False)
tb_writer = SummaryWriter(log_dir = logdir)
results = train(hyp,tb_writer, train_set, weight_path, val_set)

# cache(imroot,lroot,splits['val'],hyp['img_size'])
# test_set = Dataset(hyp, splits['val'], augment= False)
# results = test(test_set,hyp,weight_path,plot_all = False)

# #%%   
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

# cache(imroot,lroot,splits['train']+splits['val'],hyp['img_size'])
# #cache(imroot,lroot,splits['train_10'],hyp['img_size'])
# analysis = tune.run(
#         tune.with_parameters(train_hpo, img_dict =  img_dict, hyp =  hyp, splits = splits),
#         name="pbt_test",
#         scheduler=pbt,
#         metric="AP50",
#         mode="max",
#         verbose=True,
#         stop={"training_iteration": 100},
#         num_samples=4,
#         fail_fast=True,
#         config= config,
#         keep_checkpoints_num=1,
#         checkpoint_score_attr="AP50",
#         resources_per_trial = {"gpu": 1/4},
#         resume=False)

# print("Best hyperparameters found were: ", analysis.best_config)
