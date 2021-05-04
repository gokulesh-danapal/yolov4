# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:12 2021

@author: Gokulesh Danapal (GX6)

Confidentiality: Internal
"""

from yolo_backend import Dataset, Darknet, train, test, last_layer_train
from torch.utils.tensorboard import SummaryWriter

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

hyp = { 'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda    
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 80, #Number of classes
        'gs': 32, #Image size multiples
        'img_size': 608, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 10, #number of epochs
        'batch_size': 4, #train batch size
        'test_size': 1, #test batch size
        'use_adam': False, #Bool to use Adam optimiser
        'use_ema': True, #Exponential moving average control
        'multi_scale': False, #Bool to do multi-scale training
        'test_all': False, #Run test after end of each epoch
        'save_all': True, #Save checkpoints after every epoch
        
        'giou': 0.05,  # GIoU loss gain
        'cls': 0.5,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 1.0,  # obj loss gain (scale with pixels)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
        'iou_t': 0.6,  # IoU training threshold
        'conf_t':0.2, # Confidence training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.0,  # image flip left-right (probability)
        'mixup': 0.0 #mix up probability
     }


weight_path = '/home/danapalgokulesh/code/yolo/yolo_pre.pt'
imroot = '/home/danapalgokulesh/dataset/dense/images/train'
lroot = '/home/danapalgokulesh/dataset/dense/labels'
logdir = '/home/danapalgokulesh/dataset/dense/runs'
test_root = '/home/danapalgokulesh/dataset/dense/images/test_clear_day'

#weight_path = r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\yolo_pre.pt"
#imroot = r'E:\Datasets\Dense\split_cam\test_clear_night'
#logdir = r'E:\Datasets\Dense\runs'
#lroot = r'E:\Datasets\Dense\labels_4'

#test_root = r'E:\Datasets\Dense\split_cam\test_clear_day'

train_set = Dataset(hyp,imroot,lroot,augment=True)
test_set = Dataset(hyp,test_root, lroot, augment= False)
tb_writer = SummaryWriter(log_dir = logdir)


results = train(hyp,tb_writer, train_set, weight_path, test_set)

#results = test(test_set,names,hyp,weight_path,plot_all = True)

#results = last_layer_train(hyp,tb_writer, train_set, 4, weight_path, test_set)
