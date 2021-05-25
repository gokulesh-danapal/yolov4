# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:12 2021
@author: Gokulesh Danapal (GX6)
Confidentiality: Internal
"""
import torch
import os
from yolo_backend import Dataset, Darknet, train, test
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import tune



hyp = { 'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda    
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        #'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 4, #Number of classes
        'names' : ['person', 'RidableVehicle', 'car', 'LargeVehicle'],
        'gs': 32, #Image size multiples
        'img_size': 640, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 40, #number of epochs
        'batch_size': 16, #train batch size
        'test_size': 16, #test batch size
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


#weight_path = '/home/danapalgokulesh/dataset/dense/yolo_pre_4c.pt'
weight_path = '/home/danapalgokulesh/dataset/dense/runs/weights/last.pt'
imroot = '/home/danapalgokulesh/dataset/dense/images'
lroot = '/home/danapalgokulesh/dataset/dense/labels'
logdir = '/home/danapalgokulesh/dataset/dense/runs'
splits = torch.load('/home/danapalgokulesh/dataset/dense/splits.pytorch')

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

train_set = Dataset(hyp,imroot,lroot,splits['train'], augment=True)#, image_weights= splits['image_weights'])
val_set = Dataset(hyp,imroot, lroot,splits['val'], augment= False)
tb_writer = SummaryWriter(log_dir = logdir)
results = train(hyp,tb_writer, train_set, weight_path, val_set, splits)

# test_set = Dataset(hyp,imroot, lroot,splits['test_dense_fog'], augment= False)
# results = test(test_set,hyp,weight_path,plot_all = False)
class train_yolo(tune.Trainable):
    def setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = ConvNet().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9))
    
    def step(self):
        train(
            self.model, self.optimizer, self.train_loader, device=self.device)
        acc = test(self.model, self.test_loader, self.device)
        return {"mean_accuracy": acc}
    
    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
    

