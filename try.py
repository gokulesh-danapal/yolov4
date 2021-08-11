#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:00:42 2021

@author: danapalgokulesh
"""

import torch
import numpy as np
from models import Darknet
import glob
import os
from pathlib import Path
import tqdm, random
from yolo_backend import load_mosaic, random_perspective, torch_distributed_zero_first, InfiniteDataLoader, _RepeatSampler
from torch.utils.data import Dataset
from multiprocessing.pool import ThreadPool
from itertools import repeat
import cv2
from tqdm import tqdm
import torchvision as tv; image = tv.transforms.ToPILImage()

class options:
    def __init__(self):
        self.weights = ''#'/home/danapalgokulesh/dataset/nuscenes/yolo_fusion2.pt'
        self.cam_weight = '/home/danapalgokulesh/dataset/nuscenes/cam.pt'
        self.bev_weight = '/home/danapalgokulesh/dataset/nuscenes/bev.pt'
        self.splits =  '/home/danapalgokulesh/dataset/nuscenes/splits.pytorch'
        self.root = '/home/danapalgokulesh/dataset/nuscenes/radar_bev_image'
        self.nc = 10
        self.epochs = 30
        self.batch_size = 16
        self.img_size = 640
        self.device = 'cuda:1'
        self.cache_images = True
        self.workers = 8
        
opt = options()

def load_weights(model, weights):
    state_dict = {k: v for k, v in torch.load(weights).items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(state_dict, strict=False)
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
    return model
    
def create_dataloader(path, imgsz, batch_size,opt, split = 'train',cache=False,rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      opt = opt,
                                      split = split, # rectangular training
                                      cache_images=cache)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=LoadImagesAndLabels.collate_fn)  # torch.utils.data.DataLoader()
    return dataloader, dataset

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None or img.shape[0] != self.img_size:  # not cached
        path = os.path.join(self.root, self.img_files[index])
        img = cv2.imread(path)  # BGR
        img1 = cv2.imread(path.replace('radar_bev_image','radar_cam_image'+os.sep+'radarv'))
        img2 = np.expand_dims(cv2.imread(path.replace('radar_bev_image','radar_cam_image'+os.sep+'radard'),cv2.IMREAD_GRAYSCALE),2)
        img = np.concatenate((img,img1,img2),axis = 2)
        #img = cv2.resize(img, (80, 80))
        return img
    else:
        return img
    
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path ,img_size=640, batch_size=16, opt = None, split = 'train', cache_images=False):
        self.splits = torch.load(opt.splits)[split]
        self.root = path
        self.img_size = img_size
        n = len(self.splits)
        self.imgs = [None] * n
        self.img_files = os.listdir(path)
        #f = []; p = Path(path);
        #f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        #self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['png']])
        if cache_images:
            gb = 0  # Gigabytes of cached images
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)
                
    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        img = load_image(self, index)
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.copy()
        return torch.from_numpy(img), 1

    @staticmethod
    def collate_fn(batch):
        img,_ = zip(*batch)  # transposed
        return torch.stack(img, 0)

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class CBM(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size, stride, bias =True):
        super(CBM,self).__init__()                               
        self.conv = torch.nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=bias)   
        self.batchnorm = torch.nn.BatchNorm2d(num_features=out_filters,momentum=0.03, eps=1E-4)
        self.act = Mish()
    def forward(self,x):
        return self.act(self.batchnorm(self.conv(x)))
    
class TransCBM(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size, stride, bias =True):
        super(TransCBM,self).__init__()                              
        self.conv = torch.nn.ConvTranspose2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,output_padding = 1,bias=bias)   
        self.batchnorm = torch.nn.BatchNorm2d(num_features=out_filters,momentum=0.03, eps=1E-4)
        self.act = Mish()
    def forward(self,x):
        return self.act(self.batchnorm(self.conv(x)))
    
class CSPOSA(torch.nn.Module):
    def __init__(self, filters):
        super(CSPOSA,self).__init__()
        self.conv1 = CBM(in_filters=filters,out_filters = filters,kernel_size=3,stride=1,bias =False)
        self.conv2 = CBM(in_filters=filters,out_filters = filters//2,kernel_size=3,stride=1,bias =False)
        self.conv3 = CBM(in_filters=filters//2,out_filters = filters//2,kernel_size=3,stride=1,bias =False)
        self.conv4 = CBM(in_filters=filters,out_filters = filters,kernel_size=1,stride=1,bias =False)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x3,x2),dim = 1))
        return torch.cat((x4,x1),dim = 1)   
    
class tinybone(torch.nn.Module):
    def __init__(self, in_filters):
        super(tinybone,self).__init__()
        self.main1r = torch.nn.Sequential(CBM(in_filters=in_filters,out_filters=32,kernel_size=3,stride=1,bias = False),
                                CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2,bias = False))
        self.main2r = torch.nn.Sequential(CSPOSA(64),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main3r = torch.nn.Sequential(CSPOSA(128),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
    def forward(self,rad):
        x = self.main2r(self.main1r(rad))
        x3 = self.main3r(x)
        return x3

class bev2cam(torch.nn.Module):
    def __init__(self,filters):
        super(bev2cam,self).__init__()
        self.down = torch.nn.Sequential(CBM(filters,filters,5,2),CBM(filters,filters,5,2),CBM(filters,filters,5,2),CBM(filters,filters,5,2),
                                      torch.nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=5,stride=1,padding = 0,bias=True))
        self.bottle = torch.nn.Sequential(torch.nn.Linear(filters,filters),
                                              torch.nn.Linear(filters,filters))
        self.up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels = filters, out_channels = filters, kernel_size = 5, stride = 1, padding = 0, bias=True),
                                        TransCBM(filters, filters, 5, 2),TransCBM(filters, filters, 5, 2),TransCBM(filters, filters, 5, 2),TransCBM(filters, filters, 5, 2))
    
    def forward(self,bev):
        bev = self.down(bev)
        #print(bev.shape)
        bev = self.bottle(torch.squeeze(bev))
        #print(bev.shape)
        bev = torch.unsqueeze(torch.unsqueeze(bev,-1),-1)
        if len(bev.shape) < 4:
           bev = torch.unsqueeze(bev,0)
        cam = self.up(bev)
        #print(cam.shape)
        return cam

class bev2cam1(torch.nn.Module):
    def __init__(self,filters):
        super(bev2cam1,self).__init__()
        self.down = CBM(filters,filters,5,2)
        #self.bottle = torch.nn.Linear(1600,1600)
        self.bottle = torch.nn.Sequential(torch.nn.Linear(1600,800),torch.nn.ReLU(),torch.nn.Linear(800,1600))
        self.up = TransCBM(filters, filters, 5, 2)
    
    def forward(self,bev):
        bev = self.down(bev)
        shapes = bev.shape
        #print(bev.shape)
        ligs = torch.tensor_split(bev, shapes[1],dim=1)
        #bev = bev.view(shapes[0],shapes[1],-1)
        logs = [None]*len(ligs)
        for i, lig in enumerate(ligs):
            logs[i] = self.bottle(lig.view(shapes[0],-1))
        #print(bev.shape)
        bev = torch.cat(logs,1)
        bev = bev.view(shapes)
        cam = self.up(bev)
        #print(cam.shape)
        return cam
    
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=4,stride=2,padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1),
            torch.nn.Conv2d(in_channels=512,out_channels=1,kernel_size=3,stride=1,padding=0),
            torch.nn.Sigmoid())
    def forward(self,inputs):
        return self.main(inputs)

netG = bev2cam1(256).to(opt.device)
#model.load_state_dict(torch.load(opt.weights)['model'])
netD = Discriminator().to(opt.device)
bev = tinybone(3).to(opt.device)
bev = load_weights(bev, opt.bev_weight)
cam = tinybone(4).to(opt.device)
cam = load_weights(cam, opt.cam_weight)
optimizerG = torch.optim.Adam(netG.parameters(),lr = 0.001,betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(),lr=0.001,betas=(0.5, 0.9))

Lambda = 100  # according to paper authors
bce_loss = torch.nn.BCELoss()
real_labels = (torch.ones(opt.batch_size, 1, 18, 18)*0.9).to(opt.device)
fake_labels = torch.zeros(opt.batch_size, 1, 18, 18).to(opt.device)   

def trainD(src_images, tgt_images):
    
    fake_images = netG(src_images)
    
    netD.zero_grad()
    optimizerD.zero_grad()

    real_outputs = netD(torch.cat((src_images, tgt_images), 1))
    fake_outputs = netD(torch.cat((src_images, fake_images), 1))
    d_x = bce_loss(real_outputs, real_labels)
    d_g_z = bce_loss(fake_outputs, fake_labels)
    
    loss_d = d_x + d_g_z
    loss_d.backward()
    optimizerD.step()

    return loss_d       

# Train Generator

def trainG(src_images ,tgt_images):
  
    netG.zero_grad()
    optimizerG.zero_grad()
    loss_g = 0
    
    fake_images = netG(src_images)

    outputs = netD(torch.cat((src_images, fake_images), 1))
    
    loss_g = bce_loss(outputs, real_labels)

    loss_g.backward()
    optimizerG.step()

    return loss_g, fake_images



print('training for',opt.epochs,'epochs')
#print('training for',opt.epochs,'epochs')
ckpt = {}
dataloader, _ = create_dataloader(opt.root, opt.img_size, opt.batch_size, opt, split = 'train',cache =False)
K =1

for epoch in range(opt.epochs):
    d_loss = 0; g_loss = 0;
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),position=0, leave=True)  # progress bar
    for i, imgs in pbar:
        imgs = imgs/255.0
        ins = imgs[:,:3,:,:].to(opt.device).float()
        outs = imgs[:,3:,:,:].to(opt.device).float()
        ins = bev(ins); outs = cam(outs);
        for k in range(K):
            d_loss += trainD(ins,outs)
        d_loss /= K
        loss, x_hat = trainG(ins,outs)
        g_loss += loss
    print(loss)
    ckpt['modelG'] = netG.state_dict()
    ckpt['modelD'] = netD.state_dict()
    ckpt['optimG'] = optimizerG.state_dict()
    ckpt['optimD'] = optimizerD.state_dict()
    torch.save(ckpt,'/content/gdrive/My Drive/nuscenes/runs/weights.pt')
        
    



