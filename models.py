#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:12:46 2021

@author: danapalgokulesh
"""
import torch
import numpy as np

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x
    
class CBM(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size, stride):
        super(CBM,self).__init__()                               
        self.conv = torch.nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=False)   
        self.batchnorm = torch.nn.BatchNorm2d(num_features=out_filters,momentum=0.03, eps=1E-4)
        self.act = Mish()
    def forward(self,x):
        return self.act(self.batchnorm(self.conv(x)))
        
class ResUnit(torch.nn.Module):
    def __init__(self, filters, first = False):
        super(ResUnit, self).__init__()
        if first:
            self.out_filters = filters//2
        else:
            self.out_filters = filters         
        self.resroute= torch.nn.Sequential(CBM(filters, self.out_filters, kernel_size=1, stride=1),
                                                    CBM(self.out_filters, filters, kernel_size=3, stride=1))       
    def forward(self, x):
        shortcut = x
        x = self.resroute(x)
        return x+shortcut

class CSP(torch.nn.Module):
    def __init__(self, filters, nblocks):
        super(CSP,self).__init__()
        self.skip = CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1)
        self.route_list = torch.nn.ModuleList()
        self.route_list.append(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1))
        for block in range(nblocks):
            self.route_list.append(ResUnit(filters=filters//2))
        self.route_list.append(CBM(in_filters=filters//2,out_filters=filters//2,kernel_size=1,stride=1))                                         
        self.last = CBM(in_filters=filters,out_filters=filters,kernel_size=1,stride=1)
        
    def forward(self,x):
        shortcut = self.skip(x)
        for block in self.route_list:
            x = block(x)
        x = torch.cat((x,shortcut),dim = 1)
        return self.last(x)

class CSPOSA(torch.nn.Module):
    def __init__(self, filters):
        super(CSPOSA,self).__init__()
        self.conv1 = CBM(in_filters=filters,out_filters = filters,kernel_size=3,stride=1)
        self.conv2 = CBM(in_filters=filters,out_filters = filters//2,kernel_size=3,stride=1)
        self.conv3 = CBM(in_filters=filters//2,out_filters = filters//2,kernel_size=3,stride=1)
        self.conv4 = CBM(in_filters=filters,out_filters = filters,kernel_size=1,stride=1)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x3,x2),dim = 1))
        return torch.cat((x4,x1),dim = 1)
class bev2cam(torch.nn.Module):
    def __init__(self,filters):
        super(bev2cam,self).__init__()
        self.down = torch.nn.Sequential(torch.nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=5,stride=2,padding=5//2,bias=True),
                                      torch.nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=5,stride=2,padding=5//2,bias=True),
                                      torch.nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=5,stride=2,padding = 5//2,bias=True),
                                      #torch.nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=5,stride=2,padding=5//2,bias=True),
                                      torch.nn.Conv2d(in_channels=filters,out_channels=filters,kernel_size=5,stride=1,padding = 0,bias=True))
        self.bottle = torch.nn.Sequential(torch.nn.Linear(filters,filters),
                                              torch.nn.Linear(filters,filters))
        self.up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels = filters, out_channels = filters, kernel_size = 5, stride = 1, padding = 0, bias=True),
                                      #torch.nn.ConvTranspose2d(in_channels = filters, out_channels = filters, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 1, bias=True),
                                      torch.nn.ConvTranspose2d(in_channels = filters, out_channels = filters, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 1, bias=True),
                                      torch.nn.ConvTranspose2d(in_channels = filters, out_channels = filters, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 1, bias=True),
                                      torch.nn.ConvTranspose2d(in_channels = filters, out_channels = filters, kernel_size = 5, stride = 2, padding = 5//2, output_padding = 1, bias=True))
    def forward(self,bev):
        bev = self.down(bev)
        #print(bev.shape)
        bev = self.bottle(torch.squeeze(bev))
        #print(bev.shape)
        bev = torch.unsqueeze(torch.unsqueeze(bev,-1),-1)
        if len(bev.shape) < 4:
           bev = torch.unsqueeze(bev,0)
        cam = self.up(bev)
        return cam
 
class tinybone(torch.nn.Module):
    def __init__(self, channels):
        super(tinybone,self).__init__()
        if channels ==1 : 
            in_filters = 4
        else:
            in_filters = 3
        self.main1r = torch.nn.Sequential(CBM(in_filters=in_filters,out_filters=32,kernel_size=3,stride=1),
                                CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2))
        self.main2r = torch.nn.Sequential(CSPOSA(64),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main3r = torch.nn.Sequential(CSPOSA(128),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main4r = torch.nn.Sequential(CSPOSA(256),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main5r = torch.nn.Sequential(CSPOSA(512),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.transfer = bev2cam(256) 
    def forward(self,rad):
        x = self.main2r(self.main1r(rad))
        x3 = self.main3r(x)
        x3 = self.transfer(x3)
        x4 = self.main4r(x3)
        x5 = self.main5r(x4)
        return (x3,x4,x5)
      
class SPP(torch.nn.Module):
    def __init__(self,filters):
        super(SPP,self).__init__()
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=5,stride=1,padding = 5//2)
        self.maxpool9 = torch.nn.MaxPool2d(kernel_size=9,stride=1,padding = 9//2)
        self.maxpool13 = torch.nn.MaxPool2d(kernel_size=13,stride=1,padding = 13//2)
    def forward(self,x):
        x5 = self.maxpool5(x)
        x9 = self.maxpool9(x)
        x13 = self.maxpool13(x)
        return torch.cat((x13,x9,x5,x),dim=1)
            
class rCSP(torch.nn.Module):
    def __init__(self,filters,spp_block = False):
        super(rCSP,self).__init__()
        self.include_spp = spp_block
        if self.include_spp:
            self.in_filters = filters*2
        else:
            self.in_filters = filters
        self.skip = CBM(in_filters=self.in_filters,out_filters=filters,kernel_size=1,stride=1)
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Sequential(CBM(in_filters=self.in_filters,out_filters=filters,kernel_size=1,stride=1),
                                CBM(in_filters=filters,out_filters=filters,kernel_size=3,stride=1),
                                CBM(in_filters=filters,out_filters=filters,kernel_size=1,stride=1)))
        if self.include_spp:
            self.module_list.append(torch.nn.Sequential(SPP(filters=filters),
                                    CBM(in_filters=filters*4,out_filters=filters,kernel_size=1,stride=1)))
        self.module_list.append(CBM(in_filters=filters,out_filters=filters,kernel_size=3,stride=1))
        self.last = CBM(in_filters=filters*2,out_filters=filters,kernel_size=1,stride=1)
    def forward(self,x):
        shortcut = self.skip(x)
        for block in self.module_list:
            x = block(x)
        x = torch.cat((x,shortcut),dim=1)
        x = self.last(x)
        return x 
    
def up(filters):
        return torch.nn.Sequential(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1),
                                        torch.nn.Upsample(scale_factor=2))

class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors, nc, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        #self.index = yolo_index  # index of this layer in layers
        #self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        #self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        #if (self.nx, self.ny) != (nx, ny):
        self.create_grids((nx, ny), p.device)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

class ChannelPool(torch.nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class SAM2(torch.nn.Module):
    def __init__(self,size):
        super(SAM2,self).__init__()
        self.compress = ChannelPool()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(in_channels=2,out_channels=1,kernel_size=size, stride=1, padding= size//2),
                                  torch.nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True))
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x):
        x_compress = self.conv(self.compress(x))
        weight = self.sigmoid(x_compress)
        return weight

class CAM(torch.nn.Module):
    def __init__(self,channels,reduction_ratio):
        super(CAM,self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.mlp = torch.nn.Sequential(Flatten(),torch.nn.Linear(channels, channels//reduction_ratio),torch.nn.ReLU(),torch.nn.Linear(channels//reduction_ratio, channels))
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        avg_pool = torch.nn.functional.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_avg = self.mlp(avg_pool)
        
        max_pool = torch.nn.functional.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_max = self.mlp( max_pool )
        
        channel_att_sum = channel_att_avg + channel_att_max
        weight = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = x*weight
        return out #x * weight, weight
    
class TAM(torch.nn.Module):
    def __init__(self,in_channels,out_channels,reduction_ratio, s):
        super(TAM,self).__init__()
        if s > 1:
            self.embedv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=s, stride = s)
            self.embedr = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=s, stride = s)
            self.upsamplev = torch.nn.ConvTranspose2d(1, 1, s, stride=s)
            self.upsampler = torch.nn.ConvTranspose2d(1, 1, s, stride=s)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.in_channels, self.in_channels//self.reduction_ratio),torch.nn.ReLU(),torch.nn.Linear(self.in_channels//self.reduction_ratio, self.out_channels))
        self.sigmoid = torch.nn.Sigmoid()
        self.s =  s
    def forward(self, x, rad):
        B,_,s1,s2 =  x.shape
        vmax = torch.max(x,1)[0].unsqueeze(1)
        rmax = torch.max(rad,1)[0].unsqueeze(1)
        if self.s > 1:        
            vmax = self.embedv(vmax)
            rmax = self.embedr(rmax)
        vmax = vmax.view(B,-1)
        rmax = rmax.view(B,-1)
        max_att = self.mlp(torch.cat((vmax,rmax),1))
        vavg = torch.mean(x,1).unsqueeze(1)
        ravg =  torch.mean(rad,1).unsqueeze(1)
        if self.s > 1:
           vavg = self.embedv(vavg)
           ravg = self.embedr(ravg)
        vavg =  vavg.view(B,-1)
        ravg =  ravg.view(B,-1)
        avg_att = self.mlp(torch.cat((vavg,ravg),1))
        
        att_sum = max_att+avg_att
        att_v = att_sum[:,:att_sum.shape[1]//2]; att_r = att_sum[:,att_sum.shape[1]//2:];
        att_v= att_v.view(B,1,s1//self.s,s2//self.s)
        att_r= att_r.view(B,1,s1//self.s,s2//self.s)
        if self.s > 1 :
            att_v = self.upsamplev(att_v)
            att_r = self.upsampler(att_r)
        weight_v = self.sigmoid(att_v)
        weight_r = self.sigmoid(att_r)
        return x*weight_v, rad*weight_r
    
class CAM_rad(torch.nn.Module):
    def __init__(self,channels,reduction_ratio):
        super(CAM_rad,self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.mlp = torch.nn.Sequential(Flatten(),torch.nn.Linear(channels, channels//reduction_ratio),torch.nn.ReLU(),torch.nn.Linear(channels//reduction_ratio, channels//2))
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x, radm, rada):
        avg_pool = torch.nn.functional.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool = avg_pool.view(-1,1024)
        channel_att_avg = self.mlp(torch.cat((avg_pool,rada),1))
        
        max_pool = torch.nn.functional.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = max_pool.view(-1,1024)
        channel_att_max = self.mlp(torch.cat((max_pool,radm),1))
        
        channel_att_sum = channel_att_avg + channel_att_max
        weight = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = x*weight
        return out #x * weight, weight
    
class SAM1(torch.nn.Module):
    def __init__(self,filters):
        super(SAM1,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=filters,out_channels=1,kernel_size=1,stride=1,padding= 0) 
        self.conv2 = torch.nn.Conv2d(in_channels=filters,out_channels=1,kernel_size=3,stride=1,padding= 1)
        self.conv3 = torch.nn.Conv2d(in_channels=filters,out_channels=1,kernel_size=5,stride=1,padding= 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        #att_map = self.sigmoid(x1+x2+x3)
        att_map = x1 + x2 + x3
        return att_map
        
class fusebone(torch.nn.Module):
    def __init__(self,return_att):
        super(fusebone,self).__init__()
        self.return_att = return_att
        #vision branch----------------------------------------------------------
        self.main1v = torch.nn.Sequential(CBM(in_filters=3,out_filters=32,kernel_size=3,stride=1),
                                        CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                        ResUnit(filters = 64, first= True))
        self.main2v = torch.nn.Sequential(CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                        CSP(filters=128,nblocks = 2))
        self.main3v = torch.nn.Sequential(CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                        CSP(filters=256,nblocks = 8))
        self.main4v = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2),
                                        CSP(filters=512,nblocks = 8))
        self.main5v = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2),
                                        CSP(filters=1024,nblocks = 4))
        # #radar branch----------------------------------------------------------
        self.main1r = torch.nn.Sequential(CBM(in_filters=4,out_filters=32,kernel_size=3,stride=1),
                                CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2))
        self.main2r = torch.nn.Sequential(CSPOSA(64),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main3r = torch.nn.Sequential(CSPOSA(128),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main4r = torch.nn.Sequential(CSPOSA(256),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        self.main5r = torch.nn.Sequential(CSPOSA(512),
                                torch.nn.MaxPool2d(kernel_size=2,stride = 2))
        #self.transfer = bev2cam(256) 
        # #radar branch----------------------------------------------------------
        # self.main1r = torch.nn.Sequential(CBM(in_filters=3,out_filters=32,kernel_size=3,stride=1),
        #                         CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2))
        # self.main2r = CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2)
        # self.main3r = CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2)
        # self.main4r = CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2)
        # self.main5r = CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2)
        #Attentive Spatial fusion----------------------------------------------
        # self.sam1 = SAM1(64)
        #self.sam2 = SAM1(128)
        # self.sam3 = SAM1(256)
        # self.sam4 = SAM1(512)
        # self.sam5 = SAM1(1024) 
        
        # self.sam1 = SAM2(3)
        # self.sam2 = SAM2(3)
        # self.sam3 = SAM2(3)
        # self.sam4 = SAM2(3)
        # self.sam5 = SAM2(3) 
        
        # self.sam1v = SAM2(7); self.sam1r = SAM2(7); self.sam1f = SAM2(7);
        # self.sam2v = SAM2(7); self.sam2r = SAM2(7); self.sam2f = SAM2(7);
        # self.sam3v = SAM2(7); self.sam3r = SAM2(7); self.sam3f = SAM2(7);
        # self.sam4v = SAM2(7); self.sam4r = SAM2(7); self.sam4f = SAM2(7);
        # self.sam5v = SAM2(7); self.sam5r = SAM2(7); self.sam5f = SAM2(7); 
        
        #Channel fusion module---------------------------------------------------
        # self.cam3 = CAM(512,16)
        # self.cam4 = CAM(1024,16)
        # self.cam5 = CAM(2048,16)
        # self.tam3 = TAM(480,480,16,2)
        # self.tam4 = TAM(480,480,16,1)
        # self.tam5 = TAM(120,120,16,1)
        self.half3 = torch.nn.Sequential(CBM(in_filters=512,out_filters=512,kernel_size=3,stride=1),
                                          CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1))
        self.half4 = torch.nn.Sequential(CBM(in_filters=1024,out_filters=1024,kernel_size=3,stride=1),
                                          CBM(in_filters=1024,out_filters=512,kernel_size=1,stride=1))
        self.half5 = torch.nn.Sequential(CBM(in_filters=2048,out_filters=2048,kernel_size=3,stride=1),
                                          CBM(in_filters=2048,out_filters=1024,kernel_size=1,stride=1))     
    def forward(self,x):
        v = x[:,:3,:,:]; r = x[:,3:,:,:]
        #downsample1
        v1 = self.main1v(v)
        r1 = self.main1r(r)
        #v1 = v1 * self.sam1(r1)
        # va1 = self.sam1v(v1);ra1 = self.sam1r(r1); fa1 = self.sam1f(torch.cat((v1,r1),1));
        # v1 = v1 * (va1+fa1); r1 = r1 * (ra1+fa1)
        #downsample2
        v2 = self.main2v(v1)
        r2 = self.main2r(r1)
        #v2 = v2 * self.sam2(r2) 
        # va2 = self.sam2v(v2); ra2 = self.sam2r(r2); fa2 = self.sam2f(torch.cat((v2,r2),1));
        # v2 = v2 * (va2+fa2); r2 = r2 * (ra2+fa2)
        #downsample3
        v3 = self.main3v(v2)
        r3 = self.main3r(r2)
        #r3 = self.transfer(r3)
        #v3 = v3 * self.sam3(r3)
        #v3 = v3 * self.sam3(r3)
        # va3 = self.sam3v(v3); ra3 = self.sam3r(r3); fa3 = self.sam3f(torch.cat((v3,r3),1));
        # v3 = v3 * (va3+fa3); r3 = r3 * (ra3+fa3)
        #v3,r3 = self.tam3(v3,r3)
        #downsample4
        v4 = self.main4v(v3)
        r4 = self.main4r(r3)
        #v4 = v4 * self.sam4(r4)
        # va4 = self.sam4v(v4); ra4 = self.sam4r(r4); fa4 = self.sam4f(torch.cat((v4,r4),1));
        # v4 = v4 * (va4+fa4); r4 = r4 * (ra4+fa4)     
        #v4,r4 = self.tam4(v4,r4)
        #downsample5
        v5 = self.main5v(v4)
        r5 = self.main5r(r4)
        #v5 = v5 * self.sam5(r5)
        # va5 = self.sam5v(v5); ra5 = self.sam5r(r5); fa5 = self.sam5f(torch.cat((v5,r5),1));
        # v5 = v5 * (va5+fa5); r5 = r5 * (ra5+fa5) 
        #v5,_ = self.tam5(v5,r5)
        #channel attention
        # x3 =  self.half3(self.cam3(torch.cat((v3,r3),1)))
        # x4 =  self.half4(self.cam4(torch.cat((v4,r4),1)))
        # x5 =  self.half5(self.cam5(torch.cat((v5,r5),1)))
        x3 =  self.half3(torch.cat((v3,r3),1));  #x5=  v5; x4 =  v4;
        x4 =  self.half4(torch.cat((v4,r4),1)) 
        x5 =  self.half5(torch.cat((v5,r5),1))
        if self.return_att:
            return (x3,x4,x5)#,(va1,ra1,fa1,va2,ra2,fa2,va3,ra3,fa3,va4,ra4,fa4,va5,ra5,fa5)
        else:
            return (x3,x4,x5)

class Backbone(torch.nn.Module):
    def __init__(self,channels = 3):
        super(Backbone,self).__init__()
        self.main3 = torch.nn.Sequential(CBM(in_filters=channels,out_filters=32,kernel_size=3,stride=1),
                                        CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                        ResUnit(filters = 64, first= True),
                                        CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                        CSP(filters=128,nblocks = 2), 
                                        CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                        CSP(filters=256,nblocks = 8))
        self.main4 = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2),
                                        CSP(filters=512,nblocks = 8))
        self.main5 = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2),
                                        CSP(filters=1024,nblocks = 4))
    def forward(self,x):
        x3 = self.main3(x)
        x4 = self.main4(x3)
        x5 = self.main5(x4)
        return (x3,x4,x5)
    
class Neck(torch.nn.Module):
    def __init__(self, f = 256):
        super(Neck,self).__init__()
        self.main5 = rCSP(f*2,spp_block=True)
        self.up5 = up(f*2)
        self.conv1 = CBM(in_filters=f*2,out_filters=f,kernel_size=1,stride=1)
        self.conv2 = CBM(in_filters=f*2,out_filters=f,kernel_size=1,stride=1)
        self.main4 = rCSP(f)
        self.up4 = up(f)
        self.conv3 = CBM(in_filters=f,out_filters=f//2,kernel_size=1,stride=1)
        self.conv4 = CBM(in_filters=f,out_filters=f//2,kernel_size=1,stride=1)
        self.main3 = rCSP(f//2)
    def forward(self,x):
        x3 = x[0]; x4 = x[1]; x5= x[2];
        x5 = self.main5(x5)
        x4 = self.main4(self.conv2(torch.cat((self.conv1(x4),self.up5(x5)),dim=1)))
        x3 = self.main3(self.conv4(torch.cat((self.conv3(x3),self.up4(x4)),dim=1)))
        return (x3,x4,x5)
    
class Head(torch.nn.Module):
    def __init__(self,nclasses,f=256):
        super(Head,self).__init__()
        self.last_layers = 3*(4+1+nclasses)
        self.last3 = CBM(in_filters=f//2,out_filters=f,kernel_size=3,stride=1)
        self.final3 = torch.nn.Conv2d(in_channels=f,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True) 
        
        self.conv1 = CBM(in_filters=f//2,out_filters=f,kernel_size=3,stride=2)
        self.conv2 = CBM(in_filters=f*2,out_filters=f,kernel_size=1,stride=1)
        self.main4 = rCSP(f)
        self.last4 = CBM(in_filters=f,out_filters=f*2,kernel_size=3,stride=1)
        self.final4 = torch.nn.Conv2d(in_channels=f*2,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True)
        
        self.conv3 = CBM(in_filters=f,out_filters=f*2,kernel_size=3,stride=2)
        self.conv4 = CBM(in_filters=f*4,out_filters=f*2,kernel_size=1,stride=1)
        self.main5 = rCSP(f*2)
        self.last5 = CBM(in_filters=f*2,out_filters=f*4,kernel_size=3,stride=1)
        self.final5 = torch.nn.Conv2d(in_channels=f*4,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True)
    def forward(self,x):
        x3 = x[0]; x4 = x[1]; x5= x[2];
        y3 = self.final3(self.last3(x3))
        x4 = self.main4(self.conv2(torch.cat((self.conv1(x3),x4),dim=1)))
        y4 = self.final4(self.last4(x4))
        x5 = self.main5(self.conv4(torch.cat((self.conv3(x4),x5),dim=1)))
        y5 = self.final5(self.last5(x5))
        return y3,y4,y5

class Darknet(torch.nn.Module):
    def __init__(self,nclasses,anchors,channels = 3,return_att = False):
        super(Darknet,self).__init__()
        self.nclasses = nclasses
        self.anchors = np.array(anchors)
        self.return_att = return_att
        self.filters = 256
        if channels == 3:
            self.backbone = Backbone()
        elif channels in [1,2]:
            self.backbone = tinybone(channels)
        elif channels in [5,6]:
            self.backbone = fusebone(self.return_att)
            #self.filters = 512
        # elif channels == 7:
        #     self.backbone = pointnet()
        self.neck = Neck(self.filters)
        self.head = Head(self.nclasses,self.filters)
        self.yolo3 = YOLOLayer(self.anchors[0:3], self.nclasses, stride = 8)
        self.yolo4 = YOLOLayer(self.anchors[3:6], self.nclasses, stride = 16)
        self.yolo5 = YOLOLayer(self.anchors[6:9], self.nclasses, stride = 32)

    def forward(self,x):
        if self.return_att:
            if type(x) is tuple:
                y,attention = self.backbone(x[0],x[1])
                y3,y4,y5 = self.head(self.neck(y))
            else:
                y,attention = self.backbone(x)
                y3,y4,y5 = self.head(self.neck(y))
        else:
            if type(x) is tuple:
               y3,y4,y5 = self.head(self.neck(self.backbone(x[0],x[1])))
            else:
                y3,y4,y5 = self.head(self.neck(self.backbone(x)))
        y3 = self.yolo3(y3)
        y4 = self.yolo4(y4)
        y5 = self.yolo5(y5)
        yolo_out = [y3,y4,y5]
        if self.training:
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if self.return_att:
                return x, p, attention
            else:
                return x,p
