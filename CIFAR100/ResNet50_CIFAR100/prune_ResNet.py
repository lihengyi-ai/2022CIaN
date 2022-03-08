
'''
Author: IHPC ELla
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import time
import argparse
import sys
import os
import shutil
import cv2
import torch.nn as nn
from common import *
import pandas as pd
from resnet_cifar import resnet50

cfg: List[Union[int]] = [i for i in range(100)]

DIVIDER = '-----------------------------------------'

sparsity = []
for layer_in in range(0,33):
      #print(layer_in)
      layers_data = np.load("./input/"+str(layer_in)+'.npy')
      element = len(layers_data[0])*len(layers_data[0][0])*len(layers_data[0][0][0]) 
      NZero_features = 0

      for i in range(len(layers_data[0])):
          for j in range(len(layers_data[0][i])):
              for k in range(len(layers_data[0][i][j])): 
                  if(layers_data[0][i][j][k] != 0):
                      NZero_features += 1

      percentage = 1-(NZero_features/element)
      sparsity.append(percentage)

print(sparsity)

def default_loader(path):
    return Image.open(path).convert("RGB")

class GetLoader(torch.utils.data.Dataset):
    #def __init__(self,data_root,data_label):
    def __init__(self,file,loader = default_loader):
        
        imgs = []

        for i in range(len(file)):
            imgs.append((IMAGE_PATH + file[i][0] + "/" + file[i][1],int(file[i][2])))
        
        self.imags = imgs
        self.loader = loader
        self.transform = transforms.Compose(
            [transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(60),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485,0.456,0.406],
                std = [0.229,0.224,0.225,])
            ]
            
            )
        
    def __getitem__(self,index):
        
        data1,datalabel = self.imags[index]

        data2 = self.loader(data1)
        
        data2 = self.transform(data2)

        labels = datalabel
        
        return data2,labels
    
    def __len__(self):
        return len(self.imags)
# additional subgradient descent on the sparsity-induced penalty term

def train_test(dset_dir, batchsize, learnrate, epochs, float_model):

    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')
    device = torch.device('cpu')
# =============================================================================
# create the model
# =============================================================================

    model = resnet50()
    
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    # model = torch.load(save_path2)
    
    save_path2 = os.path.join('pruned_model', 'pruned_para.pth')
    
    model.to(device)
    
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
# =============================================================================
# load pretrained parameters
# =============================================================================
    
    save_path2 = os.path.join('pruned_model', 'pruned_para.pth')

    #newmodel = torch.load(save_path2)
    
    newmodel = resnet50()
    #print(model)
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    newmodel.to(device)
    newmodel.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))

# =============================================================================
# calculate the treshold according to the prunning ratio
# =============================================================================

    threhold = torch.zeros(100)
    index = 0
    layer_index = 0
    first_identity = 0
    

    block_index = ["conv2_x","conv3_x","conv4_x","conv5_x"]
    conv_flag = "conv2_x"
    conv_index = [1,4]

    for conv_flag in block_index:
        
        if conv_flag == "conv2_x":
            for i in range(3):
                for j in conv_index:
                    
                    size = model.conv2_x[i].residual_function[j].weight.data.shape[0]

                    bn = torch.zeros(size)
                    bn = model.conv2_x[i].residual_function[j].weight.data.abs().clone()
                    m, n = torch.sort(bn)
                    threshold_index = int(size * (sparsity[layer_index+1])) 
                    
                    threhold[layer_index] = m[threshold_index]
                    layer_index = layer_index + 1
                    #print("2",layer_index)
        if conv_flag == "conv3_x":
            for i in range(4):
                for j in conv_index:
                    
                    size = model.conv3_x[i].residual_function[j].weight.data.shape[0]
                    #size = len(model.conv3_x[i].residual_function[j].weight)
                    bn = torch.zeros(size)
                    bn = model.conv3_x[i].residual_function[j].weight.data.abs().clone()
                    
                    m, n = torch.sort(bn)
                    
                    # if sparsity[layer_index+1] > 0.5:
                        
                    #     sparsity[layer_index+1] = sparsity[layer_index+1] * 0.5
  
                    threshold_index = int(size * (sparsity[layer_index+1])) 
                    
                    threhold[layer_index] = m[threshold_index]
                    layer_index = layer_index + 1
                    #print("3",layer_index)
        if conv_flag == "conv4_x":
            for i in range(6):
                for j in conv_index:
                    
                    size = model.conv4_x[i].residual_function[j].weight.data.shape[0]
                    #size = len(model.conv4_x[i].residual_function[j].weight)
                    bn = torch.zeros(size)
                    bn = model.conv4_x[i].residual_function[j].weight.data.abs().clone()
                    
                    m, n = torch.sort(bn)
                    
                    # if sparsity[layer_index+1] > 0.5:
                        
                    #     sparsity[layer_index+1] = sparsity[layer_index+1] * 0.5
  
                    threshold_index = int(size * (sparsity[layer_index+1])) 
                    
                    threhold[layer_index] = m[threshold_index]
                    layer_index = layer_index + 1
                    #print("4",layer_index)
        if conv_flag == "conv5_x":
            for i in range(3):
                for j in conv_index:
                    
                    size = model.conv5_x[i].residual_function[j].weight.data.shape[0]
                    #size = len(model.conv5_x[i].residual_function[j].weight)
                    bn = torch.zeros(size)
                    bn = model.conv5_x[i].residual_function[j].weight.data.abs().clone()
                    
                    m, n = torch.sort(bn)
                    
                    # if sparsity[layer_index+1] > 0.5:
                        
                    #     sparsity[layer_index+1] = sparsity[layer_index+1] * 0.5
  
                    threshold_index = int(size * (sparsity[layer_index+1])) 
                    
                    threhold[layer_index] = m[threshold_index]
                    layer_index = layer_index + 1
                    #print("5",layer_index)
                    
    print(threhold)
# =============================================================================
# "prunning" the rudundant channels
# =============================================================================
    pruned = 0
    #cfg = []
    
    cfg_mask = []
    
    layer_index = 0 #config file index
    cfg_index = 0  # the threshold
    conv_index2 = [1,4]
    
    for conv_flag in block_index:
        
        if conv_flag == "conv2_x":
            for i in range(3):
                for j in conv_index2:
                    
                    weight_copy = model.conv2_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy

                    mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    
                    if int(torch.sum(mask)) >= 2:
                        mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    else:
                        mask = weight_copy.gt(0).float().cpu()

                        
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv2_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv2_x[i].residual_function[j].bias.data.mul_(mask)
                    #cfg.append(int(torch.sum(mask)))
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format(mask.shape[0], int(torch.sum(mask))))
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
         
        if conv_flag == "conv3_x":
            for i in range(4):
                for j in conv_index2:
                    
                    weight_copy = model.conv3_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy
                    mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    
                    if int(torch.sum(mask)) >= 2:
                        mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    else:
                        mask = weight_copy.gt(0).float().cpu()

                        
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv3_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv3_x[i].residual_function[j].bias.data.mul_(mask)
                    #cfg.append(int(torch.sum(mask)))
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format(mask.shape[0], int(torch.sum(mask))))
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv4_x":
            for i in range(6):
                for j in conv_index2:
                    
                    weight_copy = model.conv4_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy
              
                    #mask = weight_copy.gt(thre).float().cuda() 
                    
                    mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    
                    if int(torch.sum(mask)) >= 2:
                        mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    else:
                        mask = weight_copy.gt(0).float().cpu()

                        
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv4_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv4_x[i].residual_function[j].bias.data.mul_(mask)
                    #cfg.append(int(torch.sum(mask)))
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format( mask.shape[0], int(torch.sum(mask))))
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv5_x":
            for i in range(3):
                for j in conv_index2:
                    
                    weight_copy = model.conv5_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy
              
                    #mask = weight_copy.gt(thre).float().cuda() 
                    
                    mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    
                    if int(torch.sum(mask)) >= 2:
                        mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
                    else:
                        mask = weight_copy.gt(0).float().cpu()

                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv5_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv5_x[i].residual_function[j].bias.data.mul_(mask)
                    #cfg.append(int(torch.sum(mask)))
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format( mask.shape[0], int(torch.sum(mask))))
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
                    
# =============================================================================
# 
# =============================================================================
    conv_index = [0,3,6]
    block_index = ["conv2_x","conv3_x","conv4_x","conv5_x"]
    conv_flag = "conv2_x"
    #print(model)
    new_layer_index = 0

# =============================================================================
# load pretrained parameters
# =============================================================================
      
    for conv_flag in block_index:
        
        if conv_flag == "conv2_x":
            for i in range(3):
                #for j in conv_index:
                if i == 0:
                    newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                else:
                    newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                newmodel.conv2_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
                newmodel.conv2_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                newmodel.conv2_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                newmodel.conv2_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, bias=False)
                new_layer_index = new_layer_index + 2
                    #print(newmodel.conv2_x[i].residual_function[j])
        if conv_flag == "conv3_x":
            for i in range(4):
                if i == 0:
                    newmodel.conv3_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                else:
                    newmodel.conv3_x[i].residual_function[0] = nn.Conv2d(512, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                newmodel.conv3_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
                if i == 0:
                    newmodel.conv3_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=2, padding=1, bias=False)
                else:
                    newmodel.conv3_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                    
                #newmodel.conv3_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                newmodel.conv3_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                newmodel.conv3_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 512, kernel_size=1, stride=1, bias=False)
                new_layer_index = new_layer_index + 2
        if conv_flag == "conv4_x":
            for i in range(6):
                if i == 0:
                    newmodel.conv4_x[i].residual_function[0] = nn.Conv2d(512, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                else:
                    newmodel.conv4_x[i].residual_function[0] = nn.Conv2d(1024, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                newmodel.conv4_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
                if i == 0:
                    newmodel.conv4_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=2, padding=1, bias=False)
                else:
                    newmodel.conv4_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                    
                #newmodel.conv4_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                newmodel.conv4_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                newmodel.conv4_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 1024, kernel_size=1, stride=1, bias=False)
                new_layer_index = new_layer_index + 2
        if conv_flag == "conv5_x":
            for i in range(3):
                if i == 0:
                    newmodel.conv5_x[i].residual_function[0] = nn.Conv2d(1024, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                else:
                    newmodel.conv5_x[i].residual_function[0] = nn.Conv2d(2048, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                newmodel.conv5_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
                if i == 0:
                    newmodel.conv5_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=2, padding=1, bias=False)
                else:
                    newmodel.conv5_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                #newmodel.conv5_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                newmodel.conv5_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                newmodel.conv5_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 2048, kernel_size=1, stride=1, bias=False)
                new_layer_index = new_layer_index + 2
                    
# =============================================================================
# Prunning the model
# =============================================================================
    a = 0
    layer_id_in_cfg = 0
    
    cfg_mask_layer_index = 0
        
    start_mask = torch.ones(64)
    
    end_mask = cfg_mask[layer_id_in_cfg]
    
    conv_base_flag = 0
    
    bn_base_flag = 0

    count = 0   
    
    block_index = ["conv2_x","conv3_x","conv4_x","conv5_x"]
    conv_flag = "conv2_x"
    #print(model)
    new_layer_index = 0
    #newmodel = model
      
    for conv_flag in block_index:
        
        if conv_flag == "conv2_x":
            for i in range(3):
                #for j in conv_index:
                if i == 0:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    #print(start_mask,end_mask)
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv2_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()

                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    #sys.exit()
                    newmodel.conv2_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask              
                else:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv2_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv2_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask
                    
# =============================================================================
#               BN layer
#               newmodel.conv2_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
# =============================================================================
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv2_x[i].residual_function[1].weight.data = model.conv2_x[i].residual_function[1].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv2_x[i].residual_function[1].bias.data = model.conv2_x[i].residual_function[1].bias.data[idx1.tolist()].clone()
                newmodel.conv2_x[i].residual_function[1].running_mean = model.conv2_x[i].residual_function[1].running_mean[idx1.tolist()].clone()
                newmodel.conv2_x[i].residual_function[1].running_var = model.conv2_x[i].residual_function[1].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
                
# =============================================================================
#               layer2.0.conv2.weight; layer2.0.bn2.weight;layer2.0.bn2.bias;
#               convolutional layer 
#               newmodel.conv2_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv2_x[i].residual_function[3].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                
                newmodel.conv2_x[i].residual_function[3].weight.data = w1.clone()
# =============================================================================
#               BN layer
# =============================================================================
                #newmodel.conv2_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv2_x[i].residual_function[4].weight.data = model.conv2_x[i].residual_function[4].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv2_x[i].residual_function[4].bias.data = model.conv2_x[i].residual_function[4].bias.data[idx1.tolist()].clone()
                newmodel.conv2_x[i].residual_function[4].running_mean = model.conv2_x[i].residual_function[4].running_mean[idx1.tolist()].clone()
                newmodel.conv2_x[i].residual_function[4].running_var = model.conv2_x[i].residual_function[4].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
     
# =============================================================================
#  newmodel.conv2_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv2_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                #w1 = w1[idx1.tolist(), :, :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv2_x[i].residual_function[6].weight.data = w1.clone()
                start_mask = torch.ones(256)
                
# =============================================================================
#  conv3_x               
# =============================================================================
        if conv_flag == "conv3_x":
            for i in range(4):
                #for j in conv_index:
                #print(start_mask, end_mask)
                if i == 0:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    print(start_mask,end_mask)
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv3_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()

                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    #sys.exit()
                    newmodel.conv3_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask              
                else:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv3_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv3_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask
                    
# =============================================================================
#               BN layer
#               newmodel.conv2_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
# =============================================================================
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv3_x[i].residual_function[1].weight.data = model.conv3_x[i].residual_function[1].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv3_x[i].residual_function[1].bias.data = model.conv3_x[i].residual_function[1].bias.data[idx1.tolist()].clone()
                newmodel.conv3_x[i].residual_function[1].running_mean = model.conv3_x[i].residual_function[1].running_mean[idx1.tolist()].clone()
                newmodel.conv3_x[i].residual_function[1].running_var = model.conv3_x[i].residual_function[1].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
                             
# =============================================================================
#               layer2.0.conv2.weight; layer2.0.bn2.weight;layer2.0.bn2.bias;
#               convolutional layer 
#               newmodel.conv2_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv3_x[i].residual_function[3].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                
                newmodel.conv3_x[i].residual_function[3].weight.data = w1.clone()
# =============================================================================
#               BN layer
# =============================================================================
                #newmodel.conv2_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv3_x[i].residual_function[4].weight.data = model.conv3_x[i].residual_function[4].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv3_x[i].residual_function[4].bias.data = model.conv3_x[i].residual_function[4].bias.data[idx1.tolist()].clone()
                newmodel.conv3_x[i].residual_function[4].running_mean = model.conv3_x[i].residual_function[4].running_mean[idx1.tolist()].clone()
                newmodel.conv3_x[i].residual_function[4].running_var = model.conv3_x[i].residual_function[4].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
     
# =============================================================================
#  newmodel.conv2_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv3_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                #w1 = w1[idx1.tolist(), :, :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv3_x[i].residual_function[6].weight.data = w1.clone()
                
                start_mask = torch.ones(512)

        if conv_flag == "conv4_x":
            for i in range(6):
                #for j in conv_index:
                if i == 0:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    #print(start_mask,end_mask)
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv4_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()

                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    #sys.exit()
                    newmodel.conv4_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask              
                else:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv4_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv4_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask
                    
# =============================================================================
#               BN layer
#               newmodel.conv2_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
# =============================================================================
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv4_x[i].residual_function[1].weight.data = model.conv4_x[i].residual_function[1].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv4_x[i].residual_function[1].bias.data = model.conv4_x[i].residual_function[1].bias.data[idx1.tolist()].clone()
                newmodel.conv4_x[i].residual_function[1].running_mean = model.conv4_x[i].residual_function[1].running_mean[idx1.tolist()].clone()
                newmodel.conv4_x[i].residual_function[1].running_var = model.conv4_x[i].residual_function[1].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
                
# =============================================================================
#               layer2.0.conv2.weight; layer2.0.bn2.weight;layer2.0.bn2.bias;
#               convolutional layer 
#               newmodel.conv2_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv4_x[i].residual_function[3].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                
                newmodel.conv4_x[i].residual_function[3].weight.data = w1.clone()
                
                
                
# =============================================================================
#               BN layer
# =============================================================================
                #newmodel.conv2_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv4_x[i].residual_function[4].weight.data = model.conv4_x[i].residual_function[4].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv4_x[i].residual_function[4].bias.data = model.conv4_x[i].residual_function[4].bias.data[idx1.tolist()].clone()
                newmodel.conv4_x[i].residual_function[4].running_mean = model.conv4_x[i].residual_function[4].running_mean[idx1.tolist()].clone()
                newmodel.conv4_x[i].residual_function[4].running_var = model.conv4_x[i].residual_function[4].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
                
                #print(layer_id_in_cfg)
     
# =============================================================================
#  newmodel.conv2_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv4_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                #w1 = w1[idx1.tolist(), :, :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv4_x[i].residual_function[6].weight.data = w1.clone()
                
                #if i == 5:
                start_mask = torch.ones(1024)

# =============================================================================
# 
# =============================================================================
        if conv_flag == "conv5_x":
            for i in range(3):
                #for j in conv_index:
                if i == 0:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    #print(start_mask,end_mask)
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv5_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    # print(idx0)
                    # print(idx1)
                    # sys.exit()
                    #print(idx1.tolist())
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    #sys.exit()
                    newmodel.conv5_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask              
                else:
                    #newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, padding=1, bias=False)
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv5_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv5_x[i].residual_function[0].weight.data = w1.clone()
                    #start_mask = end_mask
                    
# =============================================================================
#               BN layer
#               newmodel.conv2_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
# =============================================================================
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv5_x[i].residual_function[1].weight.data = model.conv5_x[i].residual_function[1].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv5_x[i].residual_function[1].bias.data = model.conv5_x[i].residual_function[1].bias.data[idx1.tolist()].clone()
                newmodel.conv5_x[i].residual_function[1].running_mean = model.conv5_x[i].residual_function[1].running_mean[idx1.tolist()].clone()
                newmodel.conv5_x[i].residual_function[1].running_var = model.conv5_x[i].residual_function[1].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
                layer_id_in_cfg = layer_id_in_cfg + 1
                
                print(layer_id_in_cfg)
                             
# =============================================================================
#               layer2.0.conv2.weight; layer2.0.bn2.weight;layer2.0.bn2.bias;
#               convolutional layer 
#               newmodel.conv2_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv5_x[i].residual_function[3].weight.data[:, idx0.tolist(), :, :].clone()
                
                w1 = w1[idx1.tolist(), :, :, :].clone()
                
                newmodel.conv5_x[i].residual_function[3].weight.data = w1.clone()
                
# =============================================================================
#               BN layer
# =============================================================================
                #newmodel.conv2_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                newmodel.conv5_x[i].residual_function[4].weight.data = model.conv5_x[i].residual_function[4].weight.data[idx1.tolist()].clone()# convert array to list
                newmodel.conv5_x[i].residual_function[4].bias.data = model.conv5_x[i].residual_function[4].bias.data[idx1.tolist()].clone()
                newmodel.conv5_x[i].residual_function[4].running_mean = model.conv5_x[i].residual_function[4].running_mean[idx1.tolist()].clone()
                newmodel.conv5_x[i].residual_function[4].running_var = model.conv5_x[i].residual_function[4].running_var[idx1.tolist()].clone()            
                
                start_mask = cfg_mask[layer_id_in_cfg]
                
                if layer_id_in_cfg < 31:
                    
                    end_mask = cfg_mask[layer_id_in_cfg + 1]
                    
                #else:
                    
                    #end_mask = cfg_mask[layer_id_in_cfg]
                
                #if layer_id_in_cfg <= 31:
                
                # if layer_id_in_cfg < 30:
                    
                layer_id_in_cfg = layer_id_in_cfg + 1
                
                print("last",layer_id_in_cfg)
     
# =============================================================================
#  newmodel.conv2_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, padding=1, bias=False)
# =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                #idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                print('3 In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv5_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                #w1 = w1[idx1.tolist(), :, :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv5_x[i].residual_function[6].weight.data = w1.clone()
                
                start_mask = torch.ones(2048)

                print("success")
      
    save_path1 = os.path.join('pruned_model', 'pruned_para.pth')

    torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 
    sys.exit()

def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir',    type=str,  default='dataset',     help='Path to test & train datasets. Default is dataset')
    ap.add_argument('-b', '--batchsize',   type=int,  default=20,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=50,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.0001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('-m', '--float_model', type=str,  default='float_model', help='Path to folder where trained model is saved. Default is float_model')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--dset_dir     : ',args.dset_dir)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print ('--float_model  : ',args.float_model)
    print(DIVIDER)

    train_test(args.dset_dir, args.batchsize, args.learnrate, args.epochs, args.float_model)

    return


if __name__ == '__main__':
    run_main()
