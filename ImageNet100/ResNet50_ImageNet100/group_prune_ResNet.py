
'''
Ella
'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import models
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
#from resnet import *
import pandas as pd
from resnet_cifar import resnet50

cfg: List[Union[int]] = [i for i in range(100)]

DIVIDER = '-----------------------------------------'

def group_pruning(dset_dir, batchsize, learnrate, epochs, float_model,num1,num2,num3,test_f):

    device = torch.device('cpu')
# =============================================================================
# create the model
# =============================================================================

    model = resnet50()
    
    model.to(device)
    
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
# =============================================================================
# pruned template
# =============================================================================

    newmodel = resnet50()
    
    newmodel.to(device)
    
    newmodel.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
    
# =============================================================================
# calculate the treshold according to the prunning ratio
# =============================================================================

    threhold1 = torch.zeros(100)
    threhold2 = torch.zeros(100)
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
                    
                    threshold_index1 = int(size * num1) 
                    if num2 >= 1:
                        num2 = num2 -1
                        
                    threshold_index2 = int(size * (num2))
                    
                    threhold1[layer_index] = m[threshold_index1]
                    
                    threhold2[layer_index] = m[threshold_index2]
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv3_x":
            for i in range(4):
                for j in conv_index:
                    
                    size = model.conv3_x[i].residual_function[j].weight.data.shape[0]
                    bn = torch.zeros(size)
                    bn = model.conv3_x[i].residual_function[j].weight.data.abs().clone()
                    
                    m, n = torch.sort(bn)
                    
                    threshold_index1 = int(size * num1) 
                    if num2 >= 1:
                        num2 = num2 -1
                        
                    threshold_index2 = int(size * (num2))
                    
                    threhold1[layer_index] = m[threshold_index1]
                    
                    threhold2[layer_index] = m[threshold_index2]
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv4_x":
            for i in range(6):
                for j in conv_index:
                    
                    size = model.conv4_x[i].residual_function[j].weight.data.shape[0]
                    bn = torch.zeros(size)
                    bn = model.conv4_x[i].residual_function[j].weight.data.abs().clone()
                    
                    m, n = torch.sort(bn)
                    
                    threshold_index1 = int(size * num1) 
                    if num2 >= 1:
                        num2 = num2 -1
                        
                    threshold_index2 = int(size * (num2))
                    
                    threhold1[layer_index] = m[threshold_index1]
                    
                    threhold2[layer_index] = m[threshold_index2]
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv5_x":
            for i in range(3):
                for j in conv_index:
                    
                    size = model.conv5_x[i].residual_function[j].weight.data.shape[0]
                    bn = torch.zeros(size)
                    bn = model.conv5_x[i].residual_function[j].weight.data.abs().clone()
                    
                    m, n = torch.sort(bn)
                    
                    threshold_index1 = int(size * num1) 
                    if num2 >= 1:
                        num2 = num2 -1
                        
                    threshold_index2 = int(size * (num2))
                    
                    threhold1[layer_index] = m[threshold_index1]
                    
                    threhold2[layer_index] = m[threshold_index2]
                    
                    layer_index = layer_index + 1
                                    
# =============================================================================
# calculate cfg_mask file
# =============================================================================
    pruned = 0
  
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
              
                    mask = weight_copy.gt(0).float().cpu()
                    
                    mask1 = weight_copy.lt(threhold1[layer_index]).float().cpu()
                    mask2 = weight_copy.gt(threhold2[layer_index]).float().cpu()
                    
                    for i_m in range(len(mask)):
                        
                        mask[i_m] = mask1[i_m] or mask2[i_m]
                        
                    if num3 == 1:
                        
                        mask = weight_copy.lt(threhold1[layer_index]).float().cpu()
                                           
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv2_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv2_x[i].residual_function[j].bias.data.mul_(mask)
                    #cfg.append(int(torch.sum(mask)))
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    if test_f == 1:
                        print('\t total channel: {:d} \t remaining channel: {:d}'.
                            format(mask.shape[0], int(torch.sum(mask)))) 

                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv3_x":
            for i in range(4):
                for j in conv_index2:
                    
                    weight_copy = model.conv3_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy
              
                    mask = weight_copy.gt(0).float().cpu()
                    mask1 = weight_copy.lt(threhold1[layer_index]).float().cpu()
                    mask2 = weight_copy.gt(threhold2[layer_index]).float().cpu()
                    
                    for i_m in range(len(mask)):
                        
                        mask[i_m] = mask1[i_m] or mask2[i_m]
                        
                    if num3 == 1:
                        
                        mask = weight_copy.lt(threhold1[layer_index]).float().cpu()
                        
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv3_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv3_x[i].residual_function[j].bias.data.mul_(mask)
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    if test_f == 1:
                        print('\t total channel: {:d} \t remaining channel: {:d}'.
                            format(mask.shape[0], int(torch.sum(mask)))) 
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv4_x":
            for i in range(6):
                for j in conv_index2:
                    
                    weight_copy = model.conv4_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy
              
                    mask = weight_copy.gt(0).float().cpu()
                    mask1 = weight_copy.lt(threhold1[layer_index]).float().cpu()
                    mask2 = weight_copy.gt(threhold2[layer_index]).float().cpu()
                    
                    for i_m in range(len(mask)):
                       
                        mask[i_m] = mask1[i_m] or mask2[i_m]
                       
                    if num3 == 1:
                        
                        mask = weight_copy.lt(threhold1[layer_index]).float().cpu()
                        
                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv4_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv4_x[i].residual_function[j].bias.data.mul_(mask)
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    if test_f == 1:
                        print('\t total channel: {:d} \t remaining channel: {:d}'.
                            format(mask.shape[0], int(torch.sum(mask)))) 
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1
        if conv_flag == "conv5_x":
            for i in range(3):
                for j in conv_index2:
                    
                    weight_copy = model.conv5_x[i].residual_function[j].weight.data.abs().clone()
                    weight_copy = weight_copy
              
                    mask = weight_copy.gt(0).float().cpu()
                    mask1 = weight_copy.lt(threhold1[layer_index]).float().cpu()
                    mask2 = weight_copy.gt(threhold2[layer_index]).float().cpu()
                    
                    for i_m in range(len(mask)):
                        
                        mask[i_m] = mask1[i_m] or mask2[i_m]
                    
                    if num3 == 1:
                        
                        mask = weight_copy.lt(threhold1[layer_index]).float().cpu()

                    pruned = pruned + mask.shape[0] - torch.sum(mask)
                    model.conv5_x[i].residual_function[j].weight.data.mul_(mask)
                    model.conv5_x[i].residual_function[j].bias.data.mul_(mask)
                    #cfg.append(int(torch.sum(mask)))
                    
                    cfg[layer_index] = int(torch.sum(mask))
                    cfg_mask.append(mask.clone())
                    
                    if test_f == 1:
                        print('\t total channel: {:d} \t remaining channel: {:d}'.
                            format(mask.shape[0], int(torch.sum(mask)))) 
                    
                    cfg_index = cfg_index + 1
                    
                    layer_index = layer_index + 1

# =============================================================================
# 
# =============================================================================
    conv_index = [0,3,6]
    block_index = ["conv2_x","conv3_x","conv4_x","conv5_x"]
    conv_flag = "conv2_x"
    new_layer_index = 0
    
    for conv_flag in block_index:
        
        if conv_flag == "conv2_x":
            for i in range(3):
                if i == 0:
                    newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                else:
                    newmodel.conv2_x[i].residual_function[0] = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
                newmodel.conv2_x[i].residual_function[1] = nn.BatchNorm2d(cfg[new_layer_index])
                newmodel.conv2_x[i].residual_function[3] = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                newmodel.conv2_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                newmodel.conv2_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, bias=False)
                new_layer_index = new_layer_index + 2
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
                newmodel.conv5_x[i].residual_function[4] = nn.BatchNorm2d(cfg[new_layer_index+1])
                newmodel.conv5_x[i].residual_function[6] = nn.Conv2d(cfg[new_layer_index+1], 2048, kernel_size=1, stride=1, bias=False)
                new_layer_index = new_layer_index + 2
                                   
# =============================================================================
# create the newmodel according to cfg file
# =============================================================================
    savepath = os.path.join('pruned_model', 'pruned0.pth')
    
    torch.save(newmodel, savepath,_use_new_zipfile_serialization=False) 
    
# =============================================================================
# Prunning the model
# =============================================================================

    layer_id_in_cfg = 0
    
    cfg_mask_layer_index = 0
        
    start_mask = torch.ones(64)
    
    end_mask = cfg_mask[layer_id_in_cfg]

    block_index = ["conv2_x","conv3_x","conv4_x","conv5_x"]
    conv_flag = "conv2_x"
    new_layer_index = 0
      
    for conv_flag in block_index:
# =============================================================================
# conv2_x      
# =============================================================================
        if conv_flag == "conv2_x":
            for i in range(3):
                if i == 0:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))

                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv2_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()

                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv2_x[i].residual_function[0].weight.data = w1.clone()
                else:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv2_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv2_x[i].residual_function[0].weight.data = w1.clone()
                    
                # =============================================================================
                #               BN layer
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
                #               convolutional layer 
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
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
                #  conv3
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv2_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv2_x[i].residual_function[6].weight.data = w1.clone()

                start_mask = torch.ones(256)
                
# =============================================================================
#  conv3_x               
# =============================================================================
        if conv_flag == "conv3_x":
            for i in range(4):

                if i == 0:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    #print(start_mask,end_mask)
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv3_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()

                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv3_x[i].residual_function[0].weight.data = w1.clone()
                else:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv3_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv3_x[i].residual_function[0].weight.data = w1.clone()
                    
                # =============================================================================
                #               BN layer
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
                #               convolutional layer 
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
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
                # conv3
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv3_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv3_x[i].residual_function[6].weight.data = w1.clone()
                
                start_mask = torch.ones(512)

# =============================================================================
# conv4_x
# =============================================================================
        if conv_flag == "conv4_x":
            for i in range(6):
                if i == 0:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv4_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()

                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv4_x[i].residual_function[0].weight.data = w1.clone()
                else:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv4_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv4_x[i].residual_function[0].weight.data = w1.clone()
                    
                # =============================================================================
                #               BN layer
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
                #               convolutional layer 
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
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
                     
                # =============================================================================
                #  
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv4_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv4_x[i].residual_function[6].weight.data = w1.clone()
                
                start_mask = torch.ones(1024)
# =============================================================================
# conv5_x
# =============================================================================
        if conv_flag == "conv5_x":
            for i in range(3):
                #for j in conv_index:
                if i == 0:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv5_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv5_x[i].residual_function[0].weight.data = w1.clone()
                else:
                    # =============================================================================
                    # conv layer
                    # =============================================================================
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    if test_f == 1:
                        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w1 = model.conv5_x[i].residual_function[0].weight.data[:, idx0.tolist(), :, :].clone()
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    newmodel.conv5_x[i].residual_function[0].weight.data = w1.clone()
                    
                # =============================================================================
                #               BN layer
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
                
                #print(layer_id_in_cfg)
                # =============================================================================
                #               convolutional layer 
                # =============================================================================
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
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
                     
                layer_id_in_cfg = layer_id_in_cfg + 1
                
                #print("last",layer_id_in_cfg)
     
                # =============================================================================
                # conv3
                # =============================================================================

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                #idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if test_f == 1:
                    print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = model.conv5_x[i].residual_function[6].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[:, :, :, :].clone()
                
                newmodel.conv5_x[i].residual_function[6].weight.data = w1.clone()
                
                start_mask = torch.ones(2048)

    save_path1 = os.path.join('pruned_model', 'pruned_para.pth')

    torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 

    

