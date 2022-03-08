'''
Author: IHPC Ella
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
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
from torchvision import models
from common import *
import pandas as pd

DIVIDER = '-----------------------------------------'

savepath_empty = os.path.join('pruned_model', 'pruned0.pth')
save_path_new = os.path.join('pruned_model', 'pruned_group.pth')

# additional subgradient descent on the sparsity-induced penalty term

def pruning_operation(dset_dir, batchsize, learnrate, epochs, float_model,num1,num2,num3):

    device = torch.device('cpu')
# =============================================================================
# create the model
# =============================================================================
    model = CNN().to(device)
    
    #save_path_new = os.path.join('pruned_model', 'pruned1.pth')
    #model = torch.load(save_path_new)
    
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
    #print(model)
# =============================================================================
# caculate thresholds 
# =============================================================================

    threhold1 = torch.zeros(13)
    threhold2 = torch.zeros(13)
    index = 0
    layer_index = 0 
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if layer_index <= 11:
                size = m.weight.data.shape[0]
                bn = torch.zeros(size)
                bn = m.weight.data.abs().clone()
                y, i = torch.sort(bn)
                
                threshold_index1 = int(size * num1) 
                # index2 = num*0.05+0.05
                if num2 >= 1:
                    num2 = num2 -1
                threshold_index2 = int(size * (num2))
                threhold1[layer_index] = y[threshold_index1]
                
                threhold2[layer_index] = y[threshold_index2]
  
            layer_index = layer_index + 1

    pruned = 0
    cfg = []
    cfg_mask = []
    layer_index = 0
# =============================================================================
# caculate the cfg_mask file
# =============================================================================
    layer_index = 0
    for k, m in enumerate(model.modules()):
        #print("pruned",pruned)
        if isinstance(m, nn.BatchNorm2d):
            layer_index = layer_index + 1
            if layer_index <= 12:
                weight_copy = m.weight.data.abs().clone()
                weight_copy = weight_copy
    
                mask = weight_copy.gt(0).float().cpu()
                mask1 = weight_copy.lt(threhold1[layer_index-1]).float().cpu()
                mask2 = weight_copy.gt(threhold2[layer_index-1]).float().cpu()
                
                for i in range(len(mask1)):
                    mask[i] = mask1[i] or mask2[i]                   
                
                if num3 == 1:
                    #print(mask)
                    mask = mask1
                
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                #print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    #format(k, mask.shape[0], int(torch.sum(mask))))
            else:
                weight_copy = m.weight.data.abs().clone()

                #mask = weight_copy.gt(thre).float().cuda()
                mask = weight_copy.gt(0).float().cpu()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                #print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                   # format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d): 
            cfg.append('M')      
        
    print(cfg)

# =============================================================================
# create the newmodel according to cfg file
# =============================================================================
    
    #save_path1 = os.path.join('pruned_model', 'pruned_group.pth')
    
    newmodel =  vgg('vgg13_bn', cfg, True, False, True)

    #newmodel = torch.load(save_path_new)
    
    torch.save(newmodel, savepath_empty,_use_new_zipfile_serialization=False) 
    
# =============================================================================
# Prunning the model
# =============================================================================
    a = 0
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):

        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
                
        elif isinstance(m0, nn.Conv2d):
            if a <= 11:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()

            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                #idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                end_mask = torch.ones(100)
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))

                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        
            a = a + 1

# =============================================================================
#     save the model and the parameters
# =============================================================================
    
    torch.save(newmodel, save_path_new,_use_new_zipfile_serialization=False) 

















