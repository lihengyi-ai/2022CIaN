
'''
Author: Ella
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

import torch.nn as nn
from resnet import *
from common import *
import pandas as pd
from resnet_cifar import resnet50

float_model = "float_model"
test_f = 1


cfg: List[Union[int]] = [i for i in range(100)]

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

DIVIDER = '-----------------------------------------'

#def train_test1(dset_dir, batchsize, learnrate, epochs, float_model,num1,num2,num3,test_f):
    
device = torch.device('cpu')
# =============================================================================
# create the model
# =============================================================================

model = ResNet()


model.to(device)

model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
# =============================================================================
#   creat newmodel template
# =============================================================================
newmodel = ResNet()

newmodel.to(device)

newmodel.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))

# =============================================================================
# calculate the tresholds according to the prunning ratio
# =============================================================================

threhold = torch.zeros(100)
#threhold2 = torch.zeros(100)
layer_index = 0
first_identity = 0
block_index = ["layer1","layer2","layer3","layer4"]
bn_value = ["bn1","bn2"]
for conv_flag in block_index:
    
    if conv_flag == "layer1":
        for i in range(3):  
            for bn_index in bn_value:
                if bn_index == "bn1":                    
                    size = model.layer1[i].bn1.weight.data.shape[0]      
                    bn = torch.zeros(size)
                    bn = model.layer1[i].bn1.weight.data.abs().clone()                        
                elif bn_index == "bn2":                    
                    size = model.layer1[i].bn2.weight.data.shape[0]                            
                    bn = torch.zeros(size)                        
                    bn = model.layer1[i].bn2.weight.data.abs().clone()
                
                m, n = torch.sort(bn)
                threshold_index = int(size * (sparsity[layer_index+1])) 
                threhold[layer_index] = m[threshold_index]
                layer_index = layer_index + 1
                    
    if conv_flag == "layer2":
        for i in range(4):
            for bn_index in bn_value:                    
                if bn_index == "bn1":                    
                    size = model.layer2[i].bn1.weight.data.shape[0]      
                    bn = torch.zeros(size)
                    bn = model.layer2[i].bn1.weight.data.abs().clone()                        
                elif bn_index == "bn2":                    
                    size = model.layer2[i].bn2.weight.data.shape[0]                            
                    bn = torch.zeros(size)                        
                    bn = model.layer2[i].bn2.weight.data.abs().clone()    
                m, n = torch.sort(bn)
                threshold_index = int(size * (sparsity[layer_index+1])) 
                threhold[layer_index] = m[threshold_index]
                layer_index = layer_index + 1
                             
    if conv_flag == "layer3":
        for i in range(6):
            for bn_index in bn_value:                    
                if bn_index == "bn1":                    
                    size = model.layer3[i].bn1.weight.data.shape[0]      
                    bn = torch.zeros(size)
                    bn = model.layer3[i].bn1.weight.data.abs().clone()                        
                elif bn_index == "bn2":                    
                    size = model.layer3[i].bn2.weight.data.shape[0]                            
                    bn = torch.zeros(size)                        
                    bn = model.layer3[i].bn2.weight.data.abs().clone()   
                m, n = torch.sort(bn)
                threshold_index = int(size * (sparsity[layer_index+1])) 
                threhold[layer_index] = m[threshold_index]
                layer_index = layer_index + 1
                              
    if conv_flag == "layer4":
        for i in range(3):
            for bn_index in bn_value:
                
                if bn_index == "bn1":                    
                    size = model.layer4[i].bn1.weight.data.shape[0]      
                    bn = torch.zeros(size)
                    bn = model.layer4[i].bn1.weight.data.abs().clone()                        
                elif bn_index == "bn2":                    
                    size = model.layer4[i].bn2.weight.data.shape[0]                            
                    bn = torch.zeros(size)                        
                    bn = model.layer4[i].bn2.weight.data.abs().clone()
                m, n = torch.sort(bn)
                threshold_index = int(size * (sparsity[layer_index+1])) 
                threhold[layer_index] = m[threshold_index]
                layer_index = layer_index + 1
                                                
# =============================================================================
# calculate the cfg_mask files
# =============================================================================
pruned = 0

cfg_mask = []
layer_index = 0
cfg_index = 0  

for conv_flag in block_index:
    
    if conv_flag == "layer1":
        for i in range(3):
            
            for bn_index in bn_value:

                if bn_index == "bn1":
                    weight_copy = model.layer1[i].bn1.weight.data.abs().clone() 
                elif bn_index == "bn2":
                    weight_copy = model.layer1[i].bn2.weight.data.abs().clone() 
  
                weight_copy = weight_copy         
                
                mask = weight_copy.gt(threhold[layer_index]).float().cpu()
                                    
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                
                if bn_index == "bn1":
                    model.layer1[i].bn1.weight.data.mul_(mask)
                    model.layer1[i].bn1.weight.data.mul_(mask)
                elif bn_index == "bn2":
                    model.layer1[i].bn2.weight.data.mul_(mask)
                    model.layer1[i].bn2.weight.data.mul_(mask)     
                    
                cfg[layer_index] = int(torch.sum(mask))
                cfg_mask.append(mask.clone())
                
                if test_f == 1:
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format(mask.shape[0], int(torch.sum(mask))))
                
                cfg_index = cfg_index + 1
                layer_index = layer_index + 1
                   
    if conv_flag == "layer2":
        for i in range(4):
            for bn_index in bn_value:

                if bn_index == "bn1":
                    #weight_copy = model.conv2_x[i_p].residual_function[j].weight.data.abs().clone()
                    weight_copy = model.layer2[i].bn1.weight.data.abs().clone() 
                elif bn_index == "bn2":
                    weight_copy = model.layer2[i].bn2.weight.data.abs().clone() 
   
                weight_copy = weight_copy
                
                mask = weight_copy.gt(threhold[layer_index]).float().cpu()
                
                    
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                
                if bn_index == "bn1":
                    model.layer2[i].bn1.weight.data.mul_(mask)
                    model.layer2[i].bn1.weight.data.mul_(mask)
                elif bn_index == "bn2":
                    model.layer2[i].bn2.weight.data.mul_(mask)
                    model.layer2[i].bn2.weight.data.mul_(mask)     
                
                cfg[layer_index] = int(torch.sum(mask))
                cfg_mask.append(mask.clone())
                
                if test_f == 1:
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format(mask.shape[0], int(torch.sum(mask))))
                
                cfg_index = cfg_index + 1
                
                layer_index = layer_index + 1
    if conv_flag == "layer3":
        for i in range(6):
            for bn_index in bn_value:

                if bn_index == "bn1":
                    weight_copy = model.layer3[i].bn1.weight.data.abs().clone() 
                elif bn_index == "bn2":
                    weight_copy = model.layer3[i].bn2.weight.data.abs().clone() 
 
                weight_copy = weight_copy
                
                mask = weight_copy.gt(threhold[layer_index]).float().cpu()
                    
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                
                if bn_index == "bn1":
                    model.layer3[i].bn1.weight.data.mul_(mask)
                    model.layer3[i].bn1.weight.data.mul_(mask)
                elif bn_index == "bn2":
                    model.layer3[i].bn2.weight.data.mul_(mask)
                    model.layer3[i].bn2.weight.data.mul_(mask)     
                
                cfg[layer_index] = int(torch.sum(mask))
                cfg_mask.append(mask.clone())
                
                if test_f == 1:
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format(mask.shape[0], int(torch.sum(mask))))                        
                
                cfg_index = cfg_index + 1
                layer_index = layer_index + 1
                
    if conv_flag == "layer4":
        for i in range(3):
            for bn_index in bn_value:

                if bn_index == "bn1":
                    weight_copy = model.layer4[i].bn1.weight.data.abs().clone() 
                elif bn_index == "bn2":
                    weight_copy = model.layer4[i].bn2.weight.data.abs().clone() 
                                        
                weight_copy = weight_copy
                
                mask = weight_copy.gt(threhold[layer_index]).float().cpu()    
                    
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                
                if bn_index == "bn1":
                    model.layer4[i].bn1.weight.data.mul_(mask)
                    model.layer4[i].bn1.weight.data.mul_(mask)
                elif bn_index == "bn2":
                    model.layer4[i].bn2.weight.data.mul_(mask)
                    model.layer4[i].bn2.weight.data.mul_(mask)     
                    
                cfg[layer_index] = int(torch.sum(mask))
                cfg_mask.append(mask.clone())
                
                if test_f == 1:
                    print('\t total channel: {:d} \t remaining channel: {:d}'.
                        format(mask.shape[0], int(torch.sum(mask))))
                
                cfg_index = cfg_index + 1
                
                layer_index = layer_index + 1

# =============================================================================
# modify the template to be the pruned network
# =============================================================================

block_index = ["layer1","layer2","layer3","layer4"]

new_layer_index = 0
 
for conv_flag in block_index:
    
    if conv_flag == "layer1":
        for i in range(3):
            #for j in conv_index:
            if i == 0:
                newmodel.layer1[i].conv1 = nn.Conv2d(64, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            else:
                newmodel.layer1[i].conv1 = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            newmodel.layer1[i].bn1 = nn.BatchNorm2d(cfg[new_layer_index])
            newmodel.layer1[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
            newmodel.layer1[i].bn2 = nn.BatchNorm2d(cfg[new_layer_index+1])
            newmodel.layer1[i].conv3 = nn.Conv2d(cfg[new_layer_index+1], 256, kernel_size=1, stride=1, bias=False)
            new_layer_index = new_layer_index + 2
            
    if conv_flag == "layer2":
        for i in range(4):
            if i == 0:
                newmodel.layer2[i].conv1 = nn.Conv2d(256, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            else:
                newmodel.layer2[i].conv1 = nn.Conv2d(512, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            newmodel.layer2[i].bn1 = nn.BatchNorm2d(cfg[new_layer_index])
            if i == 0:
                newmodel.layer2[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=2, padding=1, bias=False)
            else:
                newmodel.layer2[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                
            newmodel.layer2[i].bn2 = nn.BatchNorm2d(cfg[new_layer_index+1])
            newmodel.layer2[i].conv3 = nn.Conv2d(cfg[new_layer_index+1], 512, kernel_size=1, stride=1, bias=False)
            new_layer_index = new_layer_index + 2
    if conv_flag == "layer3":
        for i in range(6):
            if i == 0:
                newmodel.layer3[i].conv1 = nn.Conv2d(512, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            else:
                newmodel.layer3[i].conv1 = nn.Conv2d(1024, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            newmodel.layer3[i].bn1 = nn.BatchNorm2d(cfg[new_layer_index])
            if i == 0:
                newmodel.layer3[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=2, padding=1, bias=False)
            else:
                newmodel.layer3[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
                
            newmodel.layer3[i].bn2 = nn.BatchNorm2d(cfg[new_layer_index+1])
            newmodel.layer3[i].conv3 = nn.Conv2d(cfg[new_layer_index+1], 1024, kernel_size=1, stride=1, bias=False)
            new_layer_index = new_layer_index + 2
    if conv_flag == "layer4":
        for i in range(3):
            if i == 0:
                newmodel.layer4[i].conv1 = nn.Conv2d(1024, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            else:
                newmodel.layer4[i].conv1 = nn.Conv2d(2048, cfg[new_layer_index], kernel_size=1, stride=1, bias=False)
            newmodel.layer4[i].bn1 = nn.BatchNorm2d(cfg[new_layer_index])
            if i == 0:
                newmodel.layer4[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=2, padding=1, bias=False)
            else:
                newmodel.layer4[i].conv2 = nn.Conv2d(cfg[new_layer_index], cfg[new_layer_index+1], kernel_size=3, stride=1, padding=1, bias=False)
            newmodel.layer4[i].bn2 = nn.BatchNorm2d(cfg[new_layer_index+1])
            newmodel.layer4[i].conv3 = nn.Conv2d(cfg[new_layer_index+1], 2048, kernel_size=1, stride=1, bias=False)
            new_layer_index = new_layer_index + 2
            
                
# =============================================================================
# save an empty model
# =============================================================================
savepath = os.path.join('pruned_model', 'pruned_empty.pth')

torch.save(newmodel, savepath,_use_new_zipfile_serialization=False) 

# =============================================================================
# Prunning the model
# =============================================================================
layer_id_in_cfg = 0  
start_mask = torch.ones(64)
end_mask = cfg_mask[layer_id_in_cfg]  
new_layer_index = 0
  
for conv_flag in block_index:
    
    if conv_flag == "layer1":
        #print("layer1")
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
                w1 = model.layer1[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()

                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer1[i].conv1.weight.data = w1.clone()
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
                w1 = model.layer1[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer1[i].conv1.weight.data = w1.clone()                    
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer1[i].bn1.weight.data = model.layer1[i].bn1.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer1[i].bn1.bias.data = model.layer1[i].bn1.bias.data[idx1.tolist()].clone()
            newmodel.layer1[i].bn1.running_mean = model.layer1[i].bn1.running_mean[idx1.tolist()].clone()
            newmodel.layer1[i].bn1.running_var = model.layer1[i].bn1.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
                                         
            # =============================================================================
            #              conv2
            # =============================================================================

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            if test_f == 1:
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = model.layer1[i].conv2.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            
            newmodel.layer1[i].conv2.weight.data = w1.clone()
            
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer1[i].bn2.weight.data = model.layer1[i].bn2.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer1[i].bn2.bias.data = model.layer1[i].bn2.bias.data[idx1.tolist()].clone()
            newmodel.layer1[i].bn2.running_mean = model.layer1[i].bn2.running_mean[idx1.tolist()].clone()
            newmodel.layer1[i].bn2.running_var = model.layer1[i].bn2.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
 
            # =============================================================================
            #              conv3
            # =============================================================================

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            if test_f == 1:
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = model.layer1[i].conv3.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[:, :, :, :].clone()
            
            newmodel.layer1[i].conv3.weight.data = w1.clone()

            start_mask = torch.ones(256)
            
# =============================================================================
#  layer2              
# =============================================================================
    if conv_flag == "layer2":
        #print("layer2")
        for i in range(4):

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
                w1 = model.layer2[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()

                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer2[i].conv1.weight.data = w1.clone()
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
                w1 = model.layer2[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer2[i].conv1.weight.data = w1.clone()                    
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer2[i].bn1.weight.data = model.layer2[i].bn1.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer2[i].bn1.bias.data = model.layer2[i].bn1.bias.data[idx1.tolist()].clone()
            newmodel.layer2[i].bn1.running_mean = model.layer2[i].bn1.running_mean[idx1.tolist()].clone()
            newmodel.layer2[i].bn1.running_var = model.layer2[i].bn1.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
            
            # =============================================================================
            #               
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
            w1 = model.layer2[i].conv2.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            
            newmodel.layer2[i].conv2.weight.data = w1.clone()
            
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer2[i].bn2.weight.data = model.layer2[i].bn2.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer2[i].bn2.bias.data = model.layer2[i].bn2.bias.data[idx1.tolist()].clone()
            newmodel.layer2[i].bn2.running_mean = model.layer2[i].bn2.running_mean[idx1.tolist()].clone()
            newmodel.layer2[i].bn2.running_var = model.layer2[i].bn2.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
                 
            # =============================================================================
            #               conv3
            # =============================================================================

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            if test_f == 1:
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = model.layer2[i].conv3.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[:, :, :, :].clone()
            
            newmodel.layer2[i].conv3.weight.data = w1.clone()
            
            start_mask = torch.ones(512)
# =============================================================================
# layer3
# =============================================================================
    if conv_flag == "layer3":
        #print("layer3")
        for i in range(6):
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
                w1 = model.layer3[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()

                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer3[i].conv1.weight.data = w1.clone()
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
                w1 = model.layer3[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer3[i].conv1.weight.data = w1.clone()                    
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer3[i].bn1.weight.data = model.layer3[i].bn1.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer3[i].bn1.bias.data = model.layer3[i].bn1.bias.data[idx1.tolist()].clone()
            newmodel.layer3[i].bn1.running_mean = model.layer3[i].bn1.running_mean[idx1.tolist()].clone()
            newmodel.layer3[i].bn1.running_var = model.layer3[i].bn1.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
                                         
            # =============================================================================
            #               conv2
            # =============================================================================

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            if test_f == 1:
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = model.layer3[i].conv2.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            
            newmodel.layer3[i].conv2.weight.data = w1.clone()
             
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer3[i].bn2.weight.data = model.layer3[i].bn2.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer3[i].bn2.bias.data = model.layer3[i].bn2.bias.data[idx1.tolist()].clone()
            newmodel.layer3[i].bn2.running_mean = model.layer3[i].bn2.running_mean[idx1.tolist()].clone()
            newmodel.layer3[i].bn2.running_var = model.layer3[i].bn2.running_var[idx1.tolist()].clone()            
            
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
            w1 = model.layer3[i].conv3.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[:, :, :, :].clone()
            
            newmodel.layer3[i].conv3.weight.data = w1.clone()
            
            start_mask = torch.ones(1024)

# =============================================================================
# layer4
# =============================================================================
    if conv_flag == "layer4":
        #print("layer4")
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
                w1 = model.layer4[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()
                
                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer4[i].conv1.weight.data = w1.clone()
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
                w1 = model.layer4[i].conv1.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                newmodel.layer4[i].conv1.weight.data = w1.clone()
                
            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer4[i].bn1.weight.data = model.layer4[i].bn1.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer4[i].bn1.bias.data = model.layer4[i].bn1.bias.data[idx1.tolist()].clone()
            newmodel.layer4[i].bn1.running_mean = model.layer4[i].bn1.running_mean[idx1.tolist()].clone()
            newmodel.layer4[i].bn1.running_var = model.layer4[i].bn1.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
                                         
            # =============================================================================
            #               conv2
            # =============================================================================

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            if test_f == 1:
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = model.layer4[i].conv2.weight.data[:, idx0.tolist(), :, :].clone()
            
            w1 = w1[idx1.tolist(), :, :, :].clone()
            
            newmodel.layer4[i].conv2.weight.data = w1.clone()

            # =============================================================================
            #               BN layer
            # =============================================================================
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy()))) #return the indexes of none zero data
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            newmodel.layer4[i].bn2.weight.data = model.layer4[i].bn2.weight.data[idx1.tolist()].clone()# convert array to list
            newmodel.layer4[i].bn2.bias.data = model.layer4[i].bn2.bias.data[idx1.tolist()].clone()
            newmodel.layer4[i].bn2.running_mean = model.layer4[i].bn2.running_mean[idx1.tolist()].clone()
            newmodel.layer4[i].bn2.running_var = model.layer4[i].bn2.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            if layer_id_in_cfg < 31:
                
                end_mask = cfg_mask[layer_id_in_cfg + 1]
                
            layer_id_in_cfg = layer_id_in_cfg + 1
            # =============================================================================
            # conv3
            # =============================================================================
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            
            if test_f == 1:
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = model.layer4[i].conv3.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[:, :, :, :].clone()
            
            newmodel.layer4[i].conv3.weight.data = w1.clone()
            
            start_mask = torch.ones(2048)

save_path1 = os.path.join('pruned_model', 'pruned_para.pth')

torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 

    

