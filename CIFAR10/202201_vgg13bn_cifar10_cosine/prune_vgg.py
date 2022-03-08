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
#import cv2
from torchvision import models
from common import *
import pandas as pd

DIVIDER = '-----------------------------------------'

sparsity = []
for layer_in in range(0,13):
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
    #model = CNN().to(device)
    
    save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    model = torch.load(save_path2)
    
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')
    # model = torch.load(save_path2)
    # model.to(device)
# =============================================================================
# load pretrained parameters
# =============================================================================
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
    
    
    threhold = torch.zeros(13)
    index = 0
    layer_index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if layer_index <= 11:
                size = m.weight.data.shape[0]
                bn = torch.zeros(size)
                bn = m.weight.data.abs().clone()
                #index += size
                y, i = torch.sort(bn)
                # if layer_index == 11:
                #     threshold_index = int(size * (0))
                # else:
                    
                #     threshold_index = int(size * (sparsity[layer_index+1]))
                
                threshold_index = int(size * (sparsity[layer_index+1])) 
                
                if threshold_index >= size*0.5:
                    
                    threshold_index = int(size * (sparsity[layer_index+1])*0.5) 

                if sparsity[layer_index+1] >= 0.1:
                    
                    threhold[layer_index] = y[threshold_index]
                else:
                    threhold[layer_index] = 0
            layer_index = layer_index + 1
# =============================================================================
# "prinning" the rudundant channels
# =============================================================================
    pruned = 0
    cfg = []
    cfg_mask = []
    
    layer_index = 0
    for k, m in enumerate(model.modules()):
        #print("pruned",pruned)
        if isinstance(m, nn.BatchNorm2d):
            layer_index = layer_index + 1
            if layer_index <= 12:
                weight_copy = m.weight.data.abs().clone()
                weight_copy = weight_copy
                    
                mask = weight_copy.gt(threhold[layer_index-1]).float().cpu()
                if int(torch.sum(mask)) >= 2:
                    mask = weight_copy.gt(threhold[layer_index-1]).float().cpu()
                else:
                    mask = weight_copy.gt(0).float().cpu()
                    
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, mask.shape[0], int(torch.sum(mask))))
            else:
                weight_copy = m.weight.data.abs().clone()
                # print(weight_copy)
                # print("input")
                # input()
                #mask = weight_copy.gt(thre).float().cuda()
                mask = weight_copy.gt(0).float().cpu()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                    format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d): 
            cfg.append('M')      
  
    print(cfg)
# =============================================================================
# create the newmodel according to cfg file
# =============================================================================
    savepath = os.path.join('pruned_model', 'pruned0.pth')
    save_path1 = os.path.join('pruned_model', 'pruned1.pth')
    
    newmodel =  vgg('vgg13_bn', cfg, True, False, True)
    
    torch.save(newmodel, savepath,_use_new_zipfile_serialization=False) 

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
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
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
                end_mask = torch.ones(10)
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print("index")
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
               # input()
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))

                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        
            a = a + 1

    model = newmodel

# =============================================================================
#     save the model and the parameters
# =============================================================================

    torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 
    return

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
