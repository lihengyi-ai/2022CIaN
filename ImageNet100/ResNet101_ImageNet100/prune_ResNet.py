
'''
Author: IHPC Ella
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

from common import *
from resnet import *
import pandas as pd

cfg: List[Union[int]] = [i for i in range(100)]


DIVIDER = '-----------------------------------------'

IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')


sparsity = []
for layer_in in range(0,67):
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

    #model = ResNet()
    # model.to(device)
    
    save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    model = torch.load(save_path2)
    
    model.to(device)
# =============================================================================
# load pretrained parameters
# =============================================================================
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
       
# =============================================================================
# calculate the treshold according to the prunning ratio
# =============================================================================

    threhold = torch.zeros(68)
    index = 0
    layer_index = 0
    first_identity = 0
    for batch_norm in model.named_modules():
        m = batch_norm[1]

        if batch_norm[0][-3:] == "bn1" or batch_norm[0][-3:] == "bn2":

            
            print(layer_index)
            
            size = m.weight.data.shape[0]
            bn = torch.zeros(size)
            bn = m.weight.data.abs().clone()
            
            y, i = torch.sort(bn)
            
            
            if sparsity[layer_index] > 0.5:
                sparsity[layer_index] = sparsity[layer_index] * 0.5

            threshold_index = int(size * (sparsity[layer_index])) 
            if threshold_index >= size-2:
                if threshold_index >=1: 
                    threshold_index = threshold_index - 1
                else:
                    threshold_index = 0
                
            threhold[layer_index] = y[threshold_index]
            layer_index = layer_index + 1
# =============================================================================
# "prinning" the rudundant channels
# =============================================================================
    pruned = 0
    
    cfg_mask = []
    
    layer_index = 0 #config file index
    cfg_index = 0  # the threshold
    
    for k, layer in enumerate(model.named_modules()):
        m = layer[1] 
        #if isinstance(m, nn.BatchNorm2d):
        if (layer[0][-3:] == "bn1" or layer[0][-3:] == "bn2") and first_identity == 1:

            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy
            #mask = weight_copy.gt(thre).float().cuda() 
            
            mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
            
            if int(torch.sum(mask)) >= 2:
                mask = weight_copy.gt(threhold[cfg_index]).float().cpu()
            else:
                mask = weight_copy.gt(0).float().cpu()
                
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            #cfg.append(int(torch.sum(mask)))
            
            cfg[layer_index] = int(torch.sum(mask))
            cfg_mask.append(mask.clone())
            
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
            
            cfg_index = cfg_index + 1
            
            layer_index = layer_index + 1
            
        elif layer[0][-3:] == "bn3":
            
            weight_copy = m.weight.data.abs().clone()
            weight_copy = weight_copy
            
            mask = weight_copy.gt(0).float().cpu()
            
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            
            cfg[layer_index] = int(torch.sum(mask))
            cfg_mask.append(mask.clone())
            
            layer_index = layer_index + 1
            
        if first_identity == 0:
            
            if layer[0][-3:] == "bn1":
                
                first_identity = 1
                
                layer_index = layer_index + 1
                
                cfg_index = cfg_index + 1
      
    print(cfg)

# =============================================================================
# create the newmodel according to cfg file
# =============================================================================
    savepath = os.path.join('pruned_model', 'pruned0.pth')
    
    newmodel = ResNet1(cfg = cfg)

    torch.save(newmodel, savepath,_use_new_zipfile_serialization=False) 

# =============================================================================
# Prunning the model
# =============================================================================
    a = 0
    layer_id_in_cfg = 0
        
    start_mask = torch.ones(64)
    
    end_mask = cfg_mask[layer_id_in_cfg]
    
    conv_base_flag = 0
    
    bn_base_flag = 0

    count = 0
    
    for [model0, model1] in zip(model.named_modules(), newmodel.named_modules()):
        
        identity = model0[0]
        m0 = model0[1]
        m1 = model1[1]
         
# =============================================================================
# the base layer
# =============================================================================
        # if identity == "":
        #     continue
        if bn_base_flag == 0 or conv_base_flag == 0:
            if identity == "bn1":
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                bn_base_flag = 1
                continue        
            elif identity == "conv1":
                w1 = m0.weight.data[:, :, :, :].clone()
                #b1 = m0.bias.data.clone()
                
                m1.weight.data = w1.clone()
                #m1.bias.data = b1.clone()
                conv_base_flag = 1
                continue
            
        if identity == "fc":
            
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()  
# =============================================================================
# downsample layers
# =============================================================================
        elif identity[-3:] == "e.0":
            
            w1 = m0.weight.data[:, :, :, :].clone()
            
            m1.weight.data = w1.clone()
            
            continue
                
        elif identity[-3:] == "e.1":
            
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            
            continue
# =============================================================================
#batchnormalization layer       
# =============================================================================
        elif identity[-3:] == "bn1" or identity[-3:] == "bn2":

            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()            
            
            start_mask = cfg_mask[layer_id_in_cfg]
            
            end_mask = cfg_mask[layer_id_in_cfg + 1]
            
            layer_id_in_cfg = layer_id_in_cfg + 1
            
            print(layer_id_in_cfg)

        elif identity[-3:] == "bn3": # no pruning 
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone() 

            if layer_id_in_cfg < 98:
                print("layer_id_in_cfg",layer_id_in_cfg)
            
                start_mask = cfg_mask[layer_id_in_cfg]
            
                end_mask = cfg_mask[layer_id_in_cfg + 1]
            layer_id_in_cfg = layer_id_in_cfg + 1
# =============================================================================
# convolutional layer           
# =============================================================================
                
        elif identity[-3:-1] == "nv":
            count = count + 1
            print("count",count)
            print(len(start_mask))
            print(len(end_mask))

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

# =============================================================================
#     save the model and the parameters
# =============================================================================
 
    save_path1 = os.path.join('pruned_model', 'pruned1.pth')

    torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 

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
