'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Simple PyTorch MNIST example - training & testing
'''

'''
Author: Mark Harvey, Xilinx inc
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

# from thop import profile
# from thop import clever_format
# import csv
# import seaborn as sns

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np


# from matplotlib.pyplot import MultipleLocator
# from matplotlib.ticker import FuncFormatter



DIVIDER = '-----------------------------------------'

sparsity = []
for layer_in in range(0,13):
      #print(layer_in)
      layers_data = np.load("./input/"+str(layer_in)+'.npy')
      element = len(layers_data[0])*len(layers_data[0][0])*len(layers_data[0][0][0]) 
      NZero_features = 0
      #NZero_features_member = 0
      #N_flag = 0
      #N_zero = 0
      #nzero_array =np.zeros(element,dtype = float,order = 'C')
      #data_index = 0
      for i in range(len(layers_data[0])):
          for j in range(len(layers_data[0][i])):
              for k in range(len(layers_data[0][i][j])): 
                  if(layers_data[0][i][j][k] != 0):
                      #nzero_array[data_index] = layers_data[0][i][j][k]
                      #data_index = data_index + 1
                      NZero_features += 1
      #print('NZero_features',NZero_features)     

      percentage = 1-(NZero_features/element)
      if percentage >= 0.5:
          percentage = percentage * 0.5
      
      
      sparsity.append(percentage)
      #print(percentage)
print(sparsity)
input()
#traindata = traindata[0:2000]
#print(traindata[0:2000])
 

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
    
# =============================================================================
# load the pretrained/pruned model with parameters
# =============================================================================
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')
    # model = torch.load(save_path2)
    #model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
    #torch.save(model, save_path2) 
    
# =============================================================================
# numbers of parameters
# =============================================================================

    # num_parameters = sum([param.nelement() for param in model.parameters()])
    # print("numbers",num_parameters)
    #sys.exit()
    
# =============================================================================
# statistics of flops and parameters with thop
# =============================================================================

    # input = torch.randn(1,3,224,224)

    # flops, params = profile(model,inputs = (input,))
    # flops, params = clever_format([flops,params],"%.3f")
    
    # print(flops)
    # print(params)
    # sys.exit()

    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')  
    
    # newmodel = torch.load(save_path2)
    
    # num_parameters1 = sum([param.nelement() for param in newmodel.parameters()])
    # print(num_parameters1)
    
    # sys.exit()
    
# =============================================================================
# statistics of BN layers
# =============================================================================

    # total = 0
    # layer_index = 0
    # for m in model.modules():
    #   if isinstance(m, nn.BatchNorm2d):
    #       layer_index = layer_index + 1
    #       if layer_index <= 10:
    #           total += m.weight.data.shape[0]
    # bn = torch.zeros(total)
    
# =============================================================================
# calculate the treshold according to the prunning ratio
# =============================================================================
    # index = 0
    # layer_index = 0
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         layer_index = layer_index + 1
    #         if layer_index <= 10:
    #             size = m.weight.data.shape[0]
    #             bn[index:(index+size)] = m.weight.data.abs().clone()
    #             index += size
    # y, i = torch.sort(bn)
    # thre_index = int(total * 0.80)
    # thre = y[thre_index]
    # print(thre)
    
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
    print(threhold)
# =============================================================================
#     initialize the scale factor
# =============================================================================
    # for m in model.modules():
    #   if isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 0.5)
    #     print(m.weight)
    #     print(m.weight.data)
    #     print("input")
    #     input()
    #     nn.init.constant_(m.bias, 0)

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
                #print(weight_copy)
                
                # sns.distplot(weight_copy,norm_hist = False,hist = False,kde = True,rug = False
                #               )
                # plt.xlim(0.35,0.45)
                # plt.ylabel("Density",fontsize = 12)
                # plt.xlabel("Value of inputs",fontsize = 12)
                
                # ax = plt.gca()
                # #x_major_locator=MultipleLocator(0.5)
                # #ax.xaxis.set_major_locator(x_major_locator)
                # ax.spines['right'].set_color('white')
                # ax.spines['top'].set_color('white')
                # ax.spines['bottom'].set_color('lightgray')
                # ax.spines['left'].set_color('lightgray')
                # plt.gcf().set_facecolor('white')
                # plt.grid()
                # plt.grid(linestyle='--')
                
                # plt.subplots_adjust(top=0.95,bottom=0.12,right=0.95,left=0.06,hspace=1,wspace=0.5)
                
                # plt.legend(ncol = 2,fontsize = "small", loc= "best")
                # plt.savefig('vgg_in.eps',format = "eps",dpi = 1000)
    
                #mask = weight_copy.gt(thre).float().cuda()
               
                    
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

    #pruned_ratio = pruned/total
        
    print(cfg)
    print("input")
    input()
    #sys.exit()
   # input()
    #print(cfg_mask)
    # # print(model)
    # # print(newmodel)
    #sys.exit()
# =============================================================================
# create the newmodel according to cfg file
# =============================================================================
    savepath = os.path.join('pruned_model', 'pruned0.pth')
    save_path1 = os.path.join('pruned_model', 'pruned1.pth')
    
    newmodel =  vgg('vgg13_bn', cfg, True, False, True)
    
    torch.save(newmodel, savepath,_use_new_zipfile_serialization=False) 
    # # print(newmodel)
    # # print(model)
    # torch.save(newmodel.state_dict(), savepath,_use_new_zipfile_serialization=False) 
    
    # torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 
    # sys.exit()
    
    input()
    #print(model)
    # num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    


    #torch.save(newmodel, savepath) 
    # with open(savepath, "w") as fp:
    #     fp.write("Configuration: \n"+str(cfg)+"\n")
    #     fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    #     fp.write("Test accuracy: \n"+str(123))
    

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
                # if a == 10:
                #     start_mask = end_mask.clone()
                    #end_mask = torch.ones(1120)      
                #print(m1.weight.data)
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                #idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                end_mask = torch.ones(100)
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
    #print(model)

# =============================================================================
#     save the model and the parameters
# =============================================================================

    
    
    
    
    torch.save(newmodel, save_path1,_use_new_zipfile_serialization=False) 
    sys.exit()


   

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
