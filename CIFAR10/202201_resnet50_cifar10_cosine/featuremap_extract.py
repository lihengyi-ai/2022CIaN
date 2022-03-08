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
from common import *
import pandas as pd
from resnet import *

from resnet_cifar import resnet50
from data.datasets import Dataset

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

DIVIDER = '-----------------------------------------'

torch.set_num_threads(1)

train_loader, test_loader = Dataset("cifar10", batch_size=1)


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
    
    # save_path2 = os.path.join('pruned_model', 'pruned_para.pth')

    # model = torch.load(save_path2)

    model.to(device)
# =============================================================================
# load pretrained parameters
# =============================================================================
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
    #model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))
    neuralnet = model

    def get_activation_0(number):
        def hook(model, input, output):
            #print('------------ Input -------------')
            input_data =input[0]
            #print(type(input_data),type(input_test),type(input))
            input_numpy = input_data.detach().numpy()
            #print(input_data.shape)
            np.save("./input/"+str(number)+'.npy',input_numpy)

        return hook

    neuralnet.conv1.register_forward_hook(get_activation_0(0))
    
    #neuralnet.conv2_x[0].residual_function[0].register_forward_hook(get_activation_0(1))
    neuralnet.conv2_x[0].residual_function[3].register_forward_hook(get_activation_0(1))
    neuralnet.conv2_x[0].residual_function[6].register_forward_hook(get_activation_0(2))
    
    
   # neuralnet.conv2_x[1].residual_function[0].register_forward_hook(get_activation_0(4))
    neuralnet.conv2_x[1].residual_function[3].register_forward_hook(get_activation_0(3))
    neuralnet.conv2_x[1].residual_function[6].register_forward_hook(get_activation_0(4))
    
    #neuralnet.conv2_x[2].residual_function[0].register_forward_hook(get_activation_0(7))
    neuralnet.conv2_x[2].residual_function[3].register_forward_hook(get_activation_0(5))
    neuralnet.conv2_x[2].residual_function[6].register_forward_hook(get_activation_0(6))
    
    
    #neuralnet.conv3_x[0].residual_function[0].register_forward_hook(get_activation_0(10))
    neuralnet.conv3_x[0].residual_function[3].register_forward_hook(get_activation_0(7))
    neuralnet.conv3_x[0].residual_function[6].register_forward_hook(get_activation_0(8))
    
    
    #neuralnet.conv3_x[1].residual_function[0].register_forward_hook(get_activation_0(13))
    neuralnet.conv3_x[1].residual_function[3].register_forward_hook(get_activation_0(9))
    neuralnet.conv3_x[1].residual_function[6].register_forward_hook(get_activation_0(10))
    
    #neuralnet.conv3_x[2].residual_function[0].register_forward_hook(get_activation_0(16))
    neuralnet.conv3_x[2].residual_function[3].register_forward_hook(get_activation_0(11))
    neuralnet.conv3_x[2].residual_function[6].register_forward_hook(get_activation_0(12))
    
    
    #neuralnet.conv3_x[3].residual_function[0].register_forward_hook(get_activation_0(19))
    neuralnet.conv3_x[3].residual_function[3].register_forward_hook(get_activation_0(13))
    neuralnet.conv3_x[3].residual_function[6].register_forward_hook(get_activation_0(14))
    
    
    #neuralnet.conv4_x[0].residual_function[0].register_forward_hook(get_activation_0(22))
    neuralnet.conv4_x[0].residual_function[3].register_forward_hook(get_activation_0(15))
    neuralnet.conv4_x[0].residual_function[6].register_forward_hook(get_activation_0(16))
    
    
    #neuralnet.conv4_x[1].residual_function[0].register_forward_hook(get_activation_0(25))
    neuralnet.conv4_x[1].residual_function[3].register_forward_hook(get_activation_0(17))
    neuralnet.conv4_x[1].residual_function[6].register_forward_hook(get_activation_0(18))
    
    #neuralnet.conv4_x[2].residual_function[0].register_forward_hook(get_activation_0(28))
    neuralnet.conv4_x[2].residual_function[3].register_forward_hook(get_activation_0(19))
    neuralnet.conv4_x[2].residual_function[6].register_forward_hook(get_activation_0(20))
    
    #neuralnet.conv4_x[3].residual_function[0].register_forward_hook(get_activation_0(31))
    neuralnet.conv4_x[3].residual_function[3].register_forward_hook(get_activation_0(21))
    neuralnet.conv4_x[3].residual_function[6].register_forward_hook(get_activation_0(22))
    
    #neuralnet.conv4_x[4].residual_function[0].register_forward_hook(get_activation_0(34))
    neuralnet.conv4_x[4].residual_function[3].register_forward_hook(get_activation_0(23))
    neuralnet.conv4_x[4].residual_function[6].register_forward_hook(get_activation_0(24))
    
    #neuralnet.conv4_x[5].residual_function[0].register_forward_hook(get_activation_0(37))
    neuralnet.conv4_x[5].residual_function[3].register_forward_hook(get_activation_0(25))
    neuralnet.conv4_x[5].residual_function[6].register_forward_hook(get_activation_0(26))
    
    
    
    #neuralnet.conv5_x[0].residual_function[0].register_forward_hook(get_activation_0(40))
    neuralnet.conv5_x[0].residual_function[3].register_forward_hook(get_activation_0(27))
    neuralnet.conv5_x[0].residual_function[6].register_forward_hook(get_activation_0(28))
    
    
    #neuralnet.conv5_x[1].residual_function[0].register_forward_hook(get_activation_0(43))
    neuralnet.conv5_x[1].residual_function[3].register_forward_hook(get_activation_0(29))
    neuralnet.conv5_x[1].residual_function[6].register_forward_hook(get_activation_0(30))
    
    #neuralnet.conv5_x[2].residual_function[0].register_forward_hook(get_activation_0(46))
    neuralnet.conv5_x[2].residual_function[3].register_forward_hook(get_activation_0(31))
    neuralnet.conv5_x[2].residual_function[6].register_forward_hook(get_activation_0(32))

    accuracy = 0
    loss = 0
    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        
        accuracy, loss = test(model, device, test_loader)
        break
        list1 = [epoch,accuracy,loss]
        data1 = pd.DataFrame([list1])
        data1.to_csv('./val.csv',mode = 'a',header = False, index = False)

        shutil.rmtree(float_model, ignore_errors=True)    
        os.makedirs(float_model)   

        print('Trained model written to',save_path)

    shutil.rmtree(float_model, ignore_errors=True)    

    print('Trained model written to',save_path)

    return

def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir',    type=str,  default='dataset',     help='Path to test & train datasets. Default is dataset')
    ap.add_argument('-b', '--batchsize',   type=int,  default=1,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=1,             help='Number of training epochs. Must be an integer. Default is 3')
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
