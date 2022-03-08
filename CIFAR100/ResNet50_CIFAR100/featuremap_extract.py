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
import cv2
from common import *
import pandas as pd
from resnet import *

from resnet_cifar import resnet50
from data.datasets import Dataset

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST



DIVIDER = '-----------------------------------------'

torch.set_num_threads(1)

train_loader, test_loader = Dataset("cifar10", batch_size=20)


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
    
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    # model = torch.load(save_path2)
    
    #model.to(device)
    
    #model = resnet50()
    # print(model)
    # sys.exit()
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')
    # model = torch.load(save_path2)
    #model.to(device)
    
    save_path2 = os.path.join('pruned_model', 'pruned_para.pth')

    model = torch.load(save_path2)
    #print(model)
    
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
            # input_txt = str(input_numpy)
            # with open("Layer"+layer+"_input.txt","w") as f:
            #   f.write(input_txt)
            #print('----------- Output --------------')
            #output_numpy = output.detach().numpy()
            #print('type_out',type(output))
            #print(output.shape)
            # Count_zero_feature = 0
            # for i in range(len(output[0])):
            #     Count_Nzero = 0
            #     for j in range(len(output[0][i])):
            #         for k in range(len(output[0][i][j])):
            #             if (output[0][i][j][k] <= -2):
            #                output[0][i][j][k] = 0.5
            '''
                if (Count_Nzero <= 5):
                    for m in range(len(output[0][i])):
                        for n in range(len(output[0][i][m])):
                            output[0][i][m][n] = 0
            '''
            #np.save("Layer"+layer+"_output.npy",output_numpy)
            #output_txt = str(output_numpy)
            #print('output_type',type(output))
            #print(type(output_numpy))
            #print(output.shape)
            # with open("Layer"+layer+"_output.txt","w") as f:
            #   f.write(output_txt) 
        return hook


    
    # neuralnet.conv1.register_forward_hook(get_activation_0(0))
    
    # neuralnet.conv2_x[0].residual_function[0].register_forward_hook(get_activation_0(1))
    # neuralnet.conv2_x[0].residual_function[3].register_forward_hook(get_activation_0(2))
    # neuralnet.conv2_x[0].residual_function[6].register_forward_hook(get_activation_0(3))
    
    # neuralnet.conv2_x[0].shortcut[0].register_forward_hook(get_activation_0(4))
    
    # neuralnet.conv2_x[1].residual_function[0].register_forward_hook(get_activation_0(5))
    # neuralnet.conv2_x[1].residual_function[3].register_forward_hook(get_activation_0(6))
    # neuralnet.conv2_x[1].residual_function[6].register_forward_hook(get_activation_0(7))
    
    # neuralnet.conv2_x[2].residual_function[0].register_forward_hook(get_activation_0(8))
    # neuralnet.conv2_x[2].residual_function[3].register_forward_hook(get_activation_0(9))
    # neuralnet.conv2_x[2].residual_function[6].register_forward_hook(get_activation_0(10))
    
    
    # neuralnet.conv3_x[0].residual_function[0].register_forward_hook(get_activation_0(11))
    # neuralnet.conv3_x[0].residual_function[3].register_forward_hook(get_activation_0(12))
    # neuralnet.conv3_x[0].residual_function[6].register_forward_hook(get_activation_0(13))
    
    # neuralnet.conv2_x[0].shortcut[0].register_forward_hook(get_activation_0(14))
    
    # neuralnet.conv3_x[1].residual_function[0].register_forward_hook(get_activation_0(15))
    # neuralnet.conv3_x[1].residual_function[3].register_forward_hook(get_activation_0(16))
    # neuralnet.conv3_x[1].residual_function[6].register_forward_hook(get_activation_0(17))
    
    # neuralnet.conv3_x[2].residual_function[0].register_forward_hook(get_activation_0(18))
    # neuralnet.conv3_x[2].residual_function[3].register_forward_hook(get_activation_0(19))
    # neuralnet.conv3_x[2].residual_function[6].register_forward_hook(get_activation_0(20))
    
    
    # neuralnet.conv3_x[3].residual_function[0].register_forward_hook(get_activation_0(21))
    # neuralnet.conv3_x[3].residual_function[3].register_forward_hook(get_activation_0(22))
    # neuralnet.conv3_x[3].residual_function[6].register_forward_hook(get_activation_0(23))
    
    
    # neuralnet.conv4_x[0].residual_function[0].register_forward_hook(get_activation_0(24))
    # neuralnet.conv4_x[0].residual_function[3].register_forward_hook(get_activation_0(25))
    # neuralnet.conv4_x[0].residual_function[6].register_forward_hook(get_activation_0(26))
    
    # neuralnet.conv4_x[0].shortcut[0].register_forward_hook(get_activation_0(27))
    
    # neuralnet.conv4_x[1].residual_function[0].register_forward_hook(get_activation_0(28))
    # neuralnet.conv4_x[1].residual_function[3].register_forward_hook(get_activation_0(29))
    # neuralnet.conv4_x[1].residual_function[6].register_forward_hook(get_activation_0(30))
    
    # neuralnet.conv4_x[2].residual_function[0].register_forward_hook(get_activation_0(31))
    # neuralnet.conv4_x[2].residual_function[3].register_forward_hook(get_activation_0(32))
    # neuralnet.conv4_x[2].residual_function[6].register_forward_hook(get_activation_0(33))
    
    # neuralnet.conv4_x[3].residual_function[0].register_forward_hook(get_activation_0(34))
    # neuralnet.conv4_x[3].residual_function[3].register_forward_hook(get_activation_0(35))
    # neuralnet.conv4_x[3].residual_function[6].register_forward_hook(get_activation_0(36))
    
    # neuralnet.conv4_x[4].residual_function[0].register_forward_hook(get_activation_0(37))
    # neuralnet.conv4_x[4].residual_function[3].register_forward_hook(get_activation_0(38))
    # neuralnet.conv4_x[4].residual_function[6].register_forward_hook(get_activation_0(39))
    
    # neuralnet.conv4_x[5].residual_function[0].register_forward_hook(get_activation_0(40))
    # neuralnet.conv4_x[5].residual_function[3].register_forward_hook(get_activation_0(41))
    # neuralnet.conv4_x[5].residual_function[6].register_forward_hook(get_activation_0(42))
    
    
    
    # neuralnet.conv5_x[0].residual_function[0].register_forward_hook(get_activation_0(43))
    # neuralnet.conv5_x[0].residual_function[3].register_forward_hook(get_activation_0(44))
    # neuralnet.conv5_x[0].residual_function[6].register_forward_hook(get_activation_0(45))
    
    # neuralnet.conv5_x[0].shortcut[0].register_forward_hook(get_activation_0(46))
    
    # neuralnet.conv5_x[1].residual_function[0].register_forward_hook(get_activation_0(47))
    # neuralnet.conv5_x[1].residual_function[3].register_forward_hook(get_activation_0(48))
    # neuralnet.conv5_x[1].residual_function[6].register_forward_hook(get_activation_0(49))
    
    # neuralnet.conv5_x[2].residual_function[0].register_forward_hook(get_activation_0(50))
    # neuralnet.conv5_x[2].residual_function[3].register_forward_hook(get_activation_0(51))
    # neuralnet.conv5_x[2].residual_function[6].register_forward_hook(get_activation_0(52))
    
    # neuralnet.conv1.register_forward_hook(get_activation_0(0))
    
    # neuralnet.conv2_x[0].residual_function[0].register_forward_hook(get_activation_0(1))
    # neuralnet.conv2_x[0].residual_function[3].register_forward_hook(get_activation_0(2))
    # neuralnet.conv2_x[0].residual_function[6].register_forward_hook(get_activation_0(3))
    
    
    # neuralnet.conv2_x[1].residual_function[0].register_forward_hook(get_activation_0(4))
    # neuralnet.conv2_x[1].residual_function[3].register_forward_hook(get_activation_0(5))
    # neuralnet.conv2_x[1].residual_function[6].register_forward_hook(get_activation_0(6))
    
    # neuralnet.conv2_x[2].residual_function[0].register_forward_hook(get_activation_0(7))
    # neuralnet.conv2_x[2].residual_function[3].register_forward_hook(get_activation_0(8))
    # neuralnet.conv2_x[2].residual_function[6].register_forward_hook(get_activation_0(9))
    
    
    # neuralnet.conv3_x[0].residual_function[0].register_forward_hook(get_activation_0(10))
    # neuralnet.conv3_x[0].residual_function[3].register_forward_hook(get_activation_0(11))
    # neuralnet.conv3_x[0].residual_function[6].register_forward_hook(get_activation_0(12))
    
    
    # neuralnet.conv3_x[1].residual_function[0].register_forward_hook(get_activation_0(13))
    # neuralnet.conv3_x[1].residual_function[3].register_forward_hook(get_activation_0(14))
    # neuralnet.conv3_x[1].residual_function[6].register_forward_hook(get_activation_0(15))
    
    # neuralnet.conv3_x[2].residual_function[0].register_forward_hook(get_activation_0(16))
    # neuralnet.conv3_x[2].residual_function[3].register_forward_hook(get_activation_0(17))
    # neuralnet.conv3_x[2].residual_function[6].register_forward_hook(get_activation_0(18))
    
    
    # neuralnet.conv3_x[3].residual_function[0].register_forward_hook(get_activation_0(19))
    # neuralnet.conv3_x[3].residual_function[3].register_forward_hook(get_activation_0(20))
    # neuralnet.conv3_x[3].residual_function[6].register_forward_hook(get_activation_0(21))
    
    
    # neuralnet.conv4_x[0].residual_function[0].register_forward_hook(get_activation_0(22))
    # neuralnet.conv4_x[0].residual_function[3].register_forward_hook(get_activation_0(23))
    # neuralnet.conv4_x[0].residual_function[6].register_forward_hook(get_activation_0(24))
    
    
    # neuralnet.conv4_x[1].residual_function[0].register_forward_hook(get_activation_0(25))
    # neuralnet.conv4_x[1].residual_function[3].register_forward_hook(get_activation_0(26))
    # neuralnet.conv4_x[1].residual_function[6].register_forward_hook(get_activation_0(27))
    
    # neuralnet.conv4_x[2].residual_function[0].register_forward_hook(get_activation_0(28))
    # neuralnet.conv4_x[2].residual_function[3].register_forward_hook(get_activation_0(29))
    # neuralnet.conv4_x[2].residual_function[6].register_forward_hook(get_activation_0(30))
    
    # neuralnet.conv4_x[3].residual_function[0].register_forward_hook(get_activation_0(31))
    # neuralnet.conv4_x[3].residual_function[3].register_forward_hook(get_activation_0(32))
    # neuralnet.conv4_x[3].residual_function[6].register_forward_hook(get_activation_0(33))
    
    # neuralnet.conv4_x[4].residual_function[0].register_forward_hook(get_activation_0(34))
    # neuralnet.conv4_x[4].residual_function[3].register_forward_hook(get_activation_0(35))
    # neuralnet.conv4_x[4].residual_function[6].register_forward_hook(get_activation_0(36))
    
    # neuralnet.conv4_x[5].residual_function[0].register_forward_hook(get_activation_0(37))
    # neuralnet.conv4_x[5].residual_function[3].register_forward_hook(get_activation_0(38))
    # neuralnet.conv4_x[5].residual_function[6].register_forward_hook(get_activation_0(39))
    
    
    
    # neuralnet.conv5_x[0].residual_function[0].register_forward_hook(get_activation_0(40))
    # neuralnet.conv5_x[0].residual_function[3].register_forward_hook(get_activation_0(41))
    # neuralnet.conv5_x[0].residual_function[6].register_forward_hook(get_activation_0(42))
    
    
    # neuralnet.conv5_x[1].residual_function[0].register_forward_hook(get_activation_0(43))
    # neuralnet.conv5_x[1].residual_function[3].register_forward_hook(get_activation_0(44))
    # neuralnet.conv5_x[1].residual_function[6].register_forward_hook(get_activation_0(45))
    
    # neuralnet.conv5_x[2].residual_function[0].register_forward_hook(get_activation_0(46))
    # neuralnet.conv5_x[2].residual_function[3].register_forward_hook(get_activation_0(47))
    # neuralnet.conv5_x[2].residual_function[6].register_forward_hook(get_activation_0(48))
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

    #optimizer = optim.Adam(model.parameters(), lr=learnrate)
    # optimizer = optim.SGD(model.parameters(), lr=learnrate,momentum=0.9,
    #     weight_decay=0.0001,
    #     nesterov=True,)

    #image datasets
    # train_dataset = torchvision.datasets.MNIST(dset_dir, 
    #                                            train=True, 
    #                                            download=True,
    #                                            transform=train_transform)
    # test_dataset = torchvision.datasets.MNIST(dset_dir,
    #                                           train=False, 
    #                                           download=True,
    #                                           transform=test_transform)

    #data loaders
    # train_dataset = GetLoader(traindata)
    # train_loader = torch.utils.data.DataLoader(
    #      train_dataset,
    #      batch_size=batchsize,
    #      shuffle=True,
    #      num_workers=4,
    #      pin_memory=True,
    #  )
    # test_data = GetLoader(valdata)
    # test_loader = torch.utils.data.DataLoader(
    #      test_data,
    #      batch_size=batchsize,
    #      shuffle=True,
    #      num_workers=4,
    #      pin_memory=True,
    #  )
    accuracy = 0
    loss = 0
    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        
        # accuracy,loss = train(model, device, train_loader, optimizer, epoch)
        # list0= [epoch,accuracy,loss]
        # data = pd.DataFrame([list0])
        # data.to_csv('./train.csv',mode = 'a',header = False, index = False)

        accuracy, loss = test(model, device, test_loader)
        sys.exit()
        list1 = [epoch,accuracy,loss]
        data1 = pd.DataFrame([list1])
        data1.to_csv('./val.csv',mode = 'a',header = False, index = False)

        shutil.rmtree(float_model, ignore_errors=True)    
        os.makedirs(float_model)   
        # save_path = os.path.join(float_model, 'f_model.pth')
        # torch.save(model.state_dict(), save_path) 
        print('Trained model written to',save_path)


    # save the trained model
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
