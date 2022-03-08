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
from torch.utils.data import DataLoader
import os
import shutil
import cv2
from torchvision import models
from common import *
import pandas as pd
import time
#from resnet import *
from resnet_cifar import resnet50
from data.datasets import Dataset

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from group_prune_ResNet import group_pruning
batchsize = 20
DIVIDER = '-----------------------------------------'

torch.set_num_threads(1)

trainset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,

        transform=transforms.Compose(
            [
                #transforms.Resize(256),
                transforms.RandomCrop(32,4),
                #transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(60),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                # ),
                transforms.Normalize(
                    (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
                ),
            ]
        ),
    )

valset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                #transforms.Resize(224),
                transforms.CenterCrop(32),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                # ),
                transforms.Normalize(
                    (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
                ),
            ]
        ),
    )
#train_dataset = pickle.load(open("cifar100_train","rb"))
train_loader = torch.utils.data.DataLoader(
     trainset,
     batch_size=batchsize,
     shuffle=True,
     num_workers=4,
     pin_memory=True,
 )
#test_data = pickle.load(open("cifar100_val","rb"))
test_loader = torch.utils.data.DataLoader(
     valset,
     batch_size=batchsize,
     shuffle=True,
     num_workers=4,
     pin_memory=True,
 )

# additional subgradient descent on the sparsity-induced penalty term

def train_test(dset_dir, batchsize, learnrate, epochs, float_model,line_num):


    device = torch.device('cuda:0')
    
    save_path2 = os.path.join('pruned_model', 'pruned_group.pth')

    model = torch.load(save_path2)
    
    model.to(device)
    
    #model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location=torch.device('cpu')))
 
    accuracy = 0
    loss = 0
   
    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        
        time1 = time.time()
        #accuracy,loss = train(model, device, train_loader, optimizer, epoch)
        accuracy, loss = test(model, device, test_loader)
        time2 = time.time()
        print(time2-time1)
        
        
        list0= [line_num*0.01,accuracy]
        data = pd.DataFrame([list0])
        data.to_csv('./group.csv',mode = 'a',header = False, index = False)

    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir',    type=str,  default='dataset',     help='Path to test & train datasets. Default is dataset')
    ap.add_argument('-b', '--batchsize',   type=int,  default=20,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=1,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('-m', '--float_model', type=str,  default='float_model', help='Path to folder where trained model is saved. Default is float_model')
    ap.add_argument("-num_workers",default = 4,type = int)
    args = ap.parse_args()

    # print('\n'+DIVIDER)
    # print('PyTorch version : ',torch.__version__)
    # print(sys.version)
    # print(DIVIDER)
    # print(' Command line options:')
    # print ('--dset_dir     : ',args.dset_dir)
    # print ('--batchsize    : ',args.batchsize)
    # print ('--learnrate    : ',args.learnrate)
    # print ('--epochs       : ',args.epochs)
    # print ('--float_model  : ',args.float_model)
    print(DIVIDER)
    
    for m_index in range(10): 
        
        step_value = 0.1
        
        i1 = m_index*step_value
        i2 = m_index*step_value+step_value
        test_flag = 0
        
        group_pruning(args.dset_dir, args.batchsize, args.learnrate, args.epochs, args.float_model,i1,i2,i2,test_flag)
        print("i",i1,i2)
        train_test(args.dset_dir, args.batchsize, args.learnrate, args.epochs, args.float_model,m_index)
        
        print("..................")
        
    return


if __name__ == '__main__':
    run_main()
