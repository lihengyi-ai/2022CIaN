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
import time
from gpwise_vgg13bn import *
from resnet_cifar import resnet50
from data.datasets import Dataset

import warnings
warnings.filterwarnings("ignore")

DIVIDER = '-----------------------------------------'
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

DIVIDER = '-----------------------------------------'

torch.set_num_threads(1)

train_loader, test_loader = Dataset("cifar10", batch_size=20)


def train_test(dset_dir, batchsize, learnrate, epochs, float_model,line_num):

    device = torch.device('cpu')
    
    save_path2 = os.path.join('pruned_model', 'pruned_group.pth')
    model = torch.load(save_path2)
    model.to(device)

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
        data.to_csv('./10group.csv',mode = 'a',header = False, index = False)
        

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

    print(DIVIDER)
    
    for i in range(10): 
        
        step_value = 0.1
        
        i1 = i*step_value
        i2 = i*step_value+step_value
        
        pruning_operation(args.dset_dir, args.batchsize, args.learnrate, args.epochs, args.float_model,i1,i2,i2)
        #sys.exit()
        print("i",i1,i2)
        train_test(args.dset_dir, args.batchsize, args.learnrate, args.epochs, args.float_model,i)
        
        print("..................")
        
    return

if __name__ == '__main__':
    run_main()
