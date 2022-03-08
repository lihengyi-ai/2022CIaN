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
from torchvision import models
from common import *
import pandas as pd

# from thop import profile
# from thop import clever_format


DIVIDER = '-----------------------------------------'


IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')
valdata = valdata[0:1]

  
def default_loader(path):

    return Image.open(path).convert("RGB")
   # return img

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
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')
    # model = torch.load(save_path2)
    # model.to(device)
    model = CNN().to(device)
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))

    def get_activation_0(number):
        def hook(model, input, output):
            #print('------------ Input -------------')
            input_data =input[0]
            #print(type(input_data),type(input_test),type(input))
            input_numpy = input_data.detach().numpy()
            #print(input_data.shape)
            np.save(str(number)+'.npy',input_numpy)

        return hook

    
    model.features[0].register_forward_hook(get_activation_0(0))
    model.features[3].register_forward_hook(get_activation_0(1))
    model.features[7].register_forward_hook(get_activation_0(2))
    model.features[10].register_forward_hook(get_activation_0(3))
    model.features[14].register_forward_hook(get_activation_0(4))
    model.features[17].register_forward_hook(get_activation_0(5))
    model.features[21].register_forward_hook(get_activation_0(6))
    model.features[24].register_forward_hook(get_activation_0(7))
    model.features[28].register_forward_hook(get_activation_0(8))
    model.features[31].register_forward_hook(get_activation_0(9))
    model.features[35].register_forward_hook(get_activation_0(10))
    model.features[39].register_forward_hook(get_activation_0(11))
    model.features[43].register_forward_hook(get_activation_0(12))

    train_dataset = GetLoader(traindata)
    train_loader = torch.utils.data.DataLoader(
         train_dataset,
         batch_size=batchsize,
         shuffle=True,
         num_workers=4,
         pin_memory=True,
     )
    test_data = GetLoader(valdata)
    test_loader = torch.utils.data.DataLoader(
         test_data,
         batch_size=batchsize,
         shuffle=True,
         num_workers=4,
         pin_memory=True,
     )
    accuracy = 0
    loss = 0
    # training with test after each epoch
    for epoch in range(1, epochs + 1):

        accuracy, loss = test(model, device, test_loader)
        sys.exit()
        list1 = [epoch,accuracy,loss]
        data1 = pd.DataFrame([list1])
        data1.to_csv('./val.csv',mode = 'a',header = False, index = False)

        shutil.rmtree(float_model, ignore_errors=True)    
        os.makedirs(float_model)   
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
