
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

DIVIDER = '-----------------------------------------'
  
trainset = torchvision.datasets.CIFAR100(
             root="./data",
             train=True,
             download=True,
             transform=transforms.Compose(
                 [
                     transforms.RandomCrop(32,4),
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation(60),
                     transforms.ToTensor(),
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
                 transforms.CenterCrop(32),
                 transforms.ToTensor(),
                 transforms.Normalize(
                     (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
                     #(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                 ),
             ]
         ),
     )
 

train_loader = torch.utils.data.DataLoader(
     trainset,
     batch_size=20,
     shuffle=True,
     num_workers=4,
     pin_memory=True,
 )
test_loader = torch.utils.data.DataLoader(
     valset,
     batch_size=20,
     shuffle=True,
     num_workers=4,
     pin_memory=True,
 )


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
    
    model = ResNet()
    
    model.to(device)
    
# =============================================================================
# load pretrained parameters
# =============================================================================
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
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
    
    neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(1))
    neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(2))
    neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(3))
    
    #neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
    
    neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(4))
    neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(5))
    neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(6))
    neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(7))
    neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(8))
    neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(9))
    
    
    neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(10))
    neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(11))
    neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(12))
    #neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
    neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(13))
    neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(14))
    neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(15))
    neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(16))
    neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(17))
    neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(18))
    neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(19))
    neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(20))
    neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(21))
    
    neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(22))
    neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(23))
    neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(24))
    neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(25))
    #neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(29))
    neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(26))
    neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(27))
    neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(28))
    neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(29))
    neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(30))
    neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(31))
    neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(32))
    neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(33))
    neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(34))
    neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(35))
    neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(36))
    neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(37))
    neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(38))
    neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(39))
    
    neuralnet.layer3[6].conv1.register_forward_hook(get_activation_0(40))
    neuralnet.layer3[6].conv2.register_forward_hook(get_activation_0(41))
    neuralnet.layer3[6].conv3.register_forward_hook(get_activation_0(42))
    neuralnet.layer3[7].conv1.register_forward_hook(get_activation_0(43))
    neuralnet.layer3[7].conv2.register_forward_hook(get_activation_0(44))
    neuralnet.layer3[7].conv3.register_forward_hook(get_activation_0(45))
    neuralnet.layer3[8].conv1.register_forward_hook(get_activation_0(46))
    neuralnet.layer3[8].conv2.register_forward_hook(get_activation_0(47))
    neuralnet.layer3[8].conv3.register_forward_hook(get_activation_0(48))
    neuralnet.layer3[9].conv1.register_forward_hook(get_activation_0(49))
    neuralnet.layer3[9].conv2.register_forward_hook(get_activation_0(50))
    neuralnet.layer3[9].conv3.register_forward_hook(get_activation_0(51))
    neuralnet.layer3[10].conv1.register_forward_hook(get_activation_0(52))
    neuralnet.layer3[10].conv2.register_forward_hook(get_activation_0(53))
    neuralnet.layer3[10].conv3.register_forward_hook(get_activation_0(54))
    neuralnet.layer3[11].conv1.register_forward_hook(get_activation_0(55))
    neuralnet.layer3[11].conv2.register_forward_hook(get_activation_0(56))
    neuralnet.layer3[11].conv3.register_forward_hook(get_activation_0(57))
    neuralnet.layer3[12].conv1.register_forward_hook(get_activation_0(58))
    neuralnet.layer3[12].conv2.register_forward_hook(get_activation_0(59))
    neuralnet.layer3[12].conv3.register_forward_hook(get_activation_0(60))
    neuralnet.layer3[13].conv1.register_forward_hook(get_activation_0(61))
    neuralnet.layer3[13].conv2.register_forward_hook(get_activation_0(62))
    neuralnet.layer3[13].conv3.register_forward_hook(get_activation_0(63))
    neuralnet.layer3[14].conv1.register_forward_hook(get_activation_0(64))
    neuralnet.layer3[14].conv2.register_forward_hook(get_activation_0(65))
    neuralnet.layer3[14].conv3.register_forward_hook(get_activation_0(66))
    neuralnet.layer3[15].conv1.register_forward_hook(get_activation_0(67))
    neuralnet.layer3[15].conv2.register_forward_hook(get_activation_0(68))
    neuralnet.layer3[15].conv3.register_forward_hook(get_activation_0(69))
    neuralnet.layer3[16].conv1.register_forward_hook(get_activation_0(70))
    neuralnet.layer3[16].conv2.register_forward_hook(get_activation_0(71))
    neuralnet.layer3[16].conv3.register_forward_hook(get_activation_0(72))
    neuralnet.layer3[17].conv1.register_forward_hook(get_activation_0(73))
    neuralnet.layer3[17].conv2.register_forward_hook(get_activation_0(74))
    neuralnet.layer3[17].conv3.register_forward_hook(get_activation_0(75))
    neuralnet.layer3[18].conv1.register_forward_hook(get_activation_0(76))
    neuralnet.layer3[18].conv2.register_forward_hook(get_activation_0(77))
    neuralnet.layer3[18].conv3.register_forward_hook(get_activation_0(78))
    neuralnet.layer3[19].conv1.register_forward_hook(get_activation_0(79))
    neuralnet.layer3[19].conv2.register_forward_hook(get_activation_0(80))
    neuralnet.layer3[19].conv3.register_forward_hook(get_activation_0(81))
    neuralnet.layer3[20].conv1.register_forward_hook(get_activation_0(82))
    neuralnet.layer3[20].conv2.register_forward_hook(get_activation_0(83))
    neuralnet.layer3[20].conv3.register_forward_hook(get_activation_0(84))
    neuralnet.layer3[21].conv1.register_forward_hook(get_activation_0(85))
    neuralnet.layer3[21].conv2.register_forward_hook(get_activation_0(86))
    neuralnet.layer3[21].conv3.register_forward_hook(get_activation_0(87))
    neuralnet.layer3[22].conv1.register_forward_hook(get_activation_0(88))
    neuralnet.layer3[22].conv2.register_forward_hook(get_activation_0(89))
    neuralnet.layer3[22].conv3.register_forward_hook(get_activation_0(90))
    
    neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(91))
    neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(92))
    neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(93))
    #neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(98))
    neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(94))
    neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(95))
    neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(96))
    neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(97))
    neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(98))
    neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(99))

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
