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
from common import *
import pandas as pd
from resnet import *

#from resnet_cifar import resnet50
#from data.datasets import Dataset

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST



DIVIDER = '-----------------------------------------'

torch.set_num_threads(1)



IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')

def default_loader(path):
    #print(Image.open(path))
    #img = Image.open(path).convert("RGB")
    #img = cv2.imread(path,cv2.IMREAD_COLOR)
    #img = cv2.imread(path)
    #img = cv2.resize(img,(224,224))
   # img = img[:,:,::-1].transpose((2,0,1))
    #img.crop((224,224))
    #sys.exit()
    #img = img/255.0
    #img = torch.FloatTensor(img)
    #return Image.open(path).convert("RGB")
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
train_dataset = GetLoader(traindata)
train_loader = torch.utils.data.DataLoader(
     train_dataset,
     batch_size=20,
     shuffle=True,
     num_workers=4,
     pin_memory=True,
 )
test_data = GetLoader(valdata)
test_loader = torch.utils.data.DataLoader(
     test_data,
     batch_size=20,
     shuffle=True,
     num_workers=4,
     pin_memory=True,
 )
    
    
#train_loader, test_loader = Dataset("cifar10", batch_size=20)


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
    
    #save_path2 = os.path.join('pruned_model', 'pruned_para.pth')

    #model = torch.load(save_path2)
    #print(model)
    #model = ResNet()
    save_path1 = os.path.join('pruned_model', 'pruned_para.pth')
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    model = torch.load(save_path1)
    
    #model.to(device)
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


    neuralnet.conv1.register_forward_hook(get_activation_0(0))
    
    #neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(2))
    neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(1))
    neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(2))
    #neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
    #neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(6))
    neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(3))
    neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(4))
    #neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(9))
    neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(5))
    neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(6))
    
    
    #neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(12))
    neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(7))
    neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(8))
    #neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
    #neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(16))
    neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(9))
    neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(10))
    #neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(19))
    neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(11))
    neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(12))
    #neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(22))
    neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(13))
    neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(14))
    
   # neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(25))
    neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(15))
    neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(16))
    #neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(28))
    #neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(29))
    neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(17))
    neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(18))
    #neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(32))
    neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(19))
    neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(20))
    #neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(35))
    neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(21))
    neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(22))
    #neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(38))
    neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(23))
    neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(24))
    #neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(41))
    neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(25))
    neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(26))
    
    
    #neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(44))
    neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(27))
    neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(28))
    #neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(47))
    #neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(48))
    neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(29))
    neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(30))
   #neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(51))
    neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(31))
    neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(32))
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
