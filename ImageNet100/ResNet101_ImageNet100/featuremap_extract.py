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
# from thop import profile
# from thop import clever_format


DIVIDER = '-----------------------------------------'

#IMAGE_PATH = './kuzushiji/'
    
# valdata = np.load('./20210629kuzushiji_val.npy')
# traindata = np.load('./20210629kuzushiji_train.npy')
# valdata = valdata[0:1]
# torch.set_num_threads(4)

IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')
valdata = valdata[0:5]
# for layer_in in range(0,13):
#       #print(layer_in)
#       layers_data = np.load(str(layer_in)+'.npy')
#       element = len(layers_data[0])*len(layers_data[0][0])*len(layers_data[0][0][0]) 
#       NZero_features = 0
#       #NZero_features_member = 0
#       #N_flag = 0
#       #N_zero = 0
#       #nzero_array =np.zeros(element,dtype = float,order = 'C')
#       #data_index = 0
#       for i in range(len(layers_data[0])):
#           for j in range(len(layers_data[0][i])):
#               for k in range(len(layers_data[0][i][j])): 
#                   if(layers_data[0][i][j][k] != 0):
#                       #nzero_array[data_index] = layers_data[0][i][j][k]
#                       #data_index = data_index + 1
#                       NZero_features += 1
#       #print('NZero_features',NZero_features)     
#       print(layers_data[0][1])
#       input()
#       percentage = 1-(NZero_features/element)
#       print(percentage)
# sys.exit()
  
  
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
            transforms.CenterCrop(224),
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
    #model = CNN().to(device)
    
    save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    model = torch.load(save_path2)
    
    #model.to(device)
    
    #model = ResNet()
    # print(model)
    # sys.exit()
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')
    # model = torch.load(save_path2)
    model.to(device)
# =============================================================================
# load pretrained parameters
# =============================================================================
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth'),map_location="cpu"))
    neuralnet = model
    # for k,v in model.named_parameters():
    #   print(v)
    #   print(k)
    #   input()

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
    
    # neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(1))
    # neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(2))
    # neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(3))
    
    # #neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
    
    # neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(4))
    # neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(5))
    # neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(6))
    # neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(7))
    # neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(8))
    # neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(9))
    
    
    # neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(10))
    # neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(11))
    # neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(12))
    # #neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
    # neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(13))
    # neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(14))
    # neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(15))
    # neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(16))
    # neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(17))
    # neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(18))
    # neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(19))
    # neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(20))
    # neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(21))
    
    # neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(22))
    # neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(23))
    # neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(24))
    # neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(25))
    # #neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(29))
    # neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(26))
    # neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(27))
    # neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(28))
    # neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(29))
    # neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(30))
    # neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(31))
    # neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(32))
    # neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(33))
    # neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(34))
    # neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(35))
    # neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(36))
    # neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(37))
    # neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(38))
    # neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(39))
    
    # neuralnet.layer3[6].conv1.register_forward_hook(get_activation_0(40))
    # neuralnet.layer3[6].conv2.register_forward_hook(get_activation_0(41))
    # neuralnet.layer3[6].conv3.register_forward_hook(get_activation_0(42))
    # neuralnet.layer3[7].conv1.register_forward_hook(get_activation_0(43))
    # neuralnet.layer3[7].conv2.register_forward_hook(get_activation_0(44))
    # neuralnet.layer3[7].conv3.register_forward_hook(get_activation_0(45))
    # neuralnet.layer3[8].conv1.register_forward_hook(get_activation_0(46))
    # neuralnet.layer3[8].conv2.register_forward_hook(get_activation_0(47))
    # neuralnet.layer3[8].conv3.register_forward_hook(get_activation_0(48))
    # neuralnet.layer3[9].conv1.register_forward_hook(get_activation_0(49))
    # neuralnet.layer3[9].conv2.register_forward_hook(get_activation_0(50))
    # neuralnet.layer3[9].conv3.register_forward_hook(get_activation_0(51))
    # neuralnet.layer3[10].conv1.register_forward_hook(get_activation_0(52))
    # neuralnet.layer3[10].conv2.register_forward_hook(get_activation_0(53))
    # neuralnet.layer3[10].conv3.register_forward_hook(get_activation_0(54))
    # neuralnet.layer3[11].conv1.register_forward_hook(get_activation_0(55))
    # neuralnet.layer3[11].conv2.register_forward_hook(get_activation_0(56))
    # neuralnet.layer3[11].conv3.register_forward_hook(get_activation_0(57))
    # neuralnet.layer3[12].conv1.register_forward_hook(get_activation_0(58))
    # neuralnet.layer3[12].conv2.register_forward_hook(get_activation_0(59))
    # neuralnet.layer3[12].conv3.register_forward_hook(get_activation_0(60))
    # neuralnet.layer3[13].conv1.register_forward_hook(get_activation_0(61))
    # neuralnet.layer3[13].conv2.register_forward_hook(get_activation_0(62))
    # neuralnet.layer3[13].conv3.register_forward_hook(get_activation_0(63))
    # neuralnet.layer3[14].conv1.register_forward_hook(get_activation_0(64))
    # neuralnet.layer3[14].conv2.register_forward_hook(get_activation_0(65))
    # neuralnet.layer3[14].conv3.register_forward_hook(get_activation_0(66))
    # neuralnet.layer3[15].conv1.register_forward_hook(get_activation_0(67))
    # neuralnet.layer3[15].conv2.register_forward_hook(get_activation_0(68))
    # neuralnet.layer3[15].conv3.register_forward_hook(get_activation_0(69))
    # neuralnet.layer3[16].conv1.register_forward_hook(get_activation_0(70))
    # neuralnet.layer3[16].conv2.register_forward_hook(get_activation_0(71))
    # neuralnet.layer3[16].conv3.register_forward_hook(get_activation_0(72))
    # neuralnet.layer3[17].conv1.register_forward_hook(get_activation_0(73))
    # neuralnet.layer3[17].conv2.register_forward_hook(get_activation_0(74))
    # neuralnet.layer3[17].conv3.register_forward_hook(get_activation_0(75))
    # neuralnet.layer3[18].conv1.register_forward_hook(get_activation_0(76))
    # neuralnet.layer3[18].conv2.register_forward_hook(get_activation_0(77))
    # neuralnet.layer3[18].conv3.register_forward_hook(get_activation_0(78))
    # neuralnet.layer3[19].conv1.register_forward_hook(get_activation_0(79))
    # neuralnet.layer3[19].conv2.register_forward_hook(get_activation_0(80))
    # neuralnet.layer3[19].conv3.register_forward_hook(get_activation_0(81))
    # neuralnet.layer3[20].conv1.register_forward_hook(get_activation_0(82))
    # neuralnet.layer3[20].conv2.register_forward_hook(get_activation_0(83))
    # neuralnet.layer3[20].conv3.register_forward_hook(get_activation_0(84))
    # neuralnet.layer3[21].conv1.register_forward_hook(get_activation_0(85))
    # neuralnet.layer3[21].conv2.register_forward_hook(get_activation_0(86))
    # neuralnet.layer3[21].conv3.register_forward_hook(get_activation_0(87))
    # neuralnet.layer3[22].conv1.register_forward_hook(get_activation_0(88))
    # neuralnet.layer3[22].conv2.register_forward_hook(get_activation_0(89))
    # neuralnet.layer3[22].conv3.register_forward_hook(get_activation_0(90))
    
    # neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(91))
    # neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(92))
    # neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(93))
    # #neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(98))
    # neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(94))
    # neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(95))
    # neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(96))
    # neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(97))
    # neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(98))
    # neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(99))
    neuralnet.conv1.register_forward_hook(get_activation_0(0))
    
    #neuralnet.layer1[0].conv1.register_forward_hook(get_activation_0(1))
    neuralnet.layer1[0].conv2.register_forward_hook(get_activation_0(1))
    neuralnet.layer1[0].conv3.register_forward_hook(get_activation_0(2))
    
    #neuralnet.layer1[0].downsample[0].register_forward_hook(get_activation_0(5))
    
    #neuralnet.layer1[1].conv1.register_forward_hook(get_activation_0(4))
    neuralnet.layer1[1].conv2.register_forward_hook(get_activation_0(3))
    neuralnet.layer1[1].conv3.register_forward_hook(get_activation_0(4))
    #neuralnet.layer1[2].conv1.register_forward_hook(get_activation_0(7))
    neuralnet.layer1[2].conv2.register_forward_hook(get_activation_0(5))
    neuralnet.layer1[2].conv3.register_forward_hook(get_activation_0(6))
    
    
    #neuralnet.layer2[0].conv1.register_forward_hook(get_activation_0(10))
    neuralnet.layer2[0].conv2.register_forward_hook(get_activation_0(7))
    neuralnet.layer2[0].conv3.register_forward_hook(get_activation_0(8))
    #neuralnet.layer2[0].downsample[0].register_forward_hook(get_activation_0(15))
    #neuralnet.layer2[1].conv1.register_forward_hook(get_activation_0(13))
    neuralnet.layer2[1].conv2.register_forward_hook(get_activation_0(9))
    neuralnet.layer2[1].conv3.register_forward_hook(get_activation_0(10))
    #neuralnet.layer2[2].conv1.register_forward_hook(get_activation_0(16))
    neuralnet.layer2[2].conv2.register_forward_hook(get_activation_0(11))
    neuralnet.layer2[2].conv3.register_forward_hook(get_activation_0(12))
    #neuralnet.layer2[3].conv1.register_forward_hook(get_activation_0(19))
    neuralnet.layer2[3].conv2.register_forward_hook(get_activation_0(13))
    neuralnet.layer2[3].conv3.register_forward_hook(get_activation_0(14))
    
    #neuralnet.layer3[0].conv1.register_forward_hook(get_activation_0(22))
    neuralnet.layer3[0].conv2.register_forward_hook(get_activation_0(15))
    neuralnet.layer3[0].conv3.register_forward_hook(get_activation_0(16))
    #neuralnet.layer3[0].downsample[0].register_forward_hook(get_activation_0(25))
    #neuralnet.layer3[1].conv1.register_forward_hook(get_activation_0(29))
    neuralnet.layer3[1].conv2.register_forward_hook(get_activation_0(17))
    neuralnet.layer3[1].conv3.register_forward_hook(get_activation_0(18))
    #neuralnet.layer3[2].conv1.register_forward_hook(get_activation_0(28))
    neuralnet.layer3[2].conv2.register_forward_hook(get_activation_0(19))
    neuralnet.layer3[2].conv3.register_forward_hook(get_activation_0(20))
    #neuralnet.layer3[3].conv1.register_forward_hook(get_activation_0(31))
    neuralnet.layer3[3].conv2.register_forward_hook(get_activation_0(21))
    neuralnet.layer3[3].conv3.register_forward_hook(get_activation_0(22))
    #neuralnet.layer3[4].conv1.register_forward_hook(get_activation_0(34))
    neuralnet.layer3[4].conv2.register_forward_hook(get_activation_0(23))
    neuralnet.layer3[4].conv3.register_forward_hook(get_activation_0(24))
    #neuralnet.layer3[5].conv1.register_forward_hook(get_activation_0(37))
    neuralnet.layer3[5].conv2.register_forward_hook(get_activation_0(25))
    neuralnet.layer3[5].conv3.register_forward_hook(get_activation_0(26))
    
    #neuralnet.layer3[6].conv1.register_forward_hook(get_activation_0(40))
    neuralnet.layer3[6].conv2.register_forward_hook(get_activation_0(27))
    neuralnet.layer3[6].conv3.register_forward_hook(get_activation_0(28))
    #neuralnet.layer3[7].conv1.register_forward_hook(get_activation_0(43))
    neuralnet.layer3[7].conv2.register_forward_hook(get_activation_0(29))
    neuralnet.layer3[7].conv3.register_forward_hook(get_activation_0(30))
    #neuralnet.layer3[8].conv1.register_forward_hook(get_activation_0(46))
    neuralnet.layer3[8].conv2.register_forward_hook(get_activation_0(31))
    neuralnet.layer3[8].conv3.register_forward_hook(get_activation_0(32))
    #neuralnet.layer3[9].conv1.register_forward_hook(get_activation_0(49))
    neuralnet.layer3[9].conv2.register_forward_hook(get_activation_0(33))
    neuralnet.layer3[9].conv3.register_forward_hook(get_activation_0(34))
    #neuralnet.layer3[10].conv1.register_forward_hook(get_activation_0(52))
    neuralnet.layer3[10].conv2.register_forward_hook(get_activation_0(35))
    neuralnet.layer3[10].conv3.register_forward_hook(get_activation_0(36))
    #neuralnet.layer3[11].conv1.register_forward_hook(get_activation_0(55))
    neuralnet.layer3[11].conv2.register_forward_hook(get_activation_0(37))
    neuralnet.layer3[11].conv3.register_forward_hook(get_activation_0(38))
    #neuralnet.layer3[12].conv1.register_forward_hook(get_activation_0(58))
    neuralnet.layer3[12].conv2.register_forward_hook(get_activation_0(39))
    neuralnet.layer3[12].conv3.register_forward_hook(get_activation_0(40))
    #neuralnet.layer3[13].conv1.register_forward_hook(get_activation_0(61))
    neuralnet.layer3[13].conv2.register_forward_hook(get_activation_0(41))
    neuralnet.layer3[13].conv3.register_forward_hook(get_activation_0(42))
    #neuralnet.layer3[14].conv1.register_forward_hook(get_activation_0(64))
    neuralnet.layer3[14].conv2.register_forward_hook(get_activation_0(43))
    neuralnet.layer3[14].conv3.register_forward_hook(get_activation_0(44))
    #neuralnet.layer3[15].conv1.register_forward_hook(get_activation_0(67))
    neuralnet.layer3[15].conv2.register_forward_hook(get_activation_0(45))
    neuralnet.layer3[15].conv3.register_forward_hook(get_activation_0(46))
    #neuralnet.layer3[16].conv1.register_forward_hook(get_activation_0(70))
    neuralnet.layer3[16].conv2.register_forward_hook(get_activation_0(47))
    neuralnet.layer3[16].conv3.register_forward_hook(get_activation_0(48))
    #neuralnet.layer3[17].conv1.register_forward_hook(get_activation_0(73))
    neuralnet.layer3[17].conv2.register_forward_hook(get_activation_0(49))
    neuralnet.layer3[17].conv3.register_forward_hook(get_activation_0(50))
   # neuralnet.layer3[18].conv1.register_forward_hook(get_activation_0(76))
    neuralnet.layer3[18].conv2.register_forward_hook(get_activation_0(51))
    neuralnet.layer3[18].conv3.register_forward_hook(get_activation_0(52))
    #neuralnet.layer3[19].conv1.register_forward_hook(get_activation_0(79))
    neuralnet.layer3[19].conv2.register_forward_hook(get_activation_0(53))
    neuralnet.layer3[19].conv3.register_forward_hook(get_activation_0(54))
    #neuralnet.layer3[20].conv1.register_forward_hook(get_activation_0(82))
    neuralnet.layer3[20].conv2.register_forward_hook(get_activation_0(55))
    neuralnet.layer3[20].conv3.register_forward_hook(get_activation_0(56))
    #neuralnet.layer3[21].conv1.register_forward_hook(get_activation_0(85))
    neuralnet.layer3[21].conv2.register_forward_hook(get_activation_0(57))
    neuralnet.layer3[21].conv3.register_forward_hook(get_activation_0(58))
    #neuralnet.layer3[22].conv1.register_forward_hook(get_activation_0(88))
    neuralnet.layer3[22].conv2.register_forward_hook(get_activation_0(59))
    neuralnet.layer3[22].conv3.register_forward_hook(get_activation_0(60))
    
    #neuralnet.layer4[0].conv1.register_forward_hook(get_activation_0(91))
    neuralnet.layer4[0].conv2.register_forward_hook(get_activation_0(61))
    neuralnet.layer4[0].conv3.register_forward_hook(get_activation_0(62))
    #neuralnet.layer4[0].downsample[0].register_forward_hook(get_activation_0(98))
    #neuralnet.layer4[1].conv1.register_forward_hook(get_activation_0(94))
    neuralnet.layer4[1].conv2.register_forward_hook(get_activation_0(63))
    neuralnet.layer4[1].conv3.register_forward_hook(get_activation_0(64))
    #neuralnet.layer4[2].conv1.register_forward_hook(get_activation_0(97))
    neuralnet.layer4[2].conv2.register_forward_hook(get_activation_0(65))
    neuralnet.layer4[2].conv3.register_forward_hook(get_activation_0(66))
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
