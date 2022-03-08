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
import time
from resnet import *
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

# from thop import profile
# from thop import clever_format


DIVIDER = '-----------------------------------------'

IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')
#valdata = valdata[0:2000]
#traindata = traindata[0:2000]
torch.set_num_threads(4)
#neuralnet = models.mobilenet_v2(pretrained=True)
# print(neuralnet)
# sys.exit()
#traindata = traindata[0:2000]
#print(traindata[0:2000])
#print(traindata)
#sys.exit()

# trewerwa= transforms.Compose(
#         [
#             transforms.Resize(254),
#             #transforms.CenterCrop(224),
#             transforms.CenterCrop(224),
#             #transforms.Pad(2,0),
#             transforms.ToTensor(),  
#             #transforms.Normalize(
#                   #mean=[0.485, 0.456, 0.406],
#                   #std=[0.229, 0.224, 0.225])
#         ]
#         )
# image = Image.open(IMAGE_PATH + traindata[0][0] + "/" + traindata[0][1])#.convert("RGB")
# print(trewerwa(image))
# sys.exit()
# img = cv2.imread(IMAGE_PATH + traindata[0][0] + "/" + traindata[0][1])
# img = cv2.resize(img,(224,224))
# print(img.shape)
# img = img[:,:,::-1].transpose((2,0,1))
# #img.crop((224,224))
# print(img.shape)
# #print(img[0][1])
# sys.exit()
    
# def default_loader(path):
#     #print(Image.open(path))
#     #img = Image.open(path).convert("RGB")
#     #img = cv2.imread(path,cv2.IMREAD_COLOR)
#     img = cv2.imread(path)

    
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

    #model = CNN().to(device)
    # print(model)
    # sys.exit()
    #model = ResNet()
    #model.to(device)
    
    save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    model = torch.load(save_path2)
    
    
    model.to(device)
    
    #model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))

    # for m in model.modules():
    #   if isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 0.5)
    #     nn.init.constant_(m.bias, 0)
#     # print(model)
#     # print("signal")
#     # #sys.exit()
#     # num_parameters = sum([param.nelement() for param in model.parameters()])
#     # print(num_parameters)
#     # sys.exit()
    
    
#     # print(flops)
#     # print(params)
#     # sys.exit()
#     # for k, m in enumerate(model.modules()):
#     #     print("model",k,m)
#     #     print("input")
#     #     input()
#     # print(model)
#     # a = 0
#     # b = 0
#     # for name, param in model.named_parameters():
#     #     #a = a + 1
#     #     print(param.nelement())
#     #     print(name)
#     # print(b)
#     # sys.exit()
# # =============================================================================
# # numbers of parameters
# # =============================================================================
#     # num_parameters = sum([param.nelement() for param in model.parameters()])
#     # print(num_parameters)
    
#     # save_path2 = os.path.join('pruned_model', 'pruned1.pth')  
    
#     # newmodel = torch.load(save_path2)
    
#     # num_parameters1 = sum([param.nelement() for param in newmodel.parameters()])
#     # print(num_parameters1)
    
#     # sys.exit()
    
# # =============================================================================
# # load pretrained parameters
# # =============================================================================
#     
    
    
#     # total0 = 0
#     # total1 = 0
#     # layer_index = 0
#     # for m in model.modules():
#     #   if isinstance(m, nn.BatchNorm2d):
#     #     layer_index = layer_index +1
#     #     if layer_index <= 13:
#     #         total0 += m.weight.data.shape[0]
#     #     else:
#     #         total1 += m.weight.data.shape[0]
            
#     # bn0 = torch.zeros(total0)
#     # bn1 = torch.zeros(total1)
    
#     # index = 0
#     # layer_index = 0
    
#     # for m in model.modules():
        
#     #     if isinstance(m, nn.BatchNorm2d):
#     #         layer_index = layer_index +1
#     #         if layer_index <= 13:
#     #             size = m.weight.data.shape[0]
#     #             bn0[index:(index+size)] = m.weight.data.abs().clone()
#     #             index += size
#     #     else:
                
#     # y, i = torch.sort(bn)
#     # thre_index = int(total * 0.55)
#     # thre = y[thre_index]
#     # print(thre)
#     # #sys.exit()
#     pruned = 0
#     cfg = []
#     cfg_mask = []
    
#     total = 0
#     layer_index = 0
#     for m in model.modules():
#       if isinstance(m, nn.BatchNorm2d):
#           layer_index = layer_index + 1
#           if layer_index <= 13:
#               total += m.weight.data.shape[0]
#     bn = torch.zeros(total)
    
#     index = 0
#     layer_index = 0
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             layer_index = layer_index + 1
#             if layer_index <= 13:
#                 size = m.weight.data.shape[0]
#                 bn[index:(index+size)] = m.weight.data.abs().clone()
#                 index += size
#     y, i = torch.sort(bn)
#     thre_index = int(total * 0.85)
#     thre = y[thre_index]
#     print(thre)
    
# #     pruned = 0
#     cfg = []
# #     cfg_mask = []

# # # =============================================================================
# # #     initialize the scale factor
# # # =============================================================================
    # for m in model.modules():
    #   if isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 0.5)
    #     nn.init.constant_(m.bias, 0)
    
#     layer_index = 0
#     for k, m in enumerate(model.modules()):
#         #print("pruned",pruned)
#         if isinstance(m, nn.BatchNorm2d):
#             layer_index = layer_index + 1
#             if layer_index <= 13:
#                 weight_copy = m.weight.data.abs().clone()
#                 # print(weight_copy)
#                 # print("input")
#                 # input()
#                 #mask = weight_copy.gt(thre).float().cuda()
#                 mask = weight_copy.gt(thre).float().cuda()
#                 pruned = pruned + mask.shape[0] - torch.sum(mask)
#                 m.weight.data.mul_(mask)
#                 m.bias.data.mul_(mask)
#                 cfg.append(int(torch.sum(mask)))
#                 cfg_mask.append(mask.clone())
#                 print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
#                     format(k, mask.shape[0], int(torch.sum(mask))))
#             else:
#                 weight_copy = m.weight.data.abs().clone()
#                 # print(weight_copy)
#                 # print("input")
#                 # input()
#                 #mask = weight_copy.gt(thre).float().cuda()
#                 mask = weight_copy.gt(0).float().cuda()
#                 pruned = pruned + mask.shape[0] - torch.sum(mask)
#                 m.weight.data.mul_(mask)
#                 m.bias.data.mul_(mask)
#                 cfg.append(int(torch.sum(mask)))
#                 cfg_mask.append(mask.clone())
#                 print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
#                     format(k, mask.shape[0], int(torch.sum(mask))))
#         elif isinstance(m, nn.MaxPool2d): 
#             cfg.append('M')      

#     pruned_ratio = pruned/total
  
#     newmodel =  vgg('vgg16_bn', cfg, True, False, True)
    
#     #print(cfg)
#     #model = newmodel()
    
#     # num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    
#     savepath = os.path.join('pruned_model', "prune.txt")
#     # with open(savepath, "w") as fp:
#     #     fp.write("Configuration: \n"+str(cfg)+"\n")
#     #     fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
#     #     fp.write("Test accuracy: \n"+str(123))
    
#     # layer_id_in_cfg = 0
#     # start_mask = torch.ones(3)
#     # end_mask = cfg_mask[layer_id_in_cfg]

# # =============================================================================
# # Prunning the model
# # =============================================================================
#     a = 0
#     layer_id_in_cfg = 0
#     start_mask = torch.ones(3)
#     end_mask = cfg_mask[layer_id_in_cfg]
#     for [m0, m1] in zip(model.modules(), newmodel.modules()):

#         if isinstance(m0, nn.BatchNorm2d):
#             idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#             if idx1.size == 1:
#                 idx1 = np.resize(idx1,(1,))
#             m1.weight.data = m0.weight.data[idx1.tolist()].clone()
#             m1.bias.data = m0.bias.data[idx1.tolist()].clone()
#             m1.running_mean = m0.running_mean[idx1.tolist()].clone()
#             m1.running_var = m0.running_var[idx1.tolist()].clone()
            
#             layer_id_in_cfg += 1
#             start_mask = end_mask.clone()
#             if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
#                 end_mask = cfg_mask[layer_id_in_cfg]
                
#         elif isinstance(m0, nn.Conv2d):
#             if a <= 14:
#                 idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#                 idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#                 print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#                 if idx0.size == 1:
#                     idx0 = np.resize(idx0, (1,))
#                 if idx1.size == 1:
#                     idx1 = np.resize(idx1, (1,))
#                 w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#                 w1 = w1[idx1.tolist(), :, :, :].clone()
#                 m1.weight.data = w1.clone()
#                 if a == 14:
#                     start_mask = end_mask.clone()
#                     #end_mask = torch.ones(1120)      
#                 #print(m1.weight.data)
#             else:
#                 idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#                 #idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#                 end_mask = torch.ones(1120)
#                 idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#                 print("index")
#                 print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#                 #input()
#                 if idx0.size == 1:
#                     idx0 = np.resize(idx0, (1,))

#                 if idx1.size == 1:
#                     idx1 = np.resize(idx1, (1,))
#                 w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#                 w1 = w1[idx1.tolist(), :, :, :].clone()
#                 m1.weight.data = w1.clone()
                
#                 # print("alert")
#                 # input()
#                 # print(m1.weight.data) 
#             a = a + 1

#     # model = newmodel
#     # print(model)
#     # sys.exit()
# # =============================================================================
# #     save the model and the parameters
# # =============================================================================

    #save_path1 = os.path.join('pruned_model', 'pruned1.pth')
    #save_path0 = os.path.join('pruned_model', "pruned0.pth")
    #torch.save(newmodel.state_dict(), savepath) 
    
    #torch.save(newmodel, save_path2) 
    
    #model = torch.load(save_path1)
    #model.to(device)
    #print(model)
    #sys.exit()
   # model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))
    #sys.exit()
    #print(model2)
    #torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join('pruned_model', 'pruned.pth.tar'))
    
    # model.load_state_dict(torch.load(os.path.join('pruned_model', 'pruned.pth'),map_location="cpu"))
    #model = newmodel

    #optimizer = optim.Adam(model.parameters(), lr=learnrate)
    #optimizer = optim.Adam(model.parameters(), lr=learnrate, weight_decay=0.0001)

    optimizer = optim.SGD(model.parameters(), lr=learnrate,momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 50, eta_min = 0.00001)
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
    acc_signal = np.zeros(10)
    accuracy_pre = 0
    lr_list = []
    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        
        time1 = time.time()
        accuracy,loss = train(model, device, train_loader, optimizer, epoch)
        lr_list.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        time2 = time.time()
        print(time2-time1)
        print("lr",optimizer.param_groups[0]["lr"])
        list0= [epoch,accuracy,loss]
        data = pd.DataFrame([list0])
        data.to_csv('./train.csv',mode = 'a',header = False, index = False)
        
        accuracy1, loss1 = test(model, device, test_loader)
        
        # for i in range(9):
        #     acc_signal[i] = acc_signal[i+1]
        # acc_signal[9] = accuracy
        # if (acc_signal[9]-acc_signal[0]) <= 0.1:
        #     learnrate = learnrate * 0.1
        #     for i in range(10):
        #         acc_signal[i] = 0
        #         optimizer = optim.SGD(model.parameters(), lr=learnrate,momentum=0.9, weight_decay=0.0001, nesterov=True)
        # print(acc_signal)
        # print("lr",learnrate)
        list1 = [epoch,accuracy1,loss1]
        data1 = pd.DataFrame([list1])
        data1.to_csv('./val.csv',mode = 'a',header = False, index = False)
        
        # if learnrate <= 0.000001:
        #     sys.exit()
        if accuracy_pre < accuracy1:
            accuracy_pre = accuracy1
            
            shutil.rmtree(float_model, ignore_errors=True)    
            os.makedirs(float_model)   
            save_path = os.path.join(float_model, 'f_model.pth')
            
            torch.save(model.state_dict(), save_path) 
            print('Trained model written to',save_path)
        data.to_csv('./lr.csv',mode = 'a',header = False, index = False)
        print("best accuracy",accuracy_pre)

    # save the trained model
    # shutil.rmtree(float_model, ignore_errors=True)    
    # os.makedirs(float_model)   
    # save_path = os.path.join(float_model, 'f_model.pth')
    
    # torch.save(model.state_dict(), save_path) 
    
    # print('Trained model written to',save_path)

    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dset_dir',    type=str,  default='dataset',     help='Path to test & train datasets. Default is dataset')
    ap.add_argument('-b', '--batchsize',   type=int,  default=20,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=50,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.01,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
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
