
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
#import cv2
from torchvision import models
from common import *
import pandas as pd
import time
from resnet import *
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

DIVIDER = '-----------------------------------------'

IMAGE_PATH = './ImageNet100/'
valdata = np.load('./20210812imagenet_val.npy')
traindata = np.load('./20210812imagenet_train.npy')

torch.set_num_threads(1)

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

    save_path1 = os.path.join('pruned_model', 'pruned_para.pth')
    # save_path2 = os.path.join('pruned_model', 'pruned1.pth')

    model = torch.load(save_path1)
    
    model.to(device)
    
    #model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))

    optimizer = optim.SGD(model.parameters(), lr=learnrate,momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 50, eta_min = 0.00001)

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
    accuracy1 = 0
    accuracy_pre = 0
    loss = 0
    loss1 = 0
    acc_signal = np.zeros(10)
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