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
Common functions for simple PyTorch MNIST example
'''

'''
Author: 
'''


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
import torch
import torch.nn as nn
import time
#from .utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast

 
def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))  # L1

def normalize(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            data_scale = m.weight.data.clone()
            data_min = data_scale.min()
            data_max = data_scale.max()
            dst = data_max - data_min
            norm_data = (data_scale - data_min).true_divide(dst)
            m.weight.data = norm_data
            #return norm_data

def print_scale(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            data_scale = m.weight.data.clone()
            print(data_scale)
            
#data.to_csv('/home/ihpc/Vitis-AI/20210630kuzushiji-test/files/train.csv',mode = 'a',header = False, index = False)

def train(model, device, train_loader, optimizer, epoch):
    '''
    train the model
    '''
    model.train()

    counter = 0
    correct1 = 0
    print("Epoch "+str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x = model(data)
        output = F.log_softmax(input=x,dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct1 += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward()
        updateBN(model)
        print_scale("data1",model)
        optimizer.step()
        print_scale("data2",model)
        normalize()
        print_scale("data3",model)
        counter += 1
    acc1 = 100. * correct1 / len(train_loader.dataset)
    print('\Train set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct1, len(train_loader.dataset), acc1))
    return acc1,loss.item()


global time_all


def test(model, device, test_loader):
    '''
    test the model
    '''
    model.eval()
    #test_loss = 0
    time_all = 0
    correct = 0

    with torch.no_grad():
        
        #for batch_idx1, (data, target) in enumerate(test_loader):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            time1 = time.time()
            
            y = model(data)
            
            #sys.exit()
            
            time2 = time.time()
            
            time_1 = time2-time1
            
            time_all = time_all + time_1
        
            output = F.log_softmax(input=y,dim=1)
            test_loss = F.nll_loss(output, target)
            pred1 = output.argmax(dim=1, keepdim=True)
            correct += pred1.eq(target.view_as(pred1)).sum().item()
            break
    print(time_all)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))

    return acc,test_loss.item()




