
'''
Author: IHPC Ella
'''


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast

              
def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))  # L1


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
        #updateBN(model)
        optimizer.step()
        counter += 1
    acc1 = 100. * correct1 / len(train_loader.dataset)
    print('\Train set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct1, len(train_loader.dataset), acc1))
    return acc1,loss.item()

def test(model, device, test_loader):
    '''
    test the model
    '''
    model.eval()

    correct = 0

    with torch.no_grad():
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y = model(data)
            sys.exit()
            output = F.log_softmax(input=y,dim=1)
            test_loss = F.nll_loss(output, target)
            pred1 = output.argmax(dim=1, keepdim=True)
            correct += pred1.eq(target.view_as(pred1)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))

    return acc,test_loss.item()




