#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:18:47 2021

@author: ihpc
"""

from ctypes import CDLL
import time
import os
import sys
from PIL import Image
import torch
import torchvision
import torchprof
from torchvision import models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import torch.nn as nn
import mkl
import random
import shutil
train_dataset = []
test_dataset = []
val_dataset = []

def split(dataset,shuffle = False, ratio = 0.8):
    num = len(dataset)
    offset1 = int(num*ratio)
    offset2 = int(offset1*ratio)
    if shuffle:
        random.shuffle(dataset)
        
    if num == 0 or offset1 < 1:
        return dataset,[],[]
    if offset2 < 1:
        return dataset[:offset1],dataset[offset1:offset2],[]
    
    train = dataset[:offset2]
    val = dataset[offset2:offset1]
    test = dataset[offset1:]
    return train, val,test

IMAGE_PATH = './ImageNet100/'
image_names = os.listdir(IMAGE_PATH)
image_names.sort()
# print(image_names[0])
# random.shuffle(image_names)
# print(image_names[0])
# print(len(image_names))
# image_names = image_names[0:100]

# for i in range(100):
    
#     old_file = IMAGE_PATH + image_names[i]
#     new_file = "./ImageNet100/" + image_names[i]
#     shutil.copyfile(old_file,new_file)
#     input()

i = 0
for img_path in image_names:
    img_path2 = os.listdir(IMAGE_PATH + '/' + img_path)
    img_path2.sort()
    
    img_train,img_val,img_test = split(img_path2)
    
    for img_file in img_train:
        train_dataset.append((img_path,img_file,int(i)))

    for img_file in img_val:
        val_dataset.append((img_path,img_file,int(i)))
        
    for img_file in img_test:
        #img = Image.open(IMAGE_PATH + '/' + img_file)
        test_dataset.append((img_path,img_file,int(i)))
    
    i = i + 1
#print(train_dataset)
print(len(train_dataset))
# np.save('20210812imagenet_train.npy',train_dataset)
# np.save('20210812imagenet_val.npy',val_dataset)
# np.save('20210812imagenet_test.npy',test_dataset)
    
print("over") 
sys.exit()

# for img_upath in image_names:
#     img_upath = os.listdir(IMAGE_PATH + '/' + img_upath)
#     img_upath.sort()
#     for img_file in img_upath:
#         #img = Image.open(IMAGE_PATH + '/' + img_file)
#         data_set.append((img_file,int(i)))
#     #print(data_set)
#         #print(img_file)
#     #print("input")
#     #input()
#     i = i + 1
    
# np.save('kuzushiji.npy',data_set)
# with open("kuzushiji.txt","w") as f:
#   f.write(str(data_set))
# #np.save('kuzushiji.txt',data_set)
# print("over")