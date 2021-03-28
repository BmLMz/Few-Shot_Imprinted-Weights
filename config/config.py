#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:33:26 2021

@author: admin_loc

Script defining variables
"""
import random as rd
import torchvision.transforms as transforms
import torch
import yaml

######################################
###### Global parameters ######
# Number of class of the initial dataset;
# Fs_class will be num_class, initial will be [0:num_classes]
num_classes = 40

# Path to the model basenet/fullnet
PATH = './Net/cifar_basenet_'+str(num_classes)+'.pth'
PATH_FULL = './Net/cifar_full_'+str(num_classes)+'.pth'

# Number of epochs
num_epochs = 15
######################################

######################################
###### Imprinting.py parameters ######
# Number of shots
number_shots = [1,5,10,20]

# Number of times to launch exp to get average perfo
iteration = 4

# Finetuning or not after imprinting
Finetune = True

# Few-Shot class we want to imprint (CIFAR100) to loop over on main_full_exp.py
fs_classes = rd.sample([i for i in range (50,100)], 10)
# Few Shot class to imprint in main.py
fs_class = 80

# Batch size
batch_size = 20

# Randomize immprint parameters
random = False
######################################

######################################
###### initial_net.py ######
CreateNets = False

embedding_size = 256
######################################


######################################
###### Data param ######
# Resnet Transform (from Pytorch)
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
######################################

dict_config = {
    'num_classes':num_classes,
    'PATH':PATH,
    'PATH_FULL': PATH_FULL,
    'num_epochs': 15,
    'number_shots': number_shots,
    'iteration': iteration,
    'Finetune': Finetune,
    'fs_classes': fs_classes,
    'batch_size': batch_size,
    'random': random,
    'embedding_size': embedding_size,
    'CreateNets': CreateNets,
    'fs_class': fs_class
}

with open('./config.yaml', 'w') as file:
    documents = yaml.dump(dict_config, file)
