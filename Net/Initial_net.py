#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:09:27 2021

@author: admin_loc
"""
import torchvision
import torch.nn as nn
from config.conf_class import MyConfig
import yaml

# %% Load config
with open(r'./config.yaml') as file:
    cfg_dict = yaml.load(file, Loader=yaml.FullLoader)

cfg = MyConfig(cfg_dict)



class ResNet18_Encoder(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet18_Encoder, self).__init__()
        model_resnet18 = torchvision.models.resnet18(pretrained=True)

        self.up = nn.Upsample((224, 224))
        
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features
        self.fc = nn.Linear(self.__in_features, cfg.embedding_size)

            
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x