#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 08:23:11 2021

@author: admin_loc
"""

import torch
from torch import autograd
from torch import nn
from config.conf_class import MyConfig

import yaml


# %% Load config
with open('./config.yaml') as file:
    cfg_dict = yaml.load(file, Loader=yaml.FullLoader)

cfg = MyConfig(cfg_dict)


class MyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax()

    def __init__(self):
        super().__init__()
        # Need to be defined
        weights = [1/1600]*cfg.num_classes + [1.0]
        class_weights = torch.FloatTensor(weights)
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)
        my_loss_tp = -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()
        my_loss = torch.sum(my_loss_tp)
        return my_loss