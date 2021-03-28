# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% Imports
import torch
import torch.optim as optim
import utils.utils_pytorch as t_u

# Add
import torch.nn.parallel
import torch.utils.data
import utils.utils_resnet18 as u_r
import Net.imprint_net as mynet
from config.conf_class import MyConfig
import copy
import os
import numpy as np
import utils.my_criterion as crit
import yaml
from Net.CreateBaseNetImprint import create_net
from datetime import datetime
from shutil import copyfile

# %% Define device
device = t_u.get_default_device()
print("Using " + str(device))
print()


# %% Load config
with open(r'./config.yaml') as file:
    cfg_dict = yaml.load(file, Loader=yaml.FullLoader)

cfg = MyConfig(cfg_dict)


# %% Save perfos
dict_acc = {}
for i in cfg.number_shots:
    dict_acc['global_acc_' + str(i)] = []
    dict_acc['fs_class_' + str(i)] = []
    dict_acc['global_acc_ft_' + str(i)] = []
    dict_acc['fs_class_ft_' + str(i)] = []

# %% Create nets
if cfg.CreateNets:
    # Get data
    train_set, val_set, _, test_set, my_fewshots, classes =\
        u_r.get_cifar100(transform=cfg.transform_cifar,
                         num_classes=cfg.num_classes, fs_class=80)
    # Datasets
    dataloaders_dict, _, train_dataset_size = \
        u_r.dataloaders_dict_generator(
            train_set, val_set, test_set, cfg.batch_size, cfg.num_classes)

    create_net(dataloaders_dict, classes, train_dataset_size, cfg.num_classes,
               test_set, device, cfg.PATH, cfg.PATH_FULL, cfg.batch_size)


# %% Imprintng
# Iteration to try different type of few shot set on different few shot class
for i, fs_class in enumerate(cfg.fs_classes):
    for num_shots in cfg.number_shots:
        for iteration in range(cfg.iteration):
            print('Class number :', fs_class, 'with', num_shots, 'shots.')
            print('Iteration : ', round(iteration+1), '/', round(cfg.iteration))
            train_set, val_set, val_set_ft, test_set, my_fewshots, classes =\
                u_r.get_cifar100(cfg.transform_cifar,
                                 fs_class, cfg.num_classes)

            # Datasets
            dataloaders_dict, testloader, train_dataset_size = \
                u_r.dataloaders_dict_generator(
                    train_set, val_set, test_set, cfg.batch_size, cfg.num_classes)

            novel_loader_train, few_shots =  \
                u_r.novel_class_dataloader(
                    my_fewshots, num_shots, cfg.num_classes, cfg.batch_size)

            # Get model
            model = mynet.Net(cfg.num_classes)
            model.load_state_dict(torch.load(cfg.PATH_FULL))
            model.to(device)

            # Imprinting
            model = u_r.imprint(novel_loader_train, model,
                                cfg.num_classes, device, cfg.random)
            model.weight_norm()

            print("All done")
            print()

            global_acc, fs_acc = t_u.basics_perf(
                testloader, model, device, classes, fs=True)

            dict_acc['global_acc_' + str(num_shots)].append(global_acc)
            dict_acc['fs_class_' + str(num_shots)].append(fs_acc)

            #  If we want to finetune the model after imprinting
            if cfg.Finetune:
                ft_train, ft_val =\
                    u_r.fine_tune_dataloader(
                        train_set, val_set_ft, few_shots, num_shots, cfg.num_classes, cfg.batch_size)

                criterion = crit.MyLoss()
                optimizer = optim.SGD(model.parameters(),
                                      lr=0.001, momentum=0.9)

                print("Training...")
                dataloaders = {'train': ft_train, 'val': ft_val}
                model.train()

                if num_shots > 5:
                    num_epochs = 4
                else:
                    num_epochs = 2
                model, val_acc_history, train_acc_history = \
                    u_r.train_model(model, dataloaders, device, criterion, optimizer, train_dataset_size,
                                    num_epochs=num_epochs, batch_size=cfg.batch_size, weight_norm=True)

                glob_acc_ft, fs_acc_ft = t_u.basics_perf(
                    testloader, model, device, classes, fs=True)

                dict_acc['global_acc_ft_' + str(num_shots)].append(glob_acc_ft)
                dict_acc['fs_class_ft_' + str(num_shots)].append(fs_acc_ft)

                # Clear cuda cache
                del model, dataloaders_dict
                torch.cuda.empty_cache()


# %% Stats
dict_acc_mean = copy.deepcopy(dict_acc)
for k in dict_acc_mean:
    dict_acc_mean[k] = np.mean(dict_acc_mean[k])

print(dict_acc_mean)
print()
print(dict_acc)

# %%
dict_acc_var = copy.deepcopy(dict_acc)
for k in dict_acc_var:
    dict_acc_var[k] = np.var(dict_acc_var[k])

print(dict_acc_var)

# %%
dict_acc_int = copy.deepcopy(dict_acc_var)
for k in dict_acc_int:
    dict_acc_int[k] = 1.96*np.sqrt(dict_acc_int[k])/np.sqrt(
        (len(cfg.fs_classes)*cfg.iteration*len(cfg.number_shots)))

print(dict_acc_int)

dicos = [dict_acc_mean, dict_acc, dict_acc_var, dict_acc_int]
path_res = './Results/'+str(datetime.now())[:-7].replace(' ','')
os.mkdir(path_res)
with open(path_res+'/Results.txt', 'w') as f:
    for di in dicos:
        f.write("#"*30+"\n")
        for k in di:
            f.write(str(k) + ' : ' + str(di[k]) + "\n")
        f.write("#"*30+"\n"+"\n")
copyfile('./config.yaml', path_res+'/config.yaml')