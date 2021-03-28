# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torch.optim as optim
import torch.nn as nn
import utils.utils_pytorch as t_u
import utils.utils_resnet18 as u_r

import Net.imprint_net as mynet


def create_full_net(dataloaders, train_dataset_size, num_classes, device, PATH_FULL, batch_size, test_set, classes):
    model = mynet.Net(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("Training...")

    num_epochs = 5
    model, val_acc_history, train_acc_history = \
        t_u.train_model(model, dataloaders, device, criterion, optimizer, train_dataset_size,
                        num_epochs=num_epochs, batch_size=batch_size, weight_norm=False)

    # %% Saving the model
    torch.save(model.state_dict(), PATH_FULL)
    my_model = mynet.Net(num_classes)
    my_model.load_state_dict(torch.load(PATH_FULL))
    my_model.to(device)

    # Create test (without fs)
    test_set_load = []
    for img_lbl in test_set:
        if img_lbl[1] <= num_classes-1:
            test_set_load.append(img_lbl)

    # Test
    testloader = torch.utils.data.DataLoader(test_set_load, batch_size=4,
                                             shuffle=True, num_workers=2)

    t_u.basics_perf(testloader, my_model, device, classes)
