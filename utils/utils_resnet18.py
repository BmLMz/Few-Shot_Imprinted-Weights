##################################
########## Libraries #############
##################################

# Import torch
import torch
import torch.nn as nn
import torchvision

# Other libraries
import time
import yaml
import copy
import random as rd
import numpy as np
from config.conf_class import MyConfig

# %% Load config
with open(r'./config.yaml') as file:
    cfg_dict = yaml.load(file, Loader=yaml.FullLoader)

cfg = MyConfig(cfg_dict)


##################################
########## Dataset ###############
##################################
def get_cifar100(transform, fs_class, num_classes):
    '''
    Parameters
    ----------
    transform : transform
        Transform for dataset load
    fs_class: Int
        The FS class to consider
    num_shots: Int
        Number of shots to consider
    num_classes: Int
        Number of classes of the dataset (FS included)
        
    Returns
    -------
    train_set,test_set,val_set : LIST
        List tuples (img,lbl)
    '''
    # num_classes is the number of classes, num_classes-1 is the label of the
    # last class => the Few Shot class, num_classes-2 is the last label of the
    # classes without fs_classes
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
    
    # All classes of the dataset
    all_classes = trainset.classes
    classes = all_classes[:num_classes] + [all_classes[fs_class]]
    
    trainset = list(trainset)
    
    # Get FewShot instances
    my_fewshots = []
    for img_lbl in trainset:
        if img_lbl[1] == fs_class:
            # Reencode class
            img_lbl = list(img_lbl)
            img_lbl[1] = num_classes
            img_lbl = tuple(img_lbl)
            my_fewshots.append(img_lbl)
            
    # Train/Val Sets
    train_set_tp = trainset[:int(0.8*len(trainset))]
    val_set_tp = trainset[int(0.2*len(trainset)):]
    
    # Filter datasets
    # Test_set
    test_set = []
    test_set_tp = list(testset)
    for img_lbl in test_set_tp:
        if img_lbl[1] <= num_classes-1:
            test_set.append(img_lbl)
        elif img_lbl[1] == fs_class:
            img_lbl = list(img_lbl)
            img_lbl[1] = num_classes
            img_lbl = tuple(img_lbl)
            test_set.append(img_lbl)
    
    # Train_set
    train_set = []
    for img_lbl in train_set_tp:
        # Decalage python
        if img_lbl[1] <= num_classes-1:
            train_set.append(img_lbl)
            
    
    # Val_set
    val_set = []
    val_set_ft = []
    for img_lbl in val_set_tp:
        if img_lbl[1] <= num_classes-1:
            val_set.append(img_lbl)
            val_set_ft.append(img_lbl)
        elif img_lbl[1] == fs_class:
            img_lbl = list(img_lbl)
            img_lbl[1] = num_classes
            img_lbl = tuple(img_lbl)
            val_set_ft.append(img_lbl)
            
    return train_set, val_set, val_set_ft, test_set, my_fewshots, classes


def dataloaders_dict_generator(train_arr, val_arr, test_arr, batch_size, num_classes):
    '''
    Parameters
    ----------
    train_arr, val_arr, test_arr : Lists
        Lists of tuple (img, lbl)
    input_size : Int
        Size of the input to create Dataset from MyDataset class
    batch_size : Int
        Batch size
    num_classes : Int
        Number of classes (FS included)
    num_shots : Int
        Number of shots

    Returns
    -------
    dataloaders_dict : dict
        Train and Val set in a dict 
    test_arr : array
        Array with tuples (img,lbl) from test_set
    train_dataset_size : Int
        Size of train_dataset
    '''
    # Balance classes for an unbalanced dataset
    weights_val = make_weights_for_balanced_classes(val_arr, num_classes)
    weights_val = torch.DoubleTensor(weights_val)
    sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))

    weights_train = make_weights_for_balanced_classes(train_arr, num_classes)
    weights_train = torch.DoubleTensor(weights_train)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))
    
    # Train
    trainloader = torch.utils.data.DataLoader(train_arr, batch_size=batch_size,
                                              num_workers=2, sampler=sampler_train)
    # Validation
    valloader = torch.utils.data.DataLoader(val_arr, batch_size=batch_size,
                                              num_workers=2, sampler=sampler_val)
    
    # Test
    testloader = torch.utils.data.DataLoader(test_arr, batch_size=4,
                                          shuffle=True, num_workers=2)

    # Create training and validation dataloaders
    dataloaders_dict = {'train': trainloader, 'val': valloader}
    
    # Define train_dataset_size
    train_dataset_size = len(train_arr)

    return dataloaders_dict, testloader, train_dataset_size


def novel_class_dataloader(my_fewshots, num_shots, num_classes, batch_size):
    '''
    Parameters
    ----------
    my_fewshots : List
        List containing (img, lbl) of fewshot instance
    num_shots : Int
        Number of shots
    iteration : Int
        Number of iteration
    batch_size : Int
        batch_size

    Returns
    -------
    novel_loader_train : TYPE
        DESCRIPTION.
    novel_loader_val : TYPE
        DESCRIPTION.

    '''
    # Train/Val Sets
    train_arr = rd.sample(my_fewshots, num_shots)

    novel_loader_train = torch.utils.data.DataLoader(train_arr, batch_size=batch_size,
                                              num_workers=4, shuffle=True)

    return novel_loader_train, train_arr



def fine_tune_dataloader(train_set, val_set, few_shots, num_shots, num_classes, batch_size):
    '''
    Parameters
    ----------
    my_fewshots : List
        List containing (img, lbl) of fewshot instance
    num_shots : Int
        Number of shots
    iteration : Int
        Number of iteration
    batch_size : Int
        batch_size

    Returns
    -------
    train/val : dataloaders
    train/val dataset with train and few shit 
    '''
    # Dict of images per classes in the first ones
    dict_train_tl = {}
    dict_val_tl = {}
    for i in range (0, num_classes):
        dict_train_tl[str(i)] = []
    for i in range(0, num_classes+1):
        dict_val_tl[str(i)] = []
    
    for img_lbl in train_set:
        dict_train_tl[str(img_lbl[1])].append(img_lbl)
    for img_lbl in val_set:
        dict_val_tl[str(img_lbl[1])].append(img_lbl)


    temp = []
    for k in dict_train_tl:
        temp += [len(dict_train_tl[k])]
    nu = int(np.mean(temp)/num_shots)
    
    
    # Train/Val Sets
    train_arr = few_shots*nu + train_set
    val_arr = val_set
    
    fine_tune_train = torch.utils.data.DataLoader(train_arr, batch_size=batch_size,
                                              num_workers=4, shuffle=True)
    fine_tune_val = torch.utils.data.DataLoader(val_arr, batch_size=batch_size,
                                              num_workers=4, shuffle=True)
    return fine_tune_train, fine_tune_val



def make_weights_for_balanced_classes(dataset, num_classes):
    '''
    Parameters
    ----------
    dataset : list
        List tuples (img,lbl)
    num_classes : Int
        Number of classes

    Returns
    -------
    weight : List
        List of weights per class

    '''
    all_class_num = num_classes + 1
    count = [0] * all_class_num
    weight_per_class = [0.] * all_class_num
    
    # Count image per class
    for imglbl in dataset:
        count[imglbl[1]] += 1
        
    N = float(sum(count))
    for i in range (num_classes):
        weight_per_class[i]= N /float(count[i])
    weight =  [0] * len(dataset)
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val[1]]
    return weight



########################################
########## Imprint #############
########################################
def imprint(novel_loader, model, num_classes, device, random):
    '''
    Parameters
    ----------
    novel_loader : Dataloader
        Dataloader
    model : model
        The model to imprint
    num_classes : Int
        Number of classes
    device : device
    random : Boolean
        DESCRIPTION.

    Returns
    -------
    model : model
        Our model imprinted
    '''
    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (img, lbl) in enumerate(novel_loader):
            img = img.to(device)
            lbl = lbl.float().to(device)

            # compute output
            output = model.extract(img)
            
            if batch_idx == 0:
                output_stack = output
                target_stack = lbl
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, lbl), 0)
                
    new_weight = torch.zeros(1, cfg.embedding_size)
    
    if random:
        tmp = torch.randn(cfg.embedding_size)
    else:
        # If there is only one instance of Few-Shot
        if len(target_stack) == 1:
            tmp = output_stack
        else:
            tmp = output_stack[target_stack == (num_classes)].mean(0)

    new_weight[0] = tmp / tmp.norm(p=2)
    
    weight = torch.cat((model.classifier.fc.weight.data, new_weight.cuda()))  

    model.classifier.fc = nn.Linear(cfg.embedding_size, num_classes+1, bias=False)
    model.classifier.fc.weight.data = weight

    return model
 
    
########################################
########## TRAINING #############
########################################
def train_model(model, dataloaders, device, criterion, optimizer, train_dataset_size, num_epochs=5, batch_size=500, weight_norm=True):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            step=0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.to(device))
                    _, preds = torch.max(outputs, 1)
                    compt = 0
                    if phase == 'train':
                        if compt % 2 == 0:
                            old1 = model.classifier.fc.weight.data
                        loss.backward()
                        optimizer.step()
                        if compt % 2 == 0:
                            new1 = model.classifier.fc.weight.data
                            weight1 = torch.cat((old1[:-1], new1[-1].reshape(1, cfg.embedding_size)))
                            model.classifier.fc.weight.data = weight1
                        if weight_norm:
                            model.weight_norm()
                        if (step%100==0):
                            print('Batch :', step, '/', round(train_dataset_size/ batch_size, 0))
                        compt += 1
                step+=1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            elif phase == 'train':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history
