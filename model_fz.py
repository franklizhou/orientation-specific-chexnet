from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import os
import time
from shutil import copyfile
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxp_dataset as CXP
import eval_model_fz as E

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, last_train_loss, best_val_acc, metric, epoch, best_epoch, LR, WD):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """
    
    state = {
        'model': model,
        'last_train_loss': last_train_loss,
        'best_val_acc': best_val_acc,
        'metric': metric,
        'epoch': epoch,
        'best_epoch': best_epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR,
        'WD': WD,
    }

    torch.save(state, 'results/checkpoint_' + str(epoch))


def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        dataset,
        data_transforms,
        PATH_TO_IMAGES,
        PATH_TO_CSV):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_val_acc = 0
    best_epoch = -1
    last_train_loss = -1
    last_val_acc = 0

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        with open("results/logger", 'a') as logfile:
            logfile.write('Epoch {}/{}\n'.format(epoch, num_epochs))
            logfile.write('-' * 10 + '\n')

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                #labels = Variable(labels.cuda()).long()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                #print(outputs)
                #print(labels)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item() * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            print(phase + ' epoch {}: loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))
            with open("results/logger", 'a') as logfile:
                logfile.write(phase + ' epoch {}: loss {:.4f} with data size {}\n'.format(
                epoch, epoch_loss, dataset_sizes[phase]))
            
            time_elapsed = time.time() - since
            print(phase + ' epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            with open("results/logger", 'a') as logfile:
                logfile.write(phase + ' epoch complete in {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))
            
            if phase == 'train':
                last_train_loss = epoch_loss
                
            if phase == 'val':
                _, metric = E.make_pred_multilabel(data_transforms, model, 
                                                   PATH_TO_IMAGES, PATH_TO_CSV, 
                                                   #'auc')
                                                   'auc', dataset=dataset)
                
                auc = metric.as_matrix(columns=metric.columns[1:])
                last_val_acc = auc[~np.isnan(auc)].mean() 
                
                print(metric)
                with open("results/logger", 'a') as logfile:
                    print(metric, file=logfile)
                
                print('mean epoch validation accuracy:', last_val_acc)
                with open("results/logger", 'a') as logfile:
                    logfile.write('mean epoch validation accuracy: ' + str(last_val_acc) + '\n')

#             # decay learning rate if no train loss improvement in this epoch
#             if phase == 'train' and epoch_loss > best_loss:
#                 print("Running with LR decay on TRAIN")
#                 logger.append("Running with LR decay on TRAIN")
#                 print("decay loss from " + str(LR) + " to " +
#                       str(LR / 10) + " as not seeing improvement in train loss")
#                 LR = LR / 10
#                 # create new optimizer with lower learning rate
#                 optimizer = optim.SGD(
#                     filter(
#                         lambda p: p.requires_grad,
#                         model.parameters()),
#                     lr=LR,
#                     momentum=0.9,
#                     weight_decay=weight_decay)
#                 print("created new optimizer with LR " + str(LR))

#             # checkpoint model if has best train loss yet
#             if phase == 'train' and epoch_loss < best_loss:
#                 best_loss = epoch_loss
#                 best_epoch = epoch
#                 checkpoint(model, best_loss, epoch, LR)
                
            # decay learning rate if no val accuracy improvement in this epoch
            if phase == 'val' and last_val_acc < best_val_acc and epoch >= best_epoch + 2:
                print("Running with LR decay on val accuracy")
                with open("results/logger", 'a') as logfile:
                    logfile.write("Running with LR decay on val accuracy\n")
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val accuracy")
                with open("results/logger", 'a') as logfile:
                    logfile.write("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val accuracy\n")
                LR = LR / 10
                # create new optimizer with lower learning rate
#                 optimizer = optim.SGD(
#                     filter(
#                         lambda p: p.requires_grad,
#                         model.parameters()),
#                     lr=LR,
#                     momentum=0.9,
#                     weight_decay=weight_decay)
                optimizer = optim.Adam(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=weight_decay)
                
                print("created new optimizer with LR " + str(LR))
                with open("results/logger", 'a') as logfile:
                        logfile.write("created new optimizer with LR " + str(LR) + '\n')

            # keep track of best train loss
            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                
            # checkpoint model if has best val accuracy yet
            if phase == 'val' and last_val_acc > best_val_acc:
                best_val_acc = last_val_acc
                best_epoch = epoch
                
            if phase == 'val':
                print('saving checkpoint_' + str(epoch))
                with open("results/logger", 'a') as logfile:
                    logfile.write('saving checkpoint_' + str(epoch) + '\n')
                checkpoint(model, last_train_loss, last_val_acc, metric, epoch, best_epoch, LR, weight_decay)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss", "average auc"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss, last_val_acc])

        print("best epoch: ", best_epoch)
        with open("results/logger", 'a') as logfile:
            logfile.write("best epoch: " + str(best_epoch) + '\n')
                    
        print("best train loss: ", best_loss)
        with open("results/logger", 'a') as logfile:
            logfile.write("best train loss: " + str(best_loss) + '\n')
        
        print("best val accuracy: ", best_val_acc)
        with open("results/logger", 'a') as logfile:
            logfile.write("best val accuracy: " + str(best_val_acc) + '\n')
                    
        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")
            with open("results/logger", 'a') as logfile:
                logfile.write("completed " + str(total_done) + " so far in epoch\n")

        # break if no val loss improvement in 4 epochs
        if ((epoch - best_epoch) >= 4):
            print("no improvement in 4 epochs, break")
            with open("results/logger", 'a') as logfile:
                logfile.write("no improvement in 4 epochs, break\n")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    with open("results/logger", 'a') as logfile:
        logfile.write('Training complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint_' + str(best_epoch))
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, PATH_TO_CSV, LR, WEIGHT_DECAY, NUM_IMAGES=223414, PATH_TO_CHECKPOINT=None):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    
    try:
        if os.path.exists('results/'):
            print("Remove or rename results directory")
            return
        else:
            rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")
    
    
    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    
    
    print("Running with WD, LR:", WEIGHT_DECAY, LR)
    
    with open("results/logger", 'a') as logfile:
        logfile.write("Running with WD, LR:" + str(WEIGHT_DECAY) + ' ' + str(LR) + '\n')

    
    out = []

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224), # changed for pytorch 4.0.1
            # because scale doesn't always give 224 x 224, this ensures 224 x
            # 224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXP.CXPDataset(
        path_to_images=PATH_TO_IMAGES,
        path_to_csv=PATH_TO_CSV,
        fold='train',
        uncertain = 'zeros',
        transform=data_transforms['train'],
        sample = NUM_IMAGES,
        verbose = True
    )
    transformed_datasets['val'] = CXP.CXPDataset(
        path_to_images=PATH_TO_IMAGES,
        path_to_csv=PATH_TO_CSV,
        fold='val',
        transform=data_transforms['val'],
        verbose = True
    )
    
    print("Size of train set:", len(transformed_datasets['train']))
    print("Size of val set:", len(transformed_datasets['val']))

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
        
    if PATH_TO_CHECKPOINT == None:
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        # add final layer with # outputs in same dimension of labels with sigmoid
        # activation
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

        # put model on GPU
        model = model.cuda()
    else:
        checkpoint = torch.load(PATH_TO_CHECKPOINT)
        model = checkpoint['model']

    # define criterion, optimizer for training
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
#     optimizer = optim.SGD(
#         filter(
#             lambda p: p.requires_grad,
#             model.parameters()),
#         lr=LR,
#         momentum=0.9,
#         weight_decay=WEIGHT_DECAY)
    
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=WEIGHT_DECAY,
    )
    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}
    
    print("Model training start")

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY, 
                                    dataset=transformed_datasets['train'], data_transforms=data_transforms, 
                                    PATH_TO_IMAGES=PATH_TO_IMAGES, PATH_TO_CSV=PATH_TO_CSV)
    
    print("Model training complete")

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES, PATH_TO_CSV, 'auc')

    return preds, aucs
