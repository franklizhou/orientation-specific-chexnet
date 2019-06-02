import torch
import pandas as pd
import cxp_dataset as CXP
import assembled_model as A
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np


def run_chexpert(PATH_TO_CHECKPOINT, PATH_TO_IMAGES, PATH_TO_CSV):
    
    PATH_TO_LAT = "/home/frank_li_zhou/reproduce-chexnet/results_05302019_adam_lat_uones_noflip/checkpoint_3"
    PATH_TO_PA = "/home/frank_li_zhou/reproduce-chexnet/results_05302019_adam_pa_uones_noflip/checkpoint_2"
    PATH_TO_AP = "/home/frank_li_zhou/reproduce-chexnet/results_05292019_adam_ap_uones_noflip/checkpoint_2"
    PATH_TO_ORIENT = "/home/frank_li_zhou/reproduce-chexnet/orientation_model_06012019_not_trained/checkpoint_6"

    PATH_TO_IMAGES = "/home/frank_li_zhou/"
    PATH_TO_CSV = "/home/frank_li_zhou/CheXpert-v1.0/"
    
    model = A.AssembledModel(PATH_TO_AP, PATH_TO_PA, PATH_TO_LAT, PATH_TO_ORIENT, PATH_TO_IMAGES, PATH_TO_CSV)
    
    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 5  # we are predicting 5 labels
    N_ORIENTS = 3 # we are predicting 3 orientations


    # define torchvision transforms
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    transformed_datasets = {'val': CXP.CXPDataset(
                                path_to_images=PATH_TO_IMAGES,
                                path_to_csv=PATH_TO_CSV,
                                fold='val',
                                transform=data_transforms['val'],
                                orientation='all',
                                verbose = True)
                           }
    
    dataset = transformed_datasets['val']

    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    
    # create empty df
    pred_df = pd.DataFrame(columns=["Image Index"])
    
    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data

        inputs = Variable(inputs.cuda())
        
        batch_size = inputs.shape

        probs = model.run(inputs)
        
        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            
            for k in range(len(dataset.PRED_LABEL)):
                    thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                    
            pred_df = pred_df.append(thisrow, ignore_index=True)
                    
    return pred_df
            
            
        
        
        #probs = outputs.cpu().data.numpy()
        
#         #print(outputs)

#         # get predictions and true values for each item in batch
#         for j in range(0, batch_size[0]):
#             thisrow = {}
#             thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

#             # iterate over each entry in prediction vector; each corresponds to
#             # individual label
#             for k in range(len(dataset.PRED_LABEL)):
#                 thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]

#             pred_df = pred_df.append(thisrow, ignore_index=True)

#         #if(i % 10 == 0):
#         #    print('eval_model: ' + str(i * BATCH_SIZE))
    
#     print(pred_df.to_csv(index = False))
    

def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES, PATH_TO_CSV, metric, dataset=None, verbose=False):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: torchvision transforms to preprocess raw images; same as validation transforms
        model: densenet-121 from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which NIH images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """
    if metric not in ['auc', 'f1']:
        print("make_pred_multilabel: invalid metric:", metric)
    
    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 16

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader on val set if not provided
    if dataset == None:
        dataset = CXP.CXPDataset(
            path_to_images=PATH_TO_IMAGES,
            path_to_csv=PATH_TO_CSV,
            fold="val",
            transform=data_transforms['val'])
        
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    size = len(dataset)

    # create empty dfs
    pred_df = pd.DataFrame(columns=["Image Index"])
    true_df = pd.DataFrame(columns=["Image Index"])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        
        #print(labels)
        #x = 5/0
        
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            for k in range(len(dataset.PRED_LABEL)):
                thisrow["prob_" + dataset.PRED_LABEL[k]] = probs[j, k]
                truerow[dataset.PRED_LABEL[k]] = true_labels[j, k]

            pred_df = pred_df.append(thisrow, ignore_index=True)
            true_df = true_df.append(truerow, ignore_index=True)

        #if(i % 10 == 0):
        #    print('eval_model: ' + str(i * BATCH_SIZE))

            
    if (metric == 'auc'):
        auc_df = pd.DataFrame(columns=["label", "auc"])
    elif (metric == 'f1'):
        f1_df = pd.DataFrame(columns=["label", "f1"])
        
    #print('true_df: ', true_df)
    #print('pred_df: ', pred_df)

    # calc accuracies
    for column in true_df:

        if column not in [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']:
                    continue
        actual = true_df[column]
        pred = pred_df["prob_" + column]
        thisrow = {}
        thisrow['label'] = column
        
        if (metric == 'auc'):
            thisrow['auc'] = np.nan
            try:
                thisrow['auc'] = sklm.roc_auc_score(
                    actual.as_matrix().astype(int), pred.as_matrix())
            except BaseException:
                if verbose:
                    print("can't calculate auc for " + str(column))
            auc_df = auc_df.append(thisrow, ignore_index=True)
        elif (metric == 'f1'):
            thisrow['f1'] = np.nan
            thisrow['f1'] = sklm.f1_score(
                    actual.as_matrix().astype(int), pred.as_matrix())
            try:
                thisrow['f1'] = sklm.f1_score(
                    actual.as_matrix().astype(int), pred.as_matrix())
            except BaseException:
                if verbose:
                    print("can't calculate f1 for " + str(column))
            f1_df = f1_df.append(thisrow, ignore_index=True)
            

    pred_df.to_csv("results/preds.csv", index=False)
    
    if (metric == 'auc'):
        auc_df.to_csv("results/aucs.csv", index=False)
        return pred_df, auc_df
    elif (metric == 'f1'):
        f1_df.to_csv("results/f1.csv", index=False)
        return pred_df, f1_df
    
    
