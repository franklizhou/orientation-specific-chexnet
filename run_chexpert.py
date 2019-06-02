import torch
import pandas as pd
import cxp_dataset as CXP
import assembled_model as A
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable
import numpy as np
import sys


def run_chexpert(PATH_TO_IMAGES, PATH_TO_CSV, val=False):
    
    PATH_TO_LAT = "src/model_lat"
    PATH_TO_PA = "src/model_pa"
    PATH_TO_AP = "src/model_ap"
    PATH_TO_ORIENT = "src/model_orientation"
    
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

    if val:
        transformed_datasets = {'val': CXP.CXPDataset(
                                    path_to_images=PATH_TO_IMAGES,
                                    path_to_csv=PATH_TO_CSV,
                                    fold='val',
                                    transform=data_transforms['val'],
                                    orientation='all',
                                    verbose = False)
                               }
    else:
        transformed_datasets = {'val': CXP.CXPDataset(
                                    path_to_images=PATH_TO_IMAGES,
                                    path_to_csv=PATH_TO_CSV,
                                    fold=None,
                                    transform=data_transforms['val'],
                                    orientation='all',
                                    verbose = False)
                               }
    
    dataset = transformed_datasets['val']

    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=8)
    
    # create empty df
    pred_df = pd.DataFrame(columns=["Image Index"])
    
    if val:
        true_df = pd.DataFrame(columns=["Image Index"])
    
    
    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data

        inputs = Variable(inputs.cuda())
        
        if inputs.dim() == 3:
            inputs.unsqueeze_(0)
            
        if val:
            true_labels = labels.cpu().data.numpy()
        
        batch_size = inputs.shape
        
        #print("batch_size:", batch_size)
        #print(inputs[0].shape)

        probs = model.run(inputs)
        
        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}            
            
            if val:
                truerow = {}
                truerow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
                thisrow["Image Index"] = dataset.df.index[BATCH_SIZE * i + j]
            
            else:
                thisrow["Study"] = '/'.join(dataset.df.index[BATCH_SIZE * i + j].split("/")[0:-1])
            
            for k in range(len(dataset.PRED_LABEL)):
                thisrow[dataset.PRED_LABEL[k]] = probs[j, k]
                if val:
                    truerow[dataset.PRED_LABEL[k]] = true_labels[j, k] 
                    
            pred_df = pred_df.append(thisrow, ignore_index=True)
            if val:
                true_df = true_df.append(truerow, ignore_index=True)
                
    # take the mean of predictions if images are from the same study
    pred_df = pred_df.groupby('Study', as_index=False).mean()
                
    if val:
            
        auc_df = pd.DataFrame(columns=["label", "auc"])

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
            thisrow['auc'] = np.nan
            try:
                thisrow['auc'] = sklm.roc_auc_score(
                    actual.as_matrix().astype(int), pred.as_matrix())
            except BaseException:
                if verbose:
                    print("can't calculate auc for " + str(column))
            auc_df = auc_df.append(thisrow, ignore_index=True)

        print(auc_df)
        
        auc = auc_df.as_matrix(columns=auc_df.columns[1:])
        last_val_acc = auc[~np.isnan(auc)].mean()
        
        print("mean_val_acc:", last_val_acc)

                    
    return pred_df, pred_df.to_csv(header=True, index=False)
            
if __name__ == "__main__":
    PATH_TO_CSV = sys.argv[1]
    PATH_TO_IMAGES = ""
    PATH_TO_OUTPUT = sys.argv[2]
    
    _, csv = run_chexpert(PATH_TO_IMAGES, PATH_TO_CSV, val=False)
        
    with open(PATH_TO_OUTPUT, 'w+') as f:
        f.write(csv)
    
