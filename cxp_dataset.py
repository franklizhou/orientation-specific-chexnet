import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class CXPDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            path_to_csv,
            fold,
            transform=None,
            sample=0,
            finding="any",
            starter_images=False,
            verbose = False):

        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_csv = path_to_csv
        
        if fold == "train":
            #print('cxp_dataset: ' + path_to_csv + 'train.csv')
            self.df = pd.read_csv(path_to_csv + 'train.csv')
        elif fold == "val":
            #print('cxp_dataset: ' + path_to_csv + 'valid.csv')
            self.df = pd.read_csv(path_to_csv + 'valid.csv')
        
        # blanks in dataframe assumed to be negative class    
        self.df.fillna(0, inplace=True)
        
        # make all uncertains true for now
        #print("Uncertain labels are own class")
        #self.df.replace(to_replace=-1, value=2, inplace=True)
        
        #print("Uncertain labels are positive")
        #self.df.replace(to_replace=-1, value=1, inplace=True)

        if verbose == True:
            print("Uncertain labels are negative")
        self.df.replace(to_replace=-1, value=0, inplace=True)
        
        if(starter_images):
            starter_images = pd.read_csv("starter_images.csv")
            self.df=pd.merge(left=self.df,right=starter_images, how="inner",on="Image Index")
            
        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    print("Found " + str(len(self.df[self.df[finding] == 1])) + " cases for " + finding)
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding + " as not in data - please check spelling")

        self.df = self.df.set_index("Path")
        self.PRED_LABEL = [
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
            'Support Devices']
        RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        return (image, label,self.df.index[idx])
