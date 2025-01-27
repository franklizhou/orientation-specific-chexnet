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
            uncertain=None,
            transform=None,
            sample=0,
            mask=None,
            finding="any",
            orientation="all",
            verbose = False):
        
        self.ORIENTATION_DICT = {
                'AP': 0,
                'LL': 0,
                'PA': 1,
                '0': 2}
       
        if uncertain not in [None, 'zeros', 'ones', 'self', 'multiclass']:
            print('Invalid uncertain strategy:', uncertain)
            with open("results/logger", 'a') as logfile:
                logfile.write('Invalid uncertain strategy: ' + uncertain + '\n')
            return
        
        if orientation not in ["all", "ap", "pa", "lat", 'trainer']:
            print('Invalid orientation strategy:', orientation)
            with open("results/logger", 'a') as logfile:
                logfile.write('Invalid orientation strategy: ' + orientation + '\n')
            return
        
        self.transform = transform
        self.path_to_images = path_to_images
        self.path_to_csv = path_to_csv
        self.fold = fold
        self.orientation = orientation
        
        if fold == "train":
            self.df = pd.read_csv(path_to_csv + 'train.csv')
        elif fold == "val":
            self.df = pd.read_csv(path_to_csv + 'valid.csv')
        elif fold == "traintest":
            self.df = pd.read_csv(path_to_csv + 'traintest.csv')
        elif fold == None:
            self.df = pd.read_csv(path_to_csv)
        
        # blanks in dataframe assumed to be negative class    
        self.df.fillna(0, inplace=True)

        if uncertain == None:
            if verbose == True:
                print("Using no uncertain labeling")
                with open("results/logger", 'a') as logfile:
                    logfile.write("Using no uncertain labeling\n")
        
        if uncertain == 'zeros':
            if verbose == True:
                print("Uncertain labels are negative")
                with open("results/logger", 'a') as logfile:
                    logfile.write("Uncertain labels are negative\n")
            self.df.replace(to_replace=-1, value=0, inplace=True)
            
        if uncertain == 'ones':
            if verbose == True:
                print("Uncertain labels are positive")
                with open("results/logger", 'a') as logfile:
                    logfile.write("Uncertain labels are positive\n")
            self.df.replace(to_replace=-1, value=1, inplace=True)
            
        if uncertain == 'multiclass':
            if verbose == True:
                print("Uncertain labels are own class")
                with open("results/logger", 'a') as logfile:
                    logfile.write("Uncertain labels are own class\n")
            self.df.replace(to_replace=-1, value=2, inplace=True)
            
        if orientation == 'lat':
            self.df = self.df[self.df['AP/PA'] == 0]
        elif orientation == 'ap':
            self.df = self.df[self.df['AP/PA'] == 'AP']
        elif orientation == 'pa':
            self.df = self.df[self.df['AP/PA'] == 'PA']
        
        if (sample > 0 and sample < len(self.df)):
            self.df = self.df.head(sample)
            
        if mask is not None:
            self.df = self.df[mask]
        
        if verbose and fold != None:
            print('AP')
            print((self.df[['AP/PA']] == 'AP').sum())

            print('PA')
            print((self.df[['AP/PA']] == 'PA').sum())

            print('0')
            print((self.df[['AP/PA']] == 0).sum())
        

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
        
        if orientation != 'trainer':
            self.PRED_LABEL = [
                'Cardiomegaly',
                'Edema',
                'Consolidation',
                'Atelectasis',
                'Pleural Effusion'
            ]
        else:
            self.PRED_LABEL = ['AP/PA']
        
        if fold != None:
            self.df.drop(['No Finding', 
                          'Enlarged Cardiomediastinum',
                          'Lung Opacity',
                          'Lung Lesion',
                          'Pneumonia',
                          'Pneumothorax',
                          'Pleural Other',
                          'Fracture', 
                          'Support Devices'], axis=1, inplace=True)
            
        #print(self.df)
        RESULT_PATH = "results/"


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.fold == None:
            return (image, 0, self.df.index[idx])

        if self.orientation != 'trainer':
            label = np.zeros(len(self.PRED_LABEL), dtype=int)
        else:
            label = np.zeros(3, dtype=int)
        
        
        if self.orientation != 'trainer':
            for i in range(0, len(self.PRED_LABEL)):
                # can leave zero if zero, else make one
                if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
        else:
            try:
                label = self.ORIENTATION_DICT[str(self.df[self.PRED_LABEL[0].strip()].iloc[idx])]
            except:
                print("LABEL FAILURE:", label)
                label = 0
                

        return (image, label, self.df.index[idx])
