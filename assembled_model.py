import torch
import cxp_dataset as CXP

class AssembledModel():

    def __init__(
            self,
            PATH_TO_AP,
            PATH_TO_PA,
            PATH_TO_LAT,
            PATH_TO_ORIENT,
            PATH_TO_IMAGES,
            PATH_TO_CSV):

        checkpoint_lat = torch.load(PATH_TO_LAT, map_location=lambda storage, loc: storage)
        checkpoint_pa = torch.load(PATH_TO_PA, map_location=lambda storage, loc: storage)
        checkpoint_ap = torch.load(PATH_TO_AP, map_location=lambda storage, loc: storage)
        checkpoint_orient = torch.load(PATH_TO_ORIENT, map_location=lambda storage, loc: storage)

        self.model_lat = checkpoint_lat['model']
        self.model_pa = checkpoint_pa['model']
        self.model_ap = checkpoint_ap['model']
        self.model_orient = checkpoint_orient['model']
        
        # put models on GPU
        self.model_lat.cuda()
        self.model_pa.cuda()
        self.model_ap.cuda()
        self.model_orient.cuda()
        
        self.model_lat.train(False)
        self.model_pa.train(False)
        self.model_ap.train(False)
        self.model_orient.train(False)
        
        self.ORIENTATION_DICT = {
            0: self.model_ap,
            1: self.model_pa,
            2: self.model_lat
        }
        
    def run(self, input):
        
        score_orient = self.model_orient(input)
        orientation = score_orient.argmax(dim=1)

        size = len(orientation)
        
        scores = torch.zeros(size, 5)
        
        score_ap = self.model_ap(input).cpu()
        score_pa = self.model_pa(input).cpu()
        score_lat = self.model_lat(input).cpu()
        
        scores[orientation==0, :] = score_ap[orientation==0]
        scores[orientation==1, :] = score_pa[orientation==1]
        scores[orientation==2, :] = score_lat[orientation==2]
        
        probs = scores.data.numpy()
        
        return probs
        



