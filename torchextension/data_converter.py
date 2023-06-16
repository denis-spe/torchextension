# *** Praise the Lord God ***

# Import librarys
import torch
from torch.utils.data import Dataset


class DataConverter(Dataset):
    def __init__(self, X, y, transform = None, target_transform=None) ->None:
        self.X = X
        self.y = y.reshape(-1, 1)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int):
             X = torch.FloatTensor(self.X[idx, :])
             y = torch.FloatTensor(self.y)[idx]
             
             if self.transform:
                 X = self.transform(X)
             
             if self.target_transform:
                 y = self.target_transform(y)
             
             return X, y
             
    def train_test_split(self, trainsize: float=0.75, testsize: float=0.25, random_state: int = None):
        from torch.utils.data import random_split
        
        # Get the whole number instead of float
        # Since random_split uses whole number
        # to divide the data
        trainsize = int(len(self) * trainsize)
        testsize = int(len(self) * testsize)
        
        
        if random_state:
            generator = torch.Generator().manual_seed(random_state)
            return random_split(
                self,
                [trainsize, testsize],
                generator=generator
            )
        
        return random_split(
            self,
            [trainsize, testsize]
        )