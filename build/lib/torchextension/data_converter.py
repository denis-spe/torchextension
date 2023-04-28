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