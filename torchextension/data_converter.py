# *** Praise the Lord God ***

# Import libraries
from torch.utils.data import Dataset as _Dataset
from typing import Union as _Union
import numpy as _np
import torch as _t


class DataConverter(_Dataset):

    def __init__(self,
                 x: any,
                 y: _Union[_np.ndarray, None] = None,
                 transform=None,
                 target_transform=None) -> None:
        """
        Builds torch dataset by x and y
        :param x: 2d-dimension array
        :param x: 2d-dimension array
        """
        self.X = x
        self.y = y
        if y is not None:
            self.y = y.reshape(-1, 1)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        import torch

        x = torch.FloatTensor(self.X[idx, :])
        if self.transform:
            x = self.transform(x)

        y = torch.FloatTensor(self.y)[idx]
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def train_test_split(self, train_size: float = 0.75, test_size: float = 0.25, random_state: int = None):
        import torch
        from torch.utils.data import random_split

        if self.y is None:
            # Get the whole number instead of float
            # Since random_split uses whole number
            # to divide the data
            train_size = int(len(self.X) * train_size)
            test_size = int(len(self.X) * test_size)

            if random_state:
                generator = torch.Generator().manual_seed(random_state)
                return random_split(
                    self.X,
                    [train_size, test_size],
                    generator=generator
                )

            return random_split(
                self.X,
                [train_size, test_size]
            )

        else:
            # Get the whole number instead of float
            # Since random_split uses whole number
            # to divide the data
            train_size = int(len(self) * train_size)
            test_size = int(len(self) * test_size)

            if random_state:
                generator = torch.Generator().manual_seed(random_state)
                return random_split(
                    self,
                    [train_size, test_size],
                    generator=generator
                )

            return random_split(
                self,
                [train_size, test_size]
            )
