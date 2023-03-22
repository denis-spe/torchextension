# === Glory Be To God ====

# Import need libraries
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Callable, Optional


class DataGenerator(Dataset):
    """
    ===== Generator Data samples =====
    """

    def __init__(
            self,
            n_row: int,
            n_col: int,
            noise: Union[float, int] = 3,
            split_data: bool = False,
            train_size: int = 75,
            test_size: int = 25,
            transform: Optional[Callable] = None) -> None:
        """
        Generator Data samples

        :param n_row: Number of rows
        :param n_col: Number of columns
        :param noise: Add noise to the data
        :param transform: Add a Callable
        """
        super(DataGenerator, self).__init__()
        self.__n_row = n_row
        self.__n_col = n_col
        self.len = n_row
        self.X = torch.rand((self.__n_row, self.__n_col))
        self.y = noise * .1 + torch.randn((self.__n_row, 1))

        idx = np.arange(n_row)
        np.random.shuffle(idx)

        # Create index.
        train_idx = idx[:train_size]
        test_idx = idx[test_size:]

        self.transforms = transform
        self.train = split_data

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        if self.train:
            if self.transforms:
                transformed_x = self.transforms(self.X[index, :])
                return transformed_x, self.y[index]
            return self.X[index, :], self.y[index]

        else:
            if self.transforms:
                transformed_x = self.transforms(self.X[index, :])
                return transformed_x, self.y[index]
            else:
                return self.X[index, :], self.y[index]
    
    @property
    def dataset(self):
        x = torch.tensor([
            _x
            for _x, _ in self
        ])

        y = torch.tensor([
            _y
            for _, _y in self
        ])

        return x, y
