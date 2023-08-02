# ** Glory Be To God **

# Import library
import sys

sys.path.append(sys.path[0].replace("tests", ""))

import unittest
import torch
from torch import nn
from sklearn.datasets import make_regression
from torchextension.torchmodel import Sequential
from torch.utils.data import random_split
from torchextension.metrics import MSE, MAE
from torch.utils.data import DataLoader
from torchextension.data_converter import DataConverter


class TestSequential(unittest.TestCase):

    def setUp(self):
        # Initialize the X and y.
        self.X, self.y = make_regression(n_samples=100, n_features=10)

        # Create a Dataset.
        self.dataset = DataConverter(self.X, self.y)

        # Instantiate  the Sequential model.
        self.model = Sequential([
            nn.Linear(10, 100),
            nn.Linear(100, 1)
        ])

    def test_train_process(self):
        self.model.compile(
            optimizer=torch.optim.Adam(params=self.model.parameters()),
            loss=nn.MSELoss(),
            metrics=[MSE(), MAE()]
        )

        self.model.fit(
            self.dataset,
            validation_split=0.25,
            epochs=3)

        history = self.model.fit(
            self.dataset,
            validation_data=[self.dataset],
            epochs=3)

        self.assertEqual(type(history.history).__name__, "dict")


if __name__ == '__main__':
    unittest.main()
