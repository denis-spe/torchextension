import sys
sys.path.append(sys.path[0].replace("tests", ""))

import unittest
import torch
from torch import nn
from torchmodel import Sequential
from metrics import Accuracy
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from data_generator import DataGenerator


class TestSequential(unittest.TestCase):

    def setUp(self):
        # Initialize train_sample and validation data ..........
        self.train = MNIST(root='.', train=True, download=True, transform=ToTensor())

        self.valid = MNIST(root='.', train=False, download=True, transform=ToTensor())

        # Couple the dataset into batch ...........
        self.train_dataloader = DataLoader(self.train, batch_size=64, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid, batch_size=64, shuffle=False)

        # Instantiate the Sequential object ...........
        self.mnist_model = Sequential([
            # Transpose Input data
            nn.Flatten(),

            # Input layer
            nn.Linear(in_features=28 * 28, out_features=256),
            nn.ReLU(),  # Activation function
            nn.Dropout(.2),

            # First hidden layer
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),  # Activation function
            nn.Dropout(.3),  # Drop same pixel

            # First hidden layer
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),  # Activation function
            nn.Dropout(.4),  # Drop same pixel

            # Output layer
            nn.Linear(in_features=512, out_features=10),
        ])

        # Compile the model ....
        self.mnist_model.compile(
            optimize=torch.optim.Adam(params=self.mnist_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            device="cpu",
            metrics=Accuracy()
        )

        # ------------ Use Convolutional layers -------------

        # Instantiate the Sequential object ...........
        self.mnist_conv_model = Sequential([
            nn.Conv2d(in_channels=1, out_channels=28*28, kernel_size=(2, 2)),

            # Transpose Input data
            nn.Flatten(),

            # Input layer
            nn.Linear(in_features=571536, out_features=256),
            nn.ReLU(),  # Activation function
            nn.Dropout(.2),

            # First hidden layer
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),  # Activation function
            nn.Dropout(.3),  # Drop same pixel

            # First hidden layer
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),  # Activation function
            nn.Dropout(.4),  # Drop same pixel

            # Output layer
            nn.Linear(in_features=512, out_features=10),
        ])

        # Compile the model ....
        self.mnist_conv_model.compile(
            optimize=torch.optim.Adam(params=self.mnist_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            device="cpu",
            metrics=Accuracy()
        )

    def test_fit(self):
        # Set the seed to 42 ....
        torch.manual_seed(42)

        # Fit the model.
        history = self.mnist_model.fit(
            self.train_dataloader
        )

        self.assertListEqual(list(history.keys()), ["accuracy", "loss"])
        self.failIf("loss" not in history.keys())

    def testMnistConv(self):
        # Set the seed to 42 ....
        torch.manual_seed(42)

        # Fit the model.
        history = self.mnist_conv_model.fit(
            self.train_dataloader
        )

        self.assertListEqual(list(history.keys()), ["accuracy", "loss"])
        self.failIf("loss" not in history.keys())


if __name__ == '__main__':
    unittest.main()
