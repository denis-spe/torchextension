# ** Glory Be To God **

import sys
import unittest

import torch
from sklearn.datasets import make_classification
from torch import nn

sys.path.append(sys.path[0].replace("tests", ""))
from torchextension.torchmodel import Sequential
from torchextension.metrics import BinaryAccuracy
from torchextension.data_converter import DataConverter


class TestSequential(unittest.TestCase):

    def setUp(self):
        # Create some data with classification target label.

        # Initialize the number of features and samples
        self.cls_n_features = 20
        self.n_samples = 6000

        """
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
    """

    def test_multi_classification_model(self):
        # Instantiate X and y
        x, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.cls_n_features,
            n_classes=3,
            n_informative=7
        )

        cls_dataset = DataConverter(x, y=y)
        data_split = cls_dataset.train_test_split()
        cls_train_data = data_split[0]
        cls_valid_data = data_split[1]

        # Instantiate the Sequential model
        model = Sequential(
            layers=[
                nn.Linear(
                    in_features=self.cls_n_features,
                    out_features=512
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=512,
                    out_features=1024
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=1024,
                    out_features=252
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=252,
                    out_features=1
                ),
                nn.Sigmoid()
            ]
        )

        # Compile the model
        model.compile(
            optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
            loss=nn.BCELoss(),
            metrics=[BinaryAccuracy()]
        )

        # Fit the model.
        history1 = model.fit(cls_train_data, epochs=10)
        # history2 = model.fit(x, y, epochs=3)

    # def test_binary_model(self):
    #     # Instantiate X and y
    #     x, y = make_classification(
    #         n_samples=self.n_samples,
    #         n_features=self.cls_n_features,
    #         n_classes=2,
    #         n_informative=7
    #     )
    #
    #     cls_dataset = DataConverter(x, y=y)
    #     data_split = cls_dataset.train_test_split()
    #     cls_train_data = data_split[0]
    #     cls_valid_data = data_split[1]
    #
    #     # Instantiate the Sequential model
    #     model = Sequential(
    #         layers=[
    #             nn.Linear(
    #                 in_features=self.cls_n_features,
    #                 out_features=1
    #             ),
    #             nn.Sigmoid()
    #         ]
    #     )
    #
    #     # Compile the model
    #     model.compile(
    #         optimizer=torch.optim.Adam(model.parameters()),
    #         loss=nn.BCELoss(),
    #         metrics=Accuracy()
    #     )
    #
    #     # Fit the model.
    #     history1 = model.fit(cls_train_data)
    #     history2 = model.fit(x, y)
    #
    #     self.assertGreater(
    #         history1.get("accuracy")[0],
    #         0,
    #     )
    #     self.assertGreater(
    #         history2.get("accuracy")[0],
    #         0,
    #     )


if __name__ == '__main__':
    unittest.main()
