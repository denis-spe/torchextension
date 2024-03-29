# ** Glory Be To God **

import sys
import unittest

import torch
from sklearn.datasets import make_classification, make_regression
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

sys.path.append(sys.path[0].replace("tests", ""))
from torchextension.torchmodel import Sequential
from torchextension.metrics import BinaryAccuracy, Accuracy, MSE, MAE
from torchextension.data_converter import DataConverter


class TestSequential(unittest.TestCase):

    def setUp(self):
        # Create some data with classification target label.

        # Initialize the number of features and samples
        self.cls_n_features = 20
        self.n_samples = 6000

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
            optimizer=torch.optim.Adam(params=self.mnist_model.parameters()),
            loss=nn.CrossEntropyLoss(),
            device="cpu",
            metrics=[Accuracy()]
        )

        """

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
        
    """
    def test_fit(self):
        # Set the seed to 42 ....
        torch.manual_seed(42)

        # Fit the model.
        history = self.mnist_model.fit(
            self.train_dataloader
        )

    """
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
                nn.ReLU(),
                nn.Linear(
                    in_features=512,
                    out_features=3
                ),
                nn.Softmax(dim=0)
            ]
        )

        # Compile the model
        model.compile(
            optimizer=torch.optim.Adam(model.parameters()),
            loss=nn.CrossEntropyLoss(),
            metrics=[Accuracy()]
        )

        # Fit the model.
        # model.fit(
        #     cls_dataset,
        #     epochs=1,
        #     validation_split=0.25
        # )

        # history = model.fit(
        #     cls_train_data,
        #     validation_data=[cls_valid_data],
        #     epochs=2
        # )

        # print(model.optimizer.state_dict())

        # print(history.history)

        # model.fit(
        #     x,
        #     y,
        #     epochs=1,
        #     validation_split=0.25
        # )

    def test_binary_model(self):
        # Instantiate X and y
        x, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.cls_n_features,
            n_classes=2,
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
                    out_features=1
                ),
                nn.Sigmoid()
            ]
        )

        # Compile the model
        model.compile(
            optimizer=torch.optim.Adam(model.parameters()),
            loss=nn.BCELoss(),
            metrics=[BinaryAccuracy()]
        )

        # Fit the model.
        # history1 = model.fit(
        #     cls_train_data,
        #     validation_data=[cls_valid_data]
        # )
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

    def test_linear_model(self):
        # Instantiate X and y
        x, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.cls_n_features,
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
                    out_features=1
                ),
            ]
        )

        # Compile the model
        model.compile(
            optimizer=torch.optim.Adam(model.parameters()),
            loss=nn.MSELoss(),
            metrics=[MSE(), MAE()]
        )

        # Fit the model.
        history1 = model.fit(
            cls_train_data,
            validation_data=[cls_valid_data]
        )
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
