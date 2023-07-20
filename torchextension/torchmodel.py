# pylint:disable=E1102
# === Glory Be To God ====


# import libraries
import sys
import time
from typing import Callable, Union, List
from torchextension.history import History

import numpy as np
import torch
from torch import nn as _nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from torchextension.data_converter import DataConverter
from torchextension.metrics import Metric

sys.path.append(sys.path[0].replace("tests", ""))


class Sequential(_nn.Module):
    """
    Pass list of torch nn layers.

    compile the model.

    Fit the model.

    Get the model prediction.

    parameter:
    --------
    layers: list of nn layer.

    for example:
    --------
    >>> from torch import nn
    >>> from torchextension.metrics import Accuracy
    >>> from torchextension.data_converter import DataConverter
    >>> from sklearn.datasets import make_classification
    >>>
    >>> samples = 200
    >>> classes = 2
    >>> features = 20
    >>>
    >>>
    >>> # Create binary data from sklearn make classification method
    >>> binary_X, binary_y = make_classification(
    ... n_samples=samples,
    ... n_classes=classes,
    ... n_features=features
    ... )
    >>>
    >>>
    >>> # Create demo data....
    >>> binary_data = DataConverter(binary_X, binary_y)
    >>>
    >>> # Split the data
    >>> binary_train_data, binary_valid_data  = binary_data.train_test_split()
    >>>
    >>> model = Sequential([
    ... # Input layer
    ... nn.Linear(in_features=features, out_features=32),
    ... nn.ReLU(),  # Activation function
    ... nn.Dropout(.4),
    ...
    ... # First hidden layer
    ... nn.Linear(in_features=32, out_features=1),
    ... nn.ReLU(),  # Activation function
    ... nn.Dropout(.4),  # Drop same pixel
    ...
    ... # Output layer
    ... nn.Sigmoid()
    ... ])
    >>>
    >>> # Compile the model .........
    >>> model.compile(
    ... optimizer=torch.optim.Adam(model.parameters()),
    ... loss=nn.BCELoss(),
    ... metrics=[Accuracy()],
    ... device=None
    ... )
    >>>
    >>>
    >>> # Fit the data .........
    >>> history = model.fit(
    ... binary_train_data,
    ... epochs=10,
    ... verbose=False,
    ... validation_data=binary_valid_data
    ... )
    >>> # print(model.predict(test_load))
    """

    def __init__(self, layers: List) -> None:
        super(Sequential, self).__init__()
        self.logs = None
        self.__metrics_method = None
        self.__model = None
        self.__loss = None
        self.__optimizer = None
        self.__device = None
        self.__layers = layers
        self.__stacked_layers: _nn.Sequential = _nn.Sequential(*self.__layers)
        self.__model_training: bool = True
        self.__history: History = History()

    def forward(self, x):
        return self.__stacked_layers(x)

    def compile(
            self,
            optimizer: any,
            loss: any,
            metrics: List[Metric],
            device: Union[str, None] = 'cpu'
    ) -> None:
        self.__device = device
        self.__optimizer = optimizer
        self.__loss = loss
        self.__model: Callable = Sequential(self.__layers).to(self.__device)
        self.__metrics_method = metrics
        self.__history = History(metrics=metrics)
        self.logs = self.__history.logs

    def __summaries(self, input_size=None):
        if input_size:
            return summary(self, input_size=input_size)
        else:
            return summary(self)

    def __train_process(
            self,
            x: Union[np.ndarray, Dataset, DataLoader],
            y: np.ndarray = None,
            **kwargs) -> History:
        """
        Train __model on train_data
        """

        # Validation of data if it's torch dataset or 
        # DataLoader
        data_loader = self.__data_validator(
            x=x,
            y=y,
            **kwargs
        )

        # Initialize the metric variable
        loss_list, y_list, y_hat_list = [], [], []

        # iterate over the data_loader
        for batch, (x, y) in enumerate(data_loader):

            # Switch to device
            x, y = x.to(self.__device), y.to(self.__device)

            if self.__model is None:
                raise TypeError('Compile the model before fitting it with `model.compile`')
            else:
                # Make prediction
                yhat = self.__model(x)

            # *** Backpropagation Process ***

            # Compute error by measure the degree of dissimilarity
            # from obtained result in target
            criterion = self.__loss(yhat, y)

            # Reset the gradient of the model parameters
            # Gradients by default add up; to prevent double-counting,
            # we explicitly zero them at each iteration.
            self.__optimizer.zero_grad()

            # Back propagate the prediction loss to deposit the gradient of loss
            # for learnable parameters
            criterion.backward()

            # Adjust the parameters by gradient collected in the backward pass
            self.__optimizer.step()

            # Append y, y_hat and loss to list
            y_list.append(y.item())
            y_hat_list.append(yhat.item())
            loss_list.append(criterion.item())

        # Add new loss, y_hat and y on
        self.__history.add_loss_pred_and_y(
            torch.tensor(loss_list),
            torch.tensor(y_hat_list),
            torch.tensor(y_list)
        )

        # Handle metrics
        self.__history.metrics_handler()

        return self.__history

    def __evaluate(self, x, y=None, **kwargs) -> History:
        """
        Evaluation  model with validation data
        """

        # Directing model to evaluation process
        self.__model.eval()

        # Validation of data if it's torch dataset or 
        # DataLoader
        data_loader = self.__data_validator(
            x=x,
            y=y,
            **kwargs
        )

        # Initialize the metric variable
        loss_list, y_list, y_hat_list = [], [], []

        # Disabling gradient calculation
        with torch.no_grad():
            for x, y in data_loader:
                # Set to device
                x, y = x.to(self.__device), y.to(self.__device)

                # Make prediction
                predictions = self.__model(x)

                # Compute the loss(error)
                criterion = self.__loss(predictions, y)

                # Add loss, y and predictions to empty lists
                loss_list.append(criterion.item())
                y_list.append(y)
                y_hat_list.append(predictions)

                # Add new loss, y_hat and y on
                self.__history.add_loss_pred_and_y(
                    torch.tensor(loss_list),
                    torch.tensor(y_hat_list),
                    torch.tensor(y_list),
                    train=False
                )

                # Handle metrics
                self.__history.metrics_handler(train=False)
                return self.__history

    @staticmethod
    def __data_validator(
            x,
            y=None,
            **kwargs):
        if y is not None:
            # Convert to torch Dataset
            dataset = DataConverter(x, y)
            dataset = DataLoader(dataset).dataset
            return dataset
        else:
            try:
                # It's  in torch Dataset
                dataset = x.dataset
            except AttributeError:
                dataset = DataLoader(x).dataset
            return dataset

    def fit(
            self,
            x: any,
            y: any = None,
            epochs: int = 1,
            validation_data: any = None,
            verbose: bool = True,
            callbacks: list = None,
            seed: int = 0,
            **kwargs
    ) -> History:
        """
        The Fit method make use of train_sample data and
        validation data if provided

        parameter
        ---------
        :param seed: for reproducibility.
        :param callbacks: (List) Pass a callback in list or None.
        :param verbose: (bool) Sequential training progress.
        :param validation_data: (DataLoader) Data to validate the model.
        :param epochs: (int) number of training iteration.
        :param x: (DataLoader) x data to train the model.
        :param y: (DataLoader) y data to train the model.

        :return: model's history.
        """

        # Set the reproducibility.
        torch.manual_seed(seed)

        # loop through the epoch
        for epoch in range(epochs):
            if verbose:
                print(f"\033[1m\nEpoch {epoch + 1}/{epochs}\033[0m")
                for _ in tqdm(range(100), ascii="•\\", bar_format='{l_bar}{bar:30}|', postfix=" "):
                    time.sleep(0.1)

            # Train the data return loss,
            train_metric = self.__train_process(x, y, **kwargs)

            if verbose:
                # Train logs
                train_logs = train_metric.logs.items()
                # Loop over the train logs
                for idx, (key, value) in enumerate(train_logs):
                    if not key.startswith("val"):
                        print("%s : %.4f" % (key, value[-1]), end="")
                        if idx != len(train_logs) - 1:
                            print(" - ", end="")

            if validation_data:
                valid_metrics = self.__evaluate(validation_data)

                if epoch == 0:
                    print(" - ", end="")

                if verbose:
                    # Train logs
                    valid_logs = valid_metrics.logs.items()

                    # Loop over the train logs
                    for idx, (key, value) in enumerate(valid_logs):
                        if key.startswith("val"):
                            print("%s : %.4f" % (key, value[-1]), end="")
                            if idx != len(valid_logs) - 1:
                                print(" - ", end="")

            if epoch == epochs - 1:
                print(end="\n")

            if not self.__model_training:
                # Break the training loop if model_training is false
                break

            if callbacks:
                for callback in callbacks:
                    callback(self, self.logs)

        return self.__history

    def predict(self, y: torch.tensor) -> torch.tensor:
        # list storage for predictions
        predictions = []

        # Indicate to evaluation process
        # self.__model.eval()

        # Don't use the gradient
        with torch.no_grad():
            # Instantiate the dataset
            data = y.dataset

            # Loop over the values in y
            for val in data:
                # switch to device
                val = val.to(self.__device)
                if self.__metrics_method.name in ['mae', 'mse']:
                    # Make prediction
                    predict = self.__model(val)

                    # Append the predictions to the list
                    predictions.append(predict.item())

                else:
                    # Make prediction
                    probability = self.__model(val)

                    # probability variable returns probability
                    # Therefor convert it to actual value
                    prediction = torch.argmax(probability, 1).item()

                    # Add prediction to predictions list
                    predictions.append(prediction)
        return predictions

#                                   Glory Be to Lord
