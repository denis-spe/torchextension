#pylint:disable=E1102
# === Glory Be To God ====


# import libraries
import sys
sys.path.append(sys.path[0].replace("tests", ""))

import torch
import time
import numpy as np
from tqdm import tqdm
from typing import Iterator, Tuple, Dict, Callable, Union
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchextension.data_converter import DataConverter


class Sequential(nn.Module):
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
    >>> from src.torchSeq.metrics import MAE
    >>> from src.torchSeq.data_generator import DataGenerator
    >>>
    >>> # Create demo data....
    >>> train_data = DataGenerator(100, 8)
    >>> test_data = DataGenerator(100, 8, train_sample=False)
    >>>
    >>> model = Sequential([
    ... # Input layer
    ... nn.Linear(in_features=8, out_features=32),
    ... nn.ReLU(),  # Activation function
    ... nn.Dropout(.4),
    ...
    ... # First hidden layer
    ... nn.Linear(in_features=32, out_features=32),
    ... nn.ReLU(),  # Activation function
    ... nn.Dropout(.4),  # Drop same pixel
    ...
    ... # Output layer
    ... nn.Linear(in_features=32, out_features=1)
    ... ])
    >>>
    >>> # Compile the model .........
    >>> model.compile(
    ... optimize=torch.__optimizer.Adam(model.parameters()),
    ... loss=nn.MSELoss(),
    ... metrics=MAE(),
    ... device=None
    ... )
    >>>
    >>> train_dataloader = DataLoader(dataset=train_data, shuffle=True)
    >>> valid_dataloader = DataLoader(dataset=test_data, shuffle=False)
    >>>
    >>> # Fit the data .........
    >>> history = model.fit(
    ... train_dataloader,
    ... epochs=10,
    ... verbose=False,
    ... validation_data=valid_dataloader
    ... )
    >>> # print(model.predict(test_load))
    """

    def __init__(self, layers: Iterator) -> None:
        self.metrics_method = None
        self.model = None
        self.loss = None
        self.optim = None
        self.device = None
        self.__layers = layers
        super(Sequential, self).__init__()
        self.__stacked_layers = nn.Sequential(*self.__layers)
        self.model_training = True

    def forward(self, x):
        return self.__stacked_layers(x)

    def compile(self, optimize: any, loss: any, metrics, device: Union[str, None] = 'cpu') -> None:
        self.device = device
        self.optim = optimize
        self.loss = loss
        self.model: Callable = Sequential(self.__layers).to(self.device)
        self.metrics_method = metrics

    def summaries(self, input_size=None):
        if input_size:
            return summary(self, input_size=input_size)
        else:
            return summary(self)

    def train_process(
        self, 
        X: Union[np.ndarray, Dataset, DataLoader], 
        y: np.ndarray = None, 
        metric: Callable = None,  
        **kwargs) -> Tuple[float, float, float, int]:
        """
        Train __model on train_data
        """
        # Indicating the model to training
        self.model.train()
        
        
       
        # Validation of data if it's torch dataset or 
        # DataLoader
        data_loader  = self.__data_validator(
                     X=X,
                     y=y,
                     **kwargs
                     )
        
        if isinstance(data_loader, DataLoader):
            # Number of images in  data_loader: size
            size = len(data_loader.dataset)
        else:
            size = len(data_loader)

        # Initialize the metric variable
        total_score, total_loss, count_label, current = 0, 0, 0, 0

        start = time.time()
 
        
        # iterate over the data_loader
        for batch, (x, y) in enumerate(data_loader):
            
            # Switch to device
            x, y = x.to(self.device), y.to(self.device)
            
            if self.model is None:
                raise TypeError('Compile the model before fitting it with `model.compile`')
            else:
                # Make prediction
                yhat = self.model(x)
            

            # *** Backpropagation Process ***

            # Compute error by measure the degree of dissimilarity
            # from obtained result in target
            criterion = self.loss(yhat, y)

            # Reset the gradient of the model parameters
            # Gradients by default add up; to prevent double-counting,
            # we explicitly zero them at each iteration.
            self.optim.zero_grad()

            # Back propagate the prediction loss to deposit the gradient of loss
            # for learnable parameters
            criterion.backward()

            # Adjust the parameters by gradient collected in the backward pass
            self.optim.step()

            # Count number of labels
            count_label += len(y)

            # sum each loss to total_loss variable
            total_loss += criterion.item()

            # _, predict = torch.max(yhat, 1)

            # Add every accuracy on total_acc
            # total_score += (predict == y).sum().item()
            if metric:
                total_score += metric(yhat, y)

            if batch % 100 == 0:
                current += (batch / size)

        stop = time.time()
        time_taken = round(stop - start, 3)

        return (total_loss / count_label,
                total_score / count_label, time_taken,
                int(round(current * 100))
                )

    def evaluate(self, X, y=None, **kwargs) -> Tuple[float, float]:
        """
        Evaluation  model with validation data
        """
        
        # Directing model to evaluation process
        self.model.eval()
        
        # Validation of data if it's torch dataset or 
        # DataLoader
        data_loader  = self.__data_validator(
                     X=X,
                     y=y,
                     **kwargs
                     )

        # Instantiate metric variables
        total_loss, total_score, count_labels = 0, 0, 0

        # Disabling gradient calculation
        with torch.no_grad():
            
            for X, y in data_loader.dataset:
                
                # Set to device
                X, y = X.to(self.device), y.to(self.device)

                # Make prediction
                predictions = self.model(X)

                # Compute the loss(error)
                criterion = self.loss(predictions, y)

                # Add number of label to count_labels
                count_labels += len(y)

                # Add criterion loss to total_loss
                total_loss += criterion.item()
                
                # Calculate the metrics
                if self.metrics_method is not None:
                    total_score  += self.metrics_method(predictions, y)

            # Finally, return total_loss and total_acc which each is divided by
            # count_labels
            return total_loss / count_labels, total_score
            
    def __data_validator(
    self, 
    X, 
    y=None, 
    **kwargs):
       if y is not None:
            # Convert to torch Dataset 
            data_loader = DataConverter(X, y)
       else:
            # It's  in torch Dataset
            data_loader = X
        
        # Check if data_loader is instance of DataLoader   
       if not isinstance(data_loader, DataLoader):
           data_loader = DataLoader(data_loader, **kwargs)
           try:
               data_loader = data_loader.dataset.dataset
           except AttributeError:
               data_loader = data_loader.dataset
               
       return data_loader

    def fit(
            self,
            X: any,
            y: any=None,
            epochs: int = 1,
            validation_data: DataLoader = None,
            verbose: bool = True,
            callbacks: list = None,
            seed: int = 0,
            **kwargs
    ):
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
        :param train_data: (DataLoader) Data to train_sample the model.

        :return: model's history.
        """
        # Initializing variable for storing metric score
        metrics = {}
        score_list = []
        loss_list = []
        valid_acc_list = []
        valid_loss_list = []

        # Set the reproducibility.
        torch.manual_seed(seed)

        # loop through the epoch
        for epoch in range(epochs):
            if verbose:
                print(f"\033[1m\nEpoch {epoch + 1}/{epochs}\033[0m")
                for _ in tqdm(range(100), ascii="•\\", bar_format='{l_bar}{bar:30}|'):
                    time.sleep(0.1)

            # Train the data                
            train = self.train_process(X, y, metric=self.metrics_method, **kwargs)

            # Instantiate train_sample loss and accuracy
            train_loss = round(train[0], 6)
            train_score = round(train[1], 5)

            if verbose:
                print(
                    f" loss: {train_loss} - {self.metrics_method.name}: {train_score} - \
ETA: {round(train[2] * 100, 2)}ms",
                    end="")

            # Storing the __model score
            score_list.append(train_score)
            loss_list.append(train_loss)

            if validation_data:
                valid = self.evaluate(validation_data)
                # Instantiate train_sample loss and accuracy
                valid_loss = round(valid[0], 6)
                valid_acc = round(valid[1], 4)

                if verbose:
                    print(f"- val_loss: {valid_loss} - val_{self.metrics_method.name}: {valid_acc}")

                # Store the score
                valid_loss_list.append(valid_loss)
                valid_acc_list.append(valid_acc)

            if not self.model_training:
                # Break the training loop if model_training is false
                break

            if callbacks:
                for callback in callbacks:
                    callback(self, metrics)

        metrics[self.metrics_method.name] = score_list
        metrics["loss"] = loss_list

        if validation_data:
            metrics["val_" + self.metrics_method.name] = valid_acc_list
            metrics["val_loss"] = valid_loss_list

        return metrics

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
                val = val.to(self.device)
                if self.metrics_method.name in ['mae', 'mse']:
                    # Make prediction
                    predict = self.model(val)

                    # Append the predictions to the list
                    predictions.append(predict.item())

                else:
                    # Make prediction
                    probability = self.model(val)

                    # probability variable returns probability
                    # Therefor convert it to actual value
                    prediction = torch.argmax(probability, 1).item()

                    # Add prediction to predictions list
                    predictions.append(prediction)
        return predictions

#                                   Glory Be to God
