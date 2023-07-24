# === Glory Be To God ====

# Import libraries

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
from torch import Tensor


@dataclass
class MetricImpl(ABC):
    __name = ""

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __call__(self, y: Tensor, yhat: Tensor):
        pass


class Accuracy(MetricImpl):
    """
        Compute the model accuracy from yhat (predicted) and y (truth).

        parameter:
        ------------
        yhat: predicted values in torch tensor.
        y: true value in torch tensor.

        return:
        -------
        Model accuracy.

        example
        _______
        >>> import torch
        >>>
        >>> # predicted values and true values
        >>> y_hat = torch.tensor([[0]])
        >>> y = torch.tensor([[1]])
        >>>
        >>> # Initialize the mean absolute error class.
        >>> accuracy = Accuracy()
        >>>
        >>> # Get the error.
        >>> assert accuracy(y_hat, y) == 0.0
        >>>
        >>> y_hat = torch.tensor([[1], [1], [1], [1]])
        >>> y = torch.tensor([[1], [0], [1], [0]])
        >>>
        >>> assert accuracy(y_hat, y) == 0.5
        >>>
    """

    @property
    def name(self):
        return "accuracy"

    def __call__(self, y_hat: Tensor, y: Tensor) -> Union[int, float]:
        import torch
        from sklearn.metrics import accuracy_score

        try:
            _, predict = torch.max(y_hat, 1)
        except IndexError:

            predict = y_hat

        # Convert from tensor to numpy array
        numpy_y_hat = predict.numpy()
        numpy_y = y.numpy()

        return accuracy_score(numpy_y, numpy_y_hat)


class MSE(MetricImpl):
    """
        Compute the mean squared error between yhat (predicted) and y (truth).

        parameter:
        ------------
        yhat: predicted values in torch tensor.
        y: true value in torch tensor.

        return:
        -------
        int or float mean squared error

        example
        _______
        >>> import torch
        >>>
        >>> # predicted values and true values
        >>> y_hat = torch.tensor([2.31, 1.432, 3.423, 24.2])
        >>> y = torch.tensor([2.43, 1.345, 2.98, 23.4])
        >>>
        >>> # Initialize the mean absolute error class.
        >>> mse = MSE()
        >>>
        >>> # Get the error.
        >>> assert  mse(y_hat, y) == 0.02287660539150238
    """

    @property
    def name(self):
        return "mse"

    def __call__(self, y_hat: Tensor, y: Tensor) -> Union[int, float]:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y, y_hat)


class MAE(MetricImpl):
    """
    Compute the mean absolute error between yhat (predicted) and y (truth).

    parameter:
    ------------
    yhat: predicted values in torch tensor.
    y: true value in torch tensor.

    return:
    -------
    int or float mean absolute error

    example
    _______
    >>> import torch
    >>>
    >>> # predicted values and true values
    >>> y_hat = torch.tensor([2.31, 1.432, 3.423, 24.2])
    >>> y = torch.tensor([2.43, 1.345, 2.98, 23.4])
    >>>
    >>> # Initialize the mean absolute error class.
    >>> mae = MAE()
    >>>
    >>> # Get the error.
    >>> assert mae(y_hat, y) == 0.3625003397464752
    """

    @property
    def name(self):
        return "mae"

    def __call__(self, y_hat: Tensor, y: Tensor) -> Union[int, float]:
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y, y_hat)


class BinaryAccuracy(MetricImpl):
    """
        Compute the binary accuracy of yhat (predicted) and y (truth).

        parameter:
        ------------
        yhat: predicted values in torch tensor.
        y: true value in torch tensor.

        return:
        -------
        binary accuracy

        example
        _______
        >>> import torch
        >>>
        >>> # predicted values and true values
        >>> y_hat = torch.tensor([[0.4]])
        >>> y = torch.tensor([[1]])
        >>>
        >>> # Initialize the mean absolute error class.
        >>> binary_acc = BinaryAccuracy()
        >>>
        >>> # Get the error.
        >>> assert binary_acc(y_hat, y) == 0
        >>>
        >>>
        >>> y_hat = torch.tensor([[0.4]])
        >>> y = torch.tensor([[1]])
        >>>
        >>>
        >>> # Get the error.
        >>> assert binary_acc(y_hat, y) == 0
        """

    @property
    def name(self):
        return "binary_accuracy"

    def __call__(self, y_hat: Tensor, y: Tensor, threshold: float = 0.5) -> Union[int, float]:
        import torch
        predictions = torch.tensor([
            1 if value > threshold else 0
            for value in y_hat.view(-1, 1)
        ]).view(-1, 1)

        # Instantiate the Accuracy
        accuracy = Accuracy()

        return accuracy(predictions, y.view(-1, 1))
