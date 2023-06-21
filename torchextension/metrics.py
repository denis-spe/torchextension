# === Glory Be To God ====

# import the necessary library
import torch
from torchextension.metricsInterface import Metric
from typing import Union
from torch import Tensor


class Accuracy(Metric):
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

        >>> # predicted values and true values
        >>> yhat = torch.tensor([[-0.221]])
        >>> y = torch.tensor([[0.05]])
        >>>
        >>> # Initialize the mean absolute error class.
        >>> accuracy = Accuracy()
        >>>
        >>> # Get the error.
        >>> assert accuracy(yhat, y) == 0
        >>>
        >>> yhat = torch.tensor([[1], [1], [1], [1]])
        >>> y = torch.tensor([[1], [0], [1], [0]])
        >>>
        >>> # print(accuracy(yhat, y))
        >>> # 8
    """
    name = "accuracy"

    def __call__(self, yhat: Tensor, y: Tensor) -> Union[int, float]:
        try:
            _, predict = torch.max(yhat, 1)
        except IndexError:
            predict = yhat
        return (predict == y).sum().item()


class MSE(Metric):
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

        >>> # predicted values and true values
        >>> yhat = torch.tensor([2.31, 1.432, 3.423, 24.2])
        >>> y = torch.tensor([2.43, 1.345, 2.98, 23.4])
        >>>
        >>> # Initialize the mean absolute error class.
        >>> mse = MSE()
        >>>
        >>> # Get the error.
        >>> # print(mse(yhat, y))
        >>> # 0.02287660539150238
    """
    name = "mse"

    def __call__(self, yhat: Tensor, y: Tensor) -> Union[int, float]:
        return (torch.mean(yhat - y) ** 2 / len(y)).item()


class MAE(Metric):
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

    >>> # predicted values and true values
    >>> yhat = torch.tensor([2.31, 1.432, 3.423, 24.2])
    >>> y = torch.tensor([2.43, 1.345, 2.98, 23.4])
    >>>
    >>> # Initialize the mean absolute error class.
    >>> mae = MAE()
    >>>
    >>> # Get the error.
    >>> # print(mae(yhat, y))
    >>> # 0.3625003397464752
    """
    name = "mae"

    def __call__(self, yhat: Tensor, y: Tensor) -> Union[int, float]:
        return torch.mean(torch.abs(yhat - y)).item()
