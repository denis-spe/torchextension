import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from torchextension.metrics import Metric


@dataclass
class History:
    def __init__(self, metrics: List[Metric] = None) -> None:
        self.metrics = metrics
        self.__loss = None
        self.__y = None
        self.__y_hat = None
        self.__valid_loss = None
        self.__valid_y = None
        self.__valid_y_hat = None
        self.__logs = {'loss': [], 'val_loss': []}

        if metrics:
            for metric in metrics:
                self.__logs[metric.name] = []
                self.__logs["val_" + metric.name] = []

    @property
    def logs(self) -> Dict[str, List]:
        return {
            key: value
            for key, value in self.__logs.items()
            if len(value) != 0
        }

    def add_loss_pred_and_y(self, loss, y_hat, y, train=True):
        if train:
            self.__loss = loss
            self.__y_hat = y_hat
            self.__y = y
        else:
            self.__valid_loss = loss
            self.__valid_y_hat = y_hat
            self.__valid_y = y

    def __add_loss_to_logs(self, train=True):
        if train:
            self.__logs["loss"].append(
                (self.__loss.sum() / len(self.__y)).item()
            )
        else:
            self.__logs["val_loss"].append(
                (self.__valid_loss.sum() / len(self.__valid_y)).item()
            )

    def __add_metric_score(self, metric, train=True):
        if train:
            self.__logs[metric.name].append(metric(
                self.__y_hat.view(1, -1),
                self.__y.view(1, -1)
            ))
        else:
            self.__logs["val_" + metric.name].append(metric(
                self.__valid_y_hat.view(1, -1),
                self.__valid_y.view(1, -1)
            ))

    def metrics_handler(self, train=True):
        self.__add_loss_to_logs(train=train)
        for metric in self.metrics:
            self.__add_metric_score(metric, train=train)

    @property
    def history(self):
        hist = {"loss": self.__logs["loss"]}
        if "val_loss" in self.__logs.keys():
            hist["val_loss"] = self.__logs["val_loss"]

            if self.metrics:
                hist.update({
                    "val_" + metric.name: self.__logs["val_" + metric.name]
                    for metric in self.metrics
                })
        if self.metrics:
            hist.update({
                metric.name: self.__logs[metric.name]
                for metric in self.metrics
            })

        return hist
