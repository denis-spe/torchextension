# === Glory Be To God ====

# Import libraries
from abc import ABC
from dataclasses import dataclass
from typing import Union
from torch import Tensor


@dataclass
class Metric(ABC):
    name = ...

    def __call__(self, y: Tensor, yhat: Tensor) -> Union[int, float]:
        pass
