import sys
sys.path.append(sys.path[0].replace("test", ""))

import unittest
import torch
from torch import nn
from torchextension.torchmodel import Sequential
from torchextension.metrics import MSE
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchextension.data_generator import DataGenerator

"""
class TestSequential(unittest.TestCase):
    
    def setUp(self):
        # Create a Dataset.
"""
dataset = DataGenerator(100, 1)


model = Sequential([
    nn.Linear(1, 100),
    nn.Linear(100, 1)
])

model.compile(
    optimize=torch.optim.Adam(params=model.parameters()), 
    loss=nn.MSELoss(), metrics=MSE())

model.fit(dataset, epochs=10)
# print(dataset.dataset)
    
