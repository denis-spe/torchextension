# ** Glory Be To God **

# Import library
import sys
sys.path.append(sys.path[0].replace("tests", ""))

import unittest
import torch
from torch import nn
from sklearn.datasets import make_regression
from torchextension.torchmodel import Sequential
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchextension.metrics import MSE
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchextension.data_converter import DataConverter


class TestSequential(unittest.TestCase):
    
    def setUp(self):
        # Initialize the X and y.
        self.X, self.y = make_regression(n_samples=100, n_features=10)
        
        # Create a Dataset.
        self.dataset = DataConverter(self.X, self.y)
        
        # Instantiate  the Sequential model.
        self.model = Sequential([
                    nn.Linear(10, 100),
                    nn.Linear(100, 1)
                ])
   
    def test_compile(self):
         self.model.compile(
             optimize=torch.optim.Adam(params=self.model.parameters()), 
             loss=nn.MSELoss(), 
             metrics=MSE()
         )
         
         # Test the Optimize.
         self.assertIsInstance(
              self.model.optimizer,
              torch.optim.Adam
          )
          
         # Test the Loss.
         self.assertIsInstance(
              self.model.loss,
              nn.MSELoss
          )
      
    def test_fit_with_dataloader(self):
        # Compile the model
        self.model.compile(
             optimize=torch.optim.Adam(params=self.model.parameters()), 
             loss=nn.MSELoss(), 
             metrics=MSE()
         )
        
         # Load the data into a dataloader  
        train_dataloader = DataLoader(self.dataset)
        
        # Fit the data
        history = self.model.fit(train_dataloader, epochs=1)
        
        # Is instance of dict
        self.assertIsInstance(history, dict)
        
    def test_without_dataloader(self):
        # Compile the model
        self.model.compile(
             optimize=torch.optim.Adam(params=self.model.parameters()), 
             loss=nn.MSELoss(), 
             metrics=MSE()
         )
        
        # Fit the data
        history = self.model.fit(self.dataset, epochs=1,  batch_size=16)
        
        # Is instance of dict
        self.assertIsInstance(history, dict)

    def test_fit_X_y(self):
        # Compile the model
        self.model.compile(
             optimize=torch.optim.Adam(params=self.model.parameters()), 
             loss=nn.MSELoss(), 
             metrics=MSE()
         )
        
        # Fit the data
        history = self.model.fit(self.X, self.y, epochs=1,  batch_size=16)
        
        # Is instance of dict
        self.assertIsInstance(history, dict)
    
 
    def test_evaluate(self):
        # Compile the model
        self.model.compile(
             optimize=torch.optim.Adam(params=self.model.parameters()), 
             loss=nn.MSELoss(), 
             metrics=MSE()
         )
         
         # Load the data into a dataloader  
        train_dataloader = DataLoader(self.dataset)
        
        # Split the data into train and validation
        generator = torch.Generator().manual_seed(42)
        
        train_data, valid_data = random_split(
        dataset=train_dataloader, 
        lengths=[80, 20],
        generator=generator
        )
        
        #for X, y in train_data.dataset:
            #print(X[0])
        
        # Fit the data
        self.model.fit(train_data, epochs=1,  batch_size=16)
        
        # Evaluation of model
        evaluate = self.model.evaluate(valid_data)
        
        # Does evaluate returns a tuple.
        self.assertEqual(type(evaluate), tuple)


if __name__ == '__main__':
    unittest.main()
    
