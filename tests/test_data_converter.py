# *** Holy is the Lord God **

# Import library
import sys

sys.path.append(sys.path[0].replace("tests", ""))

import unittest
import torch
from torchextension.data_converter import DataConverter
from sklearn.datasets import make_classification, make_regression


class TestDataConverter(unittest.TestCase):

    def setUp(self):
        # Initialize the number of sample and number of feature
        n_samples = 100
        n_features = 10

        # Make classification sample data.
        self.class_X, self.class_y = make_classification(
            n_samples=n_samples,
            n_features=n_features
        )
        self.reg_X, self.reg_y = make_regression(
            n_samples=n_samples,
            n_features=n_features
        )

        # Instantiate the DataConverter.
        self.data_converter_class = DataConverter(
            x=self.class_X,
            y=self.class_y
        )

        self.data_converter_reg = DataConverter(
            x=self.reg_X,
            y=self.reg_y
        )

    def test_len(self):
        self.assertEqual(len(self.data_converter_class), 100)
        self.assertEqual(len(self.data_converter_reg), 100)

    def test_index(self):
        # Test classification
        self.assertEqual(round(self.data_converter_class[0][0][0].item()), round(self.class_X[0][0]))
        self.assertEqual(self.data_converter_class[0][1], self.class_y[0])

        # Test Regression
        self.assertEqual(round(self.data_converter_reg[0][0][0].item()), round(self.reg_X[0][0]))
        self.assertEqual(self.data_converter_reg[0][1], self.reg_y[0])

    def test_datatype(self):
        # Initialize the sample data from regression data converter class.
        reg_x = self.data_converter_reg[0][0]
        reg_y = self.data_converter_reg[0][1]

        self.assertTupleEqual(
            (type(reg_x), type(reg_y)),
            (torch.Tensor, torch.Tensor)
        )

    def test_train_test_split(self):
        # Get the length of samples
        length = len(self.data_converter_class.train_test_split())

        # Check for how many samples are return
        self.assertEqual(length, 2)

    def test_single_test_train_split(self):
        # Instantiate the DataConverter instance
        data = DataConverter(self.data_converter_class).train_test_split()

        self.assertEqual(type(data[0].dataset[0]).__name__, "tuple")
        self.assertEqual(len(data[0].dataset[0]), 2)


if __name__ == '__main__':
    unittest.main()
