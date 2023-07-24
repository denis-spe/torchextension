import unittest
import torch
from torchextension.metrics import Accuracy


class MyTestCase(unittest.TestCase):

    def test_accuracy(self):
        y_hat = torch.tensor([[1], [1], [1], [1]])
        y = torch.tensor([[1], [0], [1], [0]])

        accuracy = Accuracy()

        self.assertEqual(accuracy(y_hat, y), 0.5)
        self.assertEqual(type(accuracy(y_hat, y)).__name__, "float")


if __name__ == '__main__':
    unittest.main()
