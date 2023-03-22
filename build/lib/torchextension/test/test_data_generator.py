import unittest
import torch
from src.torchSeq.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        # Set the seed to 42 ....
        torch.manual_seed(42)

        # Create a DataGenerator object .....
        data = DataGenerator(n_row=10, n_col=1)
        self.data = data

    def test_X(self) -> None:
        """
        Test X from DataGenerator
        :return: None
        """
        # Set the seed to 42 ....
        torch.manual_seed(42)

        # Create sample for testing ..
        X = torch.rand((10, 1))

        torch.testing.assert_allclose(self.data.X, X)
        torch.testing.assert_allclose(self.data.X.shape, torch.Size([10, 1]))

    def test_y(self) -> None:
        """
        Test y from DataGenerator
        :return: None
        """
        noise = 3

        # Set the seed to 42 ....
        torch.manual_seed(42)

        # Create sample for testing ..
        y = noise * .1 + torch.randn((10, 1))

        torch.testing.assert_allclose(self.data.y.shape, y.shape)


if __name__ == '__main__':
    unittest.main()
