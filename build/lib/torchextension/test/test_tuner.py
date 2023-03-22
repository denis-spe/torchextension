# Import libraries ---
import unittest
import torch
from src.torchSeq.tuner import Hyperparameter, Tuner


class TestTuner(unittest.TestCase):

    def setUp(self) -> None:
        # Instantiate the hyperparameter class
        hyperparameter = Hyperparameter()
        self.hyperparameter = hyperparameter

    def test_hyperparameter_param(self):
        self.assertEqual(len(self.hyperparameter.param), 0)

        # Add some parameters in the param.
        self.hyperparameter.Choice('rate', options=[1e-1, 1e-2, 1e-3])

        # Is it still equal to zero.
        self.assertNotEqual(len(self.hyperparameter.param), 0)

        # Does the parameter output match the expected output.
        torch.testing.assert_close(
            self.hyperparameter.param,
            {'rate': torch.tensor([1e-1, 1e-2, 1e-3])}
        )

        # Can param method have many parameters?
        self.hyperparameter.Choice('rate', options=[1e-5, 1e-4, 1e-3])
        self.hyperparameter.Int('out_feature', min_value=32, max_value=128, step=32)
        torch.testing.assert_close(
            self.hyperparameter.param,
            {'rate': torch.tensor([1.0000e-05, 1.0000e-04, 1.0000e-03]),
             'out_feature': torch.tensor([32, 64, 96, 128])
             }
        )

    def test_hyperparameter_choice(self):
        # Add some parameter to param method.
        choice = self.hyperparameter.Choice('out_features', options=[32, 64, 128, 256])

        torch.testing.assert_close(
            self.hyperparameter.param,
            {'out_features': torch.tensor([32, 64, 128, 256])}
        )

    def test_hyperparameter_int(self):
        # Using int method to add in some parameter.
        self.hyperparameter.Int('out_feature', min_value=8, max_value=16)
        torch.testing.assert_close(
            self.hyperparameter.param,
            {'out_feature': torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 16])}
        )

        # Using int method to add in some parameter.
        self.hyperparameter.Int('out_feature', min_value=8, max_value=32, step=8)
        torch.testing.assert_close(
            self.hyperparameter.param,
            {'out_feature': torch.tensor([8, 16, 24, 32])}
        )

    def test_selector(self):
        self.hyperparameter.Int('out_feature', min_value=1, max_value=3)
        self.assertEqual(self.hyperparameter.selector('out_feature'), 1)
        self.assertEqual(self.hyperparameter.selector('out_feature'), 2)
        self.assertEqual(self.hyperparameter.selector('out_feature'), 3)


if __name__ == '__main__':
    unittest.main()
