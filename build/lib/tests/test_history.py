import torch
import unittest
from torchextension.history import History
from torchextension.metrics import MAE, MSE


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # Instantiate History instance
        self.history = History(metrics=[MAE(), MSE()])
        self.history.add_loss_pred_and_y(
            torch.tensor([[2.134, 0.23]]),
            y_hat=torch.tensor([[2.134], [32.12]]),
            y=torch.tensor([[2.134], [15.21]])
        )
        self.history.metrics_handler()
        self.history.metrics_handler()

    def test_history(self):
        print(self.history.history)
        self.assertNotEqual(len(self.history.history), 0)



if __name__ == '__main__':
    unittest.main()
