import torch
import unittest
from models import SimpleCVAE


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleCVAE(16384, 5)

    def test_forward(self):
        x = torch.randn(64, 1, 16384)
        c = torch.randn(64, 16384)
        y = self.model(x, c)
        print("Model Output size:", y[0].size())

    def test_loss(self):
        x = torch.randn(64, 3, 64, 64)
        c = torch.randn(64, 5)
        result = self.model(x, labels = c)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()
