# TODO: test metrics
import torch
from pytest import approx

from src import torch_metrics as tm


class TestBalancedAcc:
    def test_one(self):
        a = tm.balanced_acc(
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([1, 2, 3, 0]),
        )
        assert a == 1.0

    def test_zero(self):
        a = tm.balanced_acc(
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([2, 1, 1, 3]),
        )
        assert a == 0.0

    def test_easy_already_balanced(self):
        a = tm.balanced_acc(
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 0]),
        )
        assert a == approx(2 / 3)
