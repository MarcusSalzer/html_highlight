import unittest

from src import torch_util


class TestSeqPad(unittest.TestCase):
    def test_simple(self):
        sequences = [
            [1, 3, 2, 2],
            [2, 2],
        ]
        tensor = torch_util.seqs2padded_tensor(sequences, 0, verbose=False)
        self.assertEqual(tuple(tensor.shape), (2, 4), "wrong shape")
        self.assertEqual(tuple(tensor[0, :]), (1, 3, 2, 2))
        self.assertEqual(tuple(tensor[1, :]), (2, 2, 0, 0))
