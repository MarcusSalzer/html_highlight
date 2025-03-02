import unittest

from src import torch_util
import torch


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


class TestLSTM(unittest.TestCase):
    def test_shape(self):
        model = torch_util.LSTMTagger(
            token_vocab_size=3,
            label_vocab_size=2,
            embedding_dim=4,
            hidden_dim=8,
            n_lstm_layers=2,
            dropout_lstm=0.1,
        )
        tokens_in = torch.tensor([0, 1, 2])
        tags_in = torch.tensor([0, 1, 1])
        out = model(tokens_in, tags_in)
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_shape_extra(self):
        model = torch_util.LSTMTagger(
            token_vocab_size=3,
            label_vocab_size=2,
            embedding_dim=4,
            hidden_dim=8,
            n_lstm_layers=2,
            dropout_lstm=0.1,
            n_extra_feats=2,
        )

        tokens_in = torch.tensor([0, 1, 2])
        tags_in = torch.tensor([0, 1, 1])
        extra_in = torch.randn((3, 2))  # (seqlen, n_extra)
        out = model(tokens_in, tags_in, extra_in)
        self.assertEqual(tuple(out.shape), (3, 2))
