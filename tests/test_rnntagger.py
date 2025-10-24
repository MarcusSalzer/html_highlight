import torch
from pytest import mark

from src import torch_util as tu

VOCAB_TOKENS = 40
VOCAB_TAGS = 7


def test_init_default():
    conf = tu.RNNTaggerConfig(
        vocab_sz_token=VOCAB_TOKENS,
        vocab_sz_tag=VOCAB_TAGS,
    )

    model = tu.RNNTagger(conf)

    tokens = torch.tensor([[1, 2, 3], [3, 4, 0]])
    tags = torch.tensor([[1, 1, 4], [3, 5, 0]])

    out = model(tokens, tags)

    assert out.shape == (2, 3, VOCAB_TAGS)


def test_init_w_mlp():
    conf = tu.RNNTaggerConfig(
        vocab_sz_token=VOCAB_TOKENS,
        vocab_sz_tag=VOCAB_TAGS,
        mlp_sizes=[16, 16],
    )

    model = tu.RNNTagger(conf)

    tokens = torch.tensor([[1, 2, 3], [3, 4, 0]])
    tags = torch.tensor([[1, 1, 4], [3, 5, 0]])

    out = model(tokens, tags)

    assert model.mlp is not None, "should have MLP when specified"
    assert out.shape == (2, 3, VOCAB_TAGS)


def test_tot_weights_small():
    conf = tu.RNNTaggerConfig(
        vocab_sz_token=4,
        vocab_sz_tag=4,
        d_emb_token=2,
        d_emb_tag=2,
        d_hidden_rnn=2,
        rnn_variant="rnn",
        n_rnn_layers=1,
        bidi=False,
        mlp_sizes=None,
    )

    model = tu.RNNTagger(conf)
    assert model.tot_weights == 44
    assert model.mlp is None, "shouldnt have MLP unless specified"
