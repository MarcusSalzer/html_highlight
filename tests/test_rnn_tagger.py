import torch

from html_highlight.models.rnn_tagger import RNNTagger, RNNTaggerConfig

VOCAB_TOKENS = 40
VOCAB_TAGS = 7


def test_init_default():
    conf = RNNTaggerConfig()

    model = RNNTagger(
        conf,
        vocab_sz_token=VOCAB_TOKENS,
        vocab_sz_tag=VOCAB_TAGS,
        n_extra=None,
    )

    tokens = torch.tensor([[1, 2, 3], [3, 4, 0]])
    tags = torch.tensor([[1, 1, 4], [3, 5, 0]])

    out = model(tokens, tags, None)

    assert out.shape == (2, 3, VOCAB_TAGS)


def test_init_w_mlp():
    conf = RNNTaggerConfig(
        mlp_sizes=[16, 16],
    )

    model = RNNTagger(
        conf,
        vocab_sz_tag=VOCAB_TAGS,
        vocab_sz_token=VOCAB_TOKENS,
        n_extra=None,
    )

    tokens = torch.tensor([[1, 2, 3], [3, 4, 0]])
    tags = torch.tensor([[1, 1, 4], [3, 5, 0]])

    out = model(tokens, tags, None)

    assert isinstance(model.mlp, torch.nn.Sequential), "should have MLP when specified"
    assert out.shape == (2, 3, VOCAB_TAGS)


def test_tot_weights_small():
    conf = RNNTaggerConfig(
        d_emb_token=2,
        d_emb_tag=2,
        d_hidden_rnn=2,
        rnn_variant="rnn",
        n_rnn_layers=1,
        bidi=False,
        mlp_sizes=None,
    )

    model = RNNTagger(
        conf,
        vocab_sz_token=4,
        vocab_sz_tag=4,
        n_extra=None,
    )
    assert isinstance(model.mlp, torch.nn.Identity), "shouldnt have MLP unless specified"


def test_tot_weights_small_wextra():
    conf = RNNTaggerConfig(
        d_emb_token=2,
        d_emb_tag=2,
        d_emb_extra=4,
        d_hidden_rnn=2,
        rnn_variant="rnn",
        n_rnn_layers=1,
        bidi=False,
        mlp_sizes=None,
    )

    model = RNNTagger(
        conf,
        vocab_sz_token=4,
        vocab_sz_tag=4,
        n_extra=3,
    )
    assert isinstance(model.mlp, torch.nn.Identity), "shouldnt have MLP unless specified"


def test_no_batch_mixing():
    print("TODO")
