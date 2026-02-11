import sys

sys.path.append(".")
from src import text_process
from src.torch_util import make_extra_feats


def test_shape():
    n_cases = len(text_process.WordCase)  # OH -> one column per case
    n_extra = n_cases + 1

    assert make_extra_feats(["abra", "cadabra", "Zim"]).shape == (3, n_extra)

    padded = make_extra_feats(["abra", "cadabra", "Zim"], padto=16)
    assert padded.shape == (16, n_extra)
    assert (padded[3:] ** 2).sum() == 0, "should be zero padded"
