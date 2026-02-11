from src.vocab import CodeVocab


def test_code_vocab_fromwords():
    """Test initialization of CodeVocab."""
    tokens = set(["apple", "banana", "cherry"])
    vocab = CodeVocab.from_words(tokens, n_unknown=2)

    assert vocab.n_unknown == 2
    assert vocab.vocab_list == ["<pad>", "<unk0>", "<unk1>", "apple", "banana", "cherry"]
    assert vocab.token2idx == {
        "<pad>": 0,
        "<unk0>": 1,
        "<unk1>": 2,
        "apple": 3,
        "banana": 4,
        "cherry": 5,
    }
    assert len(vocab) == 6


def test_code_vocab_encode():
    """Test the encode method."""
    tokens = set(["apple", "banana", "cherry"])
    vocab = CodeVocab.from_words(tokens, n_unknown=2)

    encoded = vocab.encode(["apple", "banana", "unk", "apple", "grape", "unk"])
    assert encoded == [3, 4, 1, 3, 2, 1]


def test_code_vocab_decode():
    """Test the decode method."""
    tokens = set(["apple", "banana", "cherry"])
    vocab = CodeVocab.from_words(tokens, n_unknown=2)

    decoded = vocab.decode([3, 4, 2, 3, 3])
    assert decoded == ["apple", "banana", "<unk1>", "apple", "apple"]

    decoded = vocab.decode([3, 4, 1, 2])
    assert decoded == ["apple", "banana", "<unk0>", "<unk1>"]


def test_code_vocab_encode_with_unknown_wrapping():
    """Test encode when unknown tokens wrap around."""
    tokens = set(["apple", "banana"])
    vocab = CodeVocab.from_words(tokens, n_unknown=2)
    encoded = vocab.encode(["apple", "banana", "orange", "grape", "kiwi"])
    assert encoded == [3, 4, 1, 2, 1]

    encoded = vocab.encode(["a", "b", "c", "d"])
    assert encoded == [1, 2, 1, 2]


def test_code_vocab_string():
    """Test the __str__ method."""
    tokens = set(["apple", "banana", "cherry"])
    vocab = CodeVocab.from_words(tokens, n_unknown=2)
    expected_string = f"Vocab(n={len(vocab.vocab_list)}, n_unk={vocab.n_unknown})"
    assert str(vocab) == expected_string
