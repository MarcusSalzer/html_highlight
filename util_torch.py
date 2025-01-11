import torch


def seqs2padded_tensor(sequences: list[list[int | float]], pad_value=0, verbose=True):
    t = torch.nn.utils.rnn.pad_sequence(
        (torch.tensor(s) for s in sequences),
        batch_first=True,
        padding_value=pad_value,
    )
    if verbose:
        print("padded tensor:", tuple(t.size()))
    return t
