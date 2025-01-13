import json
import sys

import torch
from src import models_torch

sys.path.append(".")

# load vocabs
with open("./models/lstmTagger_vocabs.json") as f:
    metadata = json.load(f)

# load model weights
state_dict = torch.load("./models/lstmTagger_state.pth", weights_only=True)


vocab = metadata["vocab"]
tag_vocab = metadata["tag_vocab"]
tag_map = metadata.get("tag_map")
token2idx = {t: i for i, t in enumerate(vocab)}
tag2idx = {t: i for i, t in enumerate(tag_vocab)}


def run(tokens: list[str], tags_det: list[str]) -> list[str]:
    # optionally map tags
    if tag_map is not None:
        tags_det = [tag_map.get(t, t) for t in tags_det]

    token_idxs = [token2idx.get(t, 1) for t in tokens]
    tag_det_idxs = [tag2idx.get(t, 1) for t in tags_det]

    token_tensors = models_torch.seqs2padded_tensor([token_idxs], verbose=False)
    tag_det_tensors = models_torch.seqs2padded_tensor([tag_det_idxs], verbose=False)

    model = models_torch.LSTMTagger(**metadata["constructor_params"])
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        tag_scores = model(token_tensors, tag_det_tensors)
    predictions = torch.argmax(tag_scores, dim=-1)

    tags = [tag_vocab[p] for p in predictions.ravel()]
    return tags
