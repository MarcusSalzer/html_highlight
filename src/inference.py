import json
import sys

import torch
from src import torch_util

sys.path.append(".")


class Inference:
    def __init__(self, model_name: str):
        # load meta
        with open(f"./model_inference/{model_name}_meta.json") as f:
            metadata = json.load(f)

        self.vocab = metadata["vocab"]
        self.tag_vocab = metadata["tag_vocab"]
        self.tag_map = metadata.get("tag_map")
        self.token2idx = {t: i for i, t in enumerate(self.vocab)}
        self.tag2idx = {t: i for i, t in enumerate(self.tag_vocab)}

        # load model weights
        state_dict = torch.load(f"./models/{model_name}_state.pth", weights_only=True)
        self.model = torch_util.LSTMTagger(**metadata["constructor_params"])
        self.model.load_state_dict(state_dict)

    def run(self, tokens: list[str], tags_det: list[str]) -> list[str]:
        """Run inference using model"""
        # optionally map tags
        if self.tag_map is not None:
            tags_det = [self.tag_map.get(t, t) for t in tags_det]

        token_idxs = [self.token2idx.get(t, 1) for t in tokens]
        tag_det_idxs = [self.tag2idx.get(t, 1) for t in tags_det]

        token_tensors = torch_util.seqs2padded_tensor([token_idxs], verbose=False)
        tag_det_tensors = torch_util.seqs2padded_tensor([tag_det_idxs], verbose=False)

        self.model.eval()
        with torch.no_grad():
            tag_scores = self.model(token_tensors, tag_det_tensors)
        predictions = torch.argmax(tag_scores, dim=-1)

        tags = [self.tag_vocab[p] for p in predictions.ravel()]
        return tags
