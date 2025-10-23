import json
import os
import sys

import torch
from src import torch_util

sys.path.append(".")


class Inference:
    def __init__(self, model_name: str, model_dir="."):
        # load meta
        with open(os.path.join(model_dir, f"{model_name}_meta.json")) as f:
            metadata = json.load(f)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = metadata["vocab"]
        self.tag_vocab = metadata["tag_vocab"]
        self.tag_map = metadata.get("tag_map")
        self.token2idx = {t: i for i, t in enumerate(vocab)}
        self.tag2idx = {t: i for i, t in enumerate(self.tag_vocab)}

        # load model weights
        state_dict = torch.load(
            os.path.join(model_dir, f"{model_name}_state.pth"),
            weights_only=True,
            map_location=dev,
        )

        self.model = torch_util.LSTMTagger(**metadata["setup"]["constructor"])
        self.model.load_state_dict(state_dict)

    def run(self, tokens: list[str], tags_det: list[str]) -> list[str]:
        """Run inference using model"""
        # optionally map tags
        if self.tag_map is not None:
            tags_det = [self.tag_map.get(t, t) for t in tags_det]

        token_idxs = [self.token2idx.get(t, 1) for t in tokens]
        tag_det_idxs = [self.tag2idx.get(t, 1) for t in tags_det]

        token_tensor = torch_util.seqs2padded_tensor([token_idxs], verbose=False)
        tag_det_tensor = torch_util.seqs2padded_tensor([tag_det_idxs], verbose=False)
        # extra features if needed

        extrafeats = (
            torch_util.make_extra_feats(tokens, padto=token_tensor.shape[1]).unsqueeze(
                0
            )
            if self.model.n_extra
            else None
        )
        # print("tokens", token_tensor.shape)
        # print("tags_det", tag_det_tensor.shape)
        # if extrafeats is not None:
        #     print("extra", extrafeats.shape)
        self.model.eval()
        with torch.no_grad():
            tag_scores = self.model(token_tensor, tag_det_tensor, extrafeats)
        predictions = torch.argmax(tag_scores, dim=-1)

        tags = [self.tag_vocab[p] for p in predictions.ravel()]
        return tags
