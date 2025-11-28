import json
import os
import sys
from pathlib import Path

import torch

from src import tagger_model, torch_util

sys.path.append(".")


class Inference:
    """Utility for performing model inference."""

    def __init__(self, model_name: str, model_dir=Path(".")):
        # load metadata
        conf_data = json.loads((model_dir / f"{model_name}_config.json").read_text())

        model_conf = tagger_model.RNNTaggerConfig(**conf_data["config"])

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = conf_data["vocab"]
        self.tag_vocab = conf_data["tag_vocab"]
        self.tag_map = conf_data.get("tag_map")  # Optional!

        self.token2idx = {t: i for i, t in enumerate(vocab)}
        self.tag2idx = {t: i for i, t in enumerate(self.tag_vocab)}

        # load model weights
        state_dict = torch.load(
            os.path.join(model_dir, f"{model_name}_state.pth"),
            weights_only=True,
            map_location=dev,
        )
        # Prepare model
        self.model = tagger_model.RNNTagger(model_conf, len(vocab), len(self.tag_vocab))
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

        self.model.eval()
        with torch.no_grad():
            tag_scores = self.model(token_tensor, tag_det_tensor)
        predictions = torch.argmax(tag_scores, dim=-1)

        tags = [self.tag_vocab[p] for p in predictions.ravel()]
        return tags
