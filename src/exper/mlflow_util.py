from pathlib import Path

import mlflow
import pandas as pd
import torch

import src.models.rnn_tagger
from src import vocab
from src.models import rnn_tagger
from src.models.rnn_tagger import RNNTagger


class RNNTaggerPyFunc(mlflow.pyfunc.PyFuncModel):
    """Wrap the RNNTagger"""

    def load_context(self, context):
        from src.models.rnn_tagger import RNNTaggerInferenceConfig

        config = RNNTaggerInferenceConfig.model_validate_json(
            Path(context.artifacts["config"]).read_text()
        )
        self.model = RNNTagger(
            config.model_conf,
            len(config.vocab_token),
            len(config.vocab_tag),
            n_extra=None,
        )
        state_dict = torch.load(context.artifacts["weights"])
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Init vocabs
        self.vocs = vocab.VocabDuo(
            vocab.CodeVocab(config.vocab_token),
            vocab.CodeVocab(config.vocab_tag),
        )


def predict(self, context, model_input: pd.DataFrame):
    # expect raw text input sequences
    texts = model_input["text"].tolist()

    # tokenize and preprocess.

    # encode to tensors

    # run model

    # convert back to list[str]


def complete_config(vocs: vocab.VocabDuo, model_conf: src.models.rnn_tagger.RNNTaggerConfig):
    """What data (other than weights) is needed to recreate a model."""
    return rnn_tagger.RNNTaggerInferenceConfig(
        model_conf=model_conf,
        vocab_token=vocs.token.vocab_list,
        vocab_tag=vocs.tag.vocab_list,
    )
