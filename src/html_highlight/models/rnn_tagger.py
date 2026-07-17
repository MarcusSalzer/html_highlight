from typing import Any

import pydantic
import torch
from torch import nn
from torch.types import Tensor

from html_highlight.models.tagger_model import TaggerModel


class RNNTaggerConfig(pydantic.BaseModel):
    """All model specific parameters for the RNNTagger"""

    model_config = pydantic.ConfigDict(extra="forbid")  # dont allow extra trash

    # dimensions
    d_emb_token: int = 12
    d_emb_tag: int = 8
    d_emb_extra: int = 8
    d_hidden_rnn: int = 64
    # layers
    rnn_variant: str = "lstm"
    n_rnn_layers: int = 1
    mlp_sizes: list[int] | None = None
    bidi: bool = True
    dropout_rnn: float = 0.0
    dropout_between: float = 0.0
    dropout_mlp: float = 0.0
    # vocab
    n_unk_token: int = 1

    # optionally share weights between tag embedding and classifier head
    tag_weight_sharing: bool = False

    def model_post_init(self, context: Any) -> None:
        assert self.rnn_variant in {"rnn", "gru", "lstm"}, "unexpected RNN variant"
        if self.n_rnn_layers == 1:
            assert self.dropout_rnn == 0, "Cannot apply dropout with single RNN layer"

        if self.tag_weight_sharing:
            if self.mlp_sizes:
                assert self.d_emb_tag == self.mlp_sizes[-1], (
                    "weight sharing requires matching d_emb_tag and last MLP dim"
                )
            else:
                assert self.d_emb_tag == self.d_hidden_rnn * (2 if self.bidi else 1), (
                    "weight sharing requires matching d_emb_tag and RNN output"
                )


class RNNTaggerInferenceConfig(pydantic.BaseModel):
    """The needed data to recreate a model (other than weights)"""

    model_conf: RNNTaggerConfig
    vocab_token: list[str]
    vocab_tag: list[str]


class RNNTagger(TaggerModel):
    """A recurrent network for sequence tagging"""

    # Lookup for available RNN variants
    rnn_variants = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(
        self,
        conf: RNNTaggerConfig,
        vocab_sz_token: int,
        vocab_sz_tag: int,
        n_extra: int | None,
    ):
        super().__init__(vocab_sz_token, vocab_sz_tag)

        self.token_embedding = nn.Embedding(vocab_sz_token, conf.d_emb_token, padding_idx=0)
        self.tag_embedding = nn.Embedding(vocab_sz_tag, conf.d_emb_tag, padding_idx=0)
        self.proj_extra = nn.Linear(n_extra, conf.d_emb_extra) if n_extra else None

        # choose layer type for recurrent layers
        self.rnn = self.rnn_variants[conf.rnn_variant](
            conf.d_emb_token + conf.d_emb_tag,  # LSTM will receive tokens, tags stacked
            conf.d_hidden_rnn,
            conf.n_rnn_layers,
            batch_first=True,
            dropout=conf.dropout_rnn,
            bidirectional=conf.bidi,
        )

        self.dropout_between = (
            nn.Dropout(conf.dropout_between) if conf.dropout_between > 0 else nn.Identity()
        )

        # what dimension will the hidden state have? double if bidirectional
        d_hidden = conf.d_hidden_rnn * (2 if conf.bidi else 1)

        # Build FF layers if sizes given
        if conf.mlp_sizes:
            self.mlp = nn.Sequential()
            for sz in conf.mlp_sizes:
                self.mlp.append(nn.Linear(d_hidden, sz))
                self.mlp.append(nn.ReLU(inplace=True))
                if conf.dropout_mlp > 0:
                    self.mlp.append(nn.Dropout(conf.dropout_mlp))
                d_hidden = sz  # input size for next layer
        else:
            self.mlp = nn.Identity()

        # output
        self.tag_clf = nn.Linear(d_hidden, vocab_sz_tag, bias=False)

        # Weight sharing
        if conf.tag_weight_sharing:
            self.tag_clf.weight = self.tag_embedding.weight

    def __str__(self):
        return f"RNNTagger_{type(self.rnn).__name__}"

    def forward(self, tokens: Tensor, tags_det: Tensor, extra: Tensor | None = None) -> Tensor:
        bs, seq_len = tokens.shape[:2]

        # embed tokens and inital labels
        embeds_tokens = self.token_embedding(tokens)
        embeds_labels = self.tag_embedding(tags_det)

        if extra is None:
            # Cat -> (bs, seq_len, emb_token + emb_tag)
            embeds = torch.cat([embeds_tokens, embeds_labels], dim=-1)
        else:
            # optionally take extra features
            assert self.proj_extra is not None, "Needs extra feature embeddings"
            embeds_extra = self.proj_extra(extra)
            # Cat -> (bs, seq_len, emb_token + emb_tag + emb_extra)
            embeds = torch.cat([embeds_tokens, embeds_labels, embeds_extra], dim=-1)

        lstm_out, _ = self.rnn(embeds)  #  -> (bs, seq_len, actual_hidden)
        lstm_out = self.dropout_between(lstm_out)  # Dropout or Identity
        last_hidden = self.mlp(lstm_out)  # MLP or Identity
        # final clf layer
        logits = self.tag_clf(last_hidden)  # -> (bs, seq_len, tagset_size)

        # Sanity check
        assert logits.shape == (bs, seq_len, self.vocab_sz_tag)

        return logits
