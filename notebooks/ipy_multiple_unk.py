# %%

import sys
from importlib import reload
from pathlib import Path

import numpy as np
import torch

sys.path.append("..")
from src import plotly_plots as pp
from src import tagger_model, util, vocab
from src import torch_metrics as tm
from src import torch_util as tu
from src.datamodels import split_index

# %% Load a subset of

reload(util)
reload(split_index)


split_idx = util.load_split_idx()

data = {
    sk: util.dataset_to_df(v)
    for sk, v in util.load_dataset_splits(
        split_idx.id_to_group,
        path=Path("../data/dataset.ndjson"),
        filter_lang={"python"},
    ).items()
}
for k, v in data.items():
    print(k, len(v))

# get a vocab
vocs = vocab.both_vocabs(
    data["train"],  # Build vocabs from training data
    n_unknown_token=12,  # How many unknown tokens to track
)

device = tu.get_dev()

print(f"\n{len(vocs.token)=} | {len(vocs.tag)=} | {device=}\n")

dsets = {k: tu.SequenceDataset.from_dataframe(df, vocs, device=device) for k, df in data.items()}
for k, d in dsets.items():
    print(f"{k}: {d}")


# %% Model and training
reload(tagger_model)

torch.manual_seed(999)
conf = tagger_model.RNNTaggerConfig(
    d_emb_token=16,
    d_emb_tag=16,
    d_hidden_rnn=16,
    rnn_variant="rnn",
    n_rnn_layers=1,
    mlp_sizes=[32],
    bidi=True,
    dropout_rnn=0.0,
)

model = tagger_model.RNNTagger(
    conf,
    vocab_sz_token=len(vocs.token),
    vocab_sz_tag=len(vocs.tag),
    n_extra=0,
).to(device=device)


train_settings = tagger_model.TrainSettings(
    start_lr=5e-3,
    bs_train=16,
    plateau_lr_patience=20,
    stop_patience=50,
    max_epochs=500,
)
metrics = model.complete_train_loop(train_settings, dsets["train"], dsets["val"], verbose=False)
print(f"{metrics.keys()=}")
pp.train_metrics_single_run(metrics, acc_range=(0.3, 1.0))
