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

# %% [markdown]
# ## Data
#

# %%
# load data, convert to dataframe
reload(tu)
reload(util)
reload(tagger_model)

split_idx, split_date = util.load_split_idx()
print(f"Loaded split {split_date}")
data = {
    sk: util.dataset_to_df(v)
    for sk, v in util.load_dataset_splits(split_idx, path=Path("../data/dataset.ndjson")).items()
}
# get a vocab
vocs = vocab.both_vocabs(
    data["train"],  # Build vocabs from training data
    n_unknown_token=1,  # How many unknown tokens to track
)

device = tu.get_dev()

print(f"\n{len(vocs.token)=} | {len(vocs.tag)=} | {device=}\n")

dsets = {k: tu.SequenceDataset.from_dataframe(df, vocs, device="cpu") for k, df in data.items()}
for k, d in dsets.items():
    print(f"{k}: {d}")


# %% [markdown]
# ## Label distribution

# %%
labels_tr = data["train"]["tags"].explode()
print(f"total {len(labels_tr)} labels for training")
count_map = dict(zip(*np.unique_counts(labels_tr), strict=True))
# default to 0 for missing (pad etc)
distr_vec = np.array([count_map.get(t, 0) for t in vocs.tag], dtype=float)
distr_vec /= distr_vec.sum()  # normalize
pp.go.Figure(
    pp.go.Bar(x=list(count_map.keys()), y=list(count_map.values())),
    dict(title="Train label distribution"),
)
print(f"{distr_vec.shape}")

# %% [markdown]
# ## training
#

# %%
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
    n_extra=0,  # EXTRA FEATURES?
)
model.to(device=device)

# Manually set the initial CLF bias, seems to help a little
with torch.no_grad():
    model.tag_clf.bias.copy_(torch.tensor(distr_vec))

train_settings = tagger_model.TrainSettings(
    label_smoothing=0.1,
    weight_decay=0.1,
    start_lr=5e-3,
    bs_train=16,
    plateau_lr_patience=20,
    stop_patience=25,
    max_epochs=5,
)
metrics = model.complete_train_loop(train_settings, dsets["train"], dsets["val"], verbose=True)
print(f"{metrics.keys()=}")
pp.train_metrics_single_run(metrics, acc_range=(0.3, 1.0))

# %% [markdown]
# =============================================================

# %%
yt = torch.tensor([0, 0, 0, 1])
yp = torch.tensor([0, 0, 1, 1])

acc = (yt == yp).mean(dtype=torch.float32)


print(f"{acc=}")
print(f"{tm.balanced_acc(yp, yt)=}")
# print(f"{balanced_accuracy_score(yt,yp,adjusted=False)=}")


# %%
print(yt.reshape(-1, 1) == yt)
print((yt.reshape(-1, 1) == yt).sum(0))
