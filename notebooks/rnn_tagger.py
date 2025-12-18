# %%
import sys
from importlib import reload
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

sys.path.append("..")
from src import plotly_plots as pp
from src import tagger_model, util
from src import torch_metrics as tm
from src import torch_util as tu

# %%
reload(tu)
conf = tagger_model.RNNTaggerConfig(
    mlp_sizes=[80],
)

model = tagger_model.RNNTagger(
    conf,
    vocab_sz_token=10,
    vocab_sz_tag=7,
)

tokens = torch.tensor([[1, 2, 3], [3, 4, 0]])
tags_det = torch.tensor([[1, 1, 5], [3, 2, 0]])

out = model(tokens, tags_det)
print(model.tot_weights)
print(f"{out.shape=}")


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
vocab, token2idx, tag_vocab, tag2idx = util.make_vocab(data["train"])

device = tu.get_dev()

print(f"\n{len(vocab)=} | {len(tag_vocab)=} | {device=}\n")

dsets = {
    k: tu.SequenceDataset.from_dataframe(df, token2idx, tag2idx, device="cpu")
    for k, df in data.items()
}
for k, d in dsets.items():
    print(f"{k}: {d}")

# %% [markdown]
# ## class weight?
#

# %%


cls_sorted = data["train"]["tags"].explode().unique().sort()
clsk = compute_class_weight(
    "balanced",
    classes=np.array([tag2idx[t] for t in cls_sorted]),
    y=np.array([tag2idx[t] for t in data["train"]["tags"].explode()]),
)

class_weights = dict(zip(cls_sorted, clsk, strict=True))
clw = torch.tensor(
    [class_weights.get(tag, 1) for tag in tag_vocab],
    dtype=torch.float32,
).to(device)
# for tag, w in zip(tag_vocab, clw_sk):
#     print(f"{tag:8}  {w:.3f}")

# %% [markdown]
# ## Label distribution

# %%
labels_tr = data["train"]["tags"].explode()
print(f"total {len(labels_tr)} labels for training")
count_map = dict(zip(*np.unique_counts(labels_tr), strict=True))
# default to 0 for missing (pad etc)
distr_vec = np.array([count_map.get(t, 0) for t in tag_vocab], dtype=float)
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

torch.manual_seed(9)
conf = tagger_model.RNNTaggerConfig(
    d_emb_token=16,
    d_emb_tag=16,
    d_hidden_rnn=16,
    rnn_variant="rnn",
    n_rnn_layers=1,
    mlp_sizes=[64],
    bidi=True,
    dropout_rnn=0.0,
)

model = tagger_model.RNNTagger(
    conf,
    vocab_sz_token=len(vocab),
    vocab_sz_tag=len(tag_vocab),
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
    # loss_weights=clw,
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
