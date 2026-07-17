# %%

import sys
from importlib import reload
from pathlib import Path

import torch

import html_highlight.models.rnn_tagger
from html_highlight.models import tagger_model

sys.path.append("..")
from html_highlight.datamodels import dataset_record, split_index
from html_highlight.plots import plotly_plots as pp
from src import text_process, util, vocab
from src import torch_util as tu

# %% Load a subset of

reload(util)
reload(split_index)


split_idx = util.load_split_idx()

data = {
    sk: util.dataset_to_df(v)
    for sk, v in util.load_dataset_splits(
        split_idx.id_to_group,
        path=Path("../data/dataset.ndjson"),
        limit=100,
        filter_lang={"python"},
    ).items()
}
for k, v in data.items():
    print(k, len(v))

# get a vocab
vocs = vocab.both_vocabs(
    data["train"],  # Build vocabs from training data
    n_unknown_token=4,  # How many unknown tokens to track
)

device = tu.get_dev()

print(f"\n{len(vocs.token)=} | {len(vocs.tag)=} | {device=}\n")

dsets = {k: tu.SequenceDataset.from_dataframe(df, vocs, device=device) for k, df in data.items()}
for k, d in dsets.items():
    print(f"{k}: {d}")


# %% Model and training
reload(tagger_model)

torch.manual_seed(999)
conf = html_highlight.models.rnn_tagger.RNNTaggerConfig(
    d_emb_token=16,
    d_emb_tag=16,
    d_hidden_rnn=16,
    rnn_variant="rnn",
    n_rnn_layers=1,
    mlp_sizes=[32],
    bidi=True,
    dropout_rnn=0.0,
)

model = html_highlight.models.rnn_tagger.RNNTagger(
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


# %% Synthetic example
reload(dataset_record)

examples = [
    dataset_record.DatasetRecord(
        name="module",
        lang="python",
        tokens=["import", " ", "x", "\n", "print", "(", "x", ".", "y", ")"],
        tags=["kwim", "ws", "mo", "nl", "fnfr", "brop", "mo", "sy", "at", "brcl"],
    ),
    dataset_record.DatasetRecord(
        name="var",
        lang="python",
        tokens=["x", "=", "Thing", "(", ")", "\n", "print", "(", "x", ".", "y", ")"],
        tags=["v", "opas", "clco", "brop", "brcl", "nl", "fnfr", "brop", "va", "sy", "at", "brcl"],
    ),
]


processed = [text_process.process(e.to_string()) for e in examples]


model_out = [
    model(
        torch.tensor(vocs.token.encode(tokens), device=device).unsqueeze(0),
        torch.tensor(vocs.tag.encode(tags_det), device=device).unsqueeze(0),
    )
    .squeeze()
    .argmax(-1)
    for tokens, tags_det in processed
]
preds = [vocs.tag.decode(seq) for seq in model_out]

# concise print
for ex, pr in zip(examples, preds, strict=True):
    acc = sum(t == p for t, p in zip(ex.tags, pr)) / len(pr)
    print("---")
    print(f"{acc=:.1%}")
    print(ex.to_string())
    print(" ".join(t for t in pr if t not in {"ws", "nl"}))
