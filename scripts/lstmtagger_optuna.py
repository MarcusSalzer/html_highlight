import json
import sys
from datetime import datetime
from pathlib import Path

import optuna
import torch
from torch import nn, optim

sys.path.append(".")
from src import torch_util, types, util


# load hyperparameters
param_fn = "model_params/lstm_small.json"
with Path(param_fn).open() as f:
    setup = types.ModelTrainSetup(**json.load(f))


now = datetime.now()
model_name = f"{Path(param_fn).name.split('.')[0]}_{now.month:02d}{now.day:02d}"

# load data, convert to dataframe
split_idx = util.load_split_idx(setup.split)
data = {
    sk: util.dataset_to_df(v) for sk, v in util.load_dataset_splits(split_idx).items()
}

# get a vocab
vocab, token2idx, tag_vocab, tag2idx = util.make_vocab(data["train"])

# Update and save metadata
setup.constructor.update(
    {
        "token_vocab_size": len(vocab),
        "label_vocab_size": len(tag_vocab),
    }
)


device = torch_util.get_dev()

# Training details

lossfn = nn.CrossEntropyLoss()


print(f"\nTraining {model_name} on {device}")


def objective(trial: optuna.Trial):
    # params
    bs = 2 ** trial.suggest_int("bs_exp", 1, 3)
    emb_dim = trial.suggest_int("emb_dim", 4, 32, step=4)
    hidden_dim = trial.suggest_int("hidden_dim", 16 * 8, 16 * 12, step=16)

    # model
    model = torch_util.LSTMTagger(
        len(vocab),
        len(tag_vocab),
        emb_dim,
        hidden_dim,
        n_lstm_layers=trial.suggest_int("n_lstm_layers", 1, 2),
        dropout_lstm=0.3,
        bidi=True,
    )  # type: ignore
    model.to(device=device)

    train_dl = torch_util.data2torch(
        data["train"], bs, token2idx, tag2idx, device, model.n_extra
    )
    val_dl = torch_util.data2torch(
        data["val"], 2 * bs, token2idx, tag2idx, device, model.n_extra
    )

    opt = optim.Adam(model.parameters(), setup.train.lr)
    metrics = torch_util.train_loop(
        model,
        train_dl,
        val_dl,
        500,
        opt,
        lossfn,
        name=model_name,
        printerval=None,
        reduce_lr_on_plat=setup.train.lrs_plat,
        stop_patience=10,
        # epoch_callback=lambda x: trial.report(),
    )
    val_acc = metrics["val_acc"]

    best_acc = torch.tensor(val_acc).max().item()
    return -best_acc


# optimize
study = optuna.create_study(
    storage="sqlite:///optuna/data.db",
    study_name="lstmtagger",
    load_if_exists=True,
    # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
)
study.optimize(objective, n_trials=10)

print("\n=== best params ===")
print(study.best_params)
