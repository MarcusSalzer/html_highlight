import json
import sys
from datetime import datetime
from pathlib import Path

import optuna
import torch
from torch import nn, optim

sys.path.append(".")
from src import torch_util, types, util, plotly_plots


ntrials = int(sys.argv[1])

MEDIA_DIR = Path("optuna/media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


# load hyperparameters
param_fn = "model_params/lstm_small.json"
with Path(param_fn).open() as f:
    setup = types.ModelTrainSetup(**json.load(f))


now = datetime.now()
model_name = f"{Path(param_fn).name.split('.')[0]}_{now.month:02d}{now.day:02d}"

# load data, convert to dataframe
split_idx, split_date = util.load_split_idx()
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


print(f"\nTraining {model_name} on {device}")


def epoch_callback(trial: optuna.Trial, value: float, step: int):
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned()


def objective(trial: optuna.Trial):
    # params
    bs = 2 * 2 ** trial.suggest_int("bs_exp", 0, 5)
    trial.set_user_attr("bs", bs)
    emb_dim = trial.suggest_int("emb_dim", 4, 32, step=4)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 192, step=16)
    lossfn = nn.CrossEntropyLoss(
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.3)
    )
    n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 3, log=True)

    if n_lstm_layers > 1:
        dropout = trial.suggest_float("dropout", 0.01, 0.5)
    else:
        dropout = 0.0

    start_lr = trial.suggest_float("start_lr", 1e-4, 1e-2)

    # model
    model = torch_util.LSTMTagger(
        len(vocab),
        len(tag_vocab),
        emb_dim,
        hidden_dim,
        n_lstm_layers=n_lstm_layers,
        dropout_lstm=dropout,
        bidi=True,
    )  # type: ignore
    model.to(device=device)

    train_dl = torch_util.data2torch(
        data["train"], bs, token2idx, tag2idx, device, model.n_extra
    )
    val_dl = torch_util.data2torch(
        data["val"], 2 * bs, token2idx, tag2idx, device, model.n_extra
    )

    trainer = torch_util.Trainer(
        model,
        train_dl,
        val_dl,
        optimizer=optim.Adam(model.parameters(), start_lr),
        loss_function=lossfn,
        name=model_name,
        printerval=None,
        reduce_lr_on_plat=setup.train.lrs_plat,
        stop_patience=10,
        epoch_callback=lambda x: epoch_callback(
            trial,
            value=-x["val_acc"],
            step=x["epoch"],
        ),
    )
    metrics = trainer.train_loop(max_epochs=1000)

    plot = plotly_plots.train_metrics_single_run(metrics)
    plot_file = MEDIA_DIR / f"{model_name}_{trial.number:03d}_metrics.png"
    plot.write_image(plot_file, width=1200, height=600)
    val_acc = metrics["val_acc"]

    best_acc = torch.tensor(val_acc).max().item()
    return -best_acc


# optimize
study = optuna.create_study(
    storage="sqlite:///optuna/data.db",
    study_name=f"lstmtagger_{split_date}",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
)

study.set_user_attr("n_data_train", len(data["train"]))
study.set_user_attr("n_data_val", len(data["val"]))

study.optimize(objective, n_trials=ntrials)

print("\n=== best params ===")
print(study.best_params)
