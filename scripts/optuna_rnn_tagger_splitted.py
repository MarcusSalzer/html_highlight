import sys

import optuna

sys.path.append(".")
from src import tagger_model, util
from src import torch_metrics as tm
from src import torch_util as tu

NTRIALS = int(sys.argv[1])
VARIANT = sys.argv[2]
assert VARIANT in ("rnn", "gru", "lstm")
OPT_METRIC = "acc"
MET_FACTOR = tm.METRIC_DIR[OPT_METRIC]

# load data, convert to dataframe
split_idx, split_date = util.load_split_idx()
data = {sk: util.dataset_to_df(v) for sk, v in util.load_dataset_splits(split_idx).items()}

model_key = f"tagger_{VARIANT}_split{split_date}_{OPT_METRIC}"


# get a vocab
vocab, token2idx, tag_vocab, tag2idx = util.make_vocab(data["train"])
device = tu.get_dev()
print(f"\n{len(vocab)=} | {len(tag_vocab)=} | {device=}\n")

dsets = {
    k: tu.SequenceDataset.from_dataframe(df, token2idx, tag2idx, device) for k, df in data.items()
}
print(f"\nTraining {model_key}")


def epoch_callback(trial: optuna.Trial, value: float, step: int):
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned()


def objective(trial: optuna.Trial):
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3, log=True)
    # params
    model_conf = tagger_model.RNNTaggerConfig(
        d_emb_token=trial.suggest_int("d_emb_token", 8, 64, step=4),
        d_emb_tag=trial.suggest_int("d_emb_tag", 8, 64, step=4),
        d_hidden_rnn=trial.suggest_int("d_hidden_rnn", 16, 192, step=16),
        rnn_variant=VARIANT,
        n_rnn_layers=n_rnn_layers,
        mlp_sizes=[trial.suggest_int("mlp_layer_size", 16, 192, step=16)],
        dropout_rnn=trial.suggest_float("dropout_rnn", 0.0, 0.5) if n_rnn_layers > 1 else 0.0,
    )
    train_conf = tagger_model.TrainSettings(
        start_lr=trial.suggest_float("start_lr", 1e-4, 1e-2),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.9),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.9),
        bs_train=trial.suggest_int("bs_train", 4, 64, step=4),
    )

    # model
    model = tagger_model.RNNTagger(
        model_conf,
        vocab_sz_token=len(vocab),
        vocab_sz_tag=len(tag_vocab),
    )
    model.to(device=device)

    metrics = model.complete_train_loop(
        train_conf,
        dsets["train"],
        dsets["val"],
        epoch_cb=lambda ep, mets: epoch_callback(
            trial,
            MET_FACTOR * mets[f"val_{OPT_METRIC}"],
            ep,
        ),
    )

    return max(MET_FACTOR * metrics[f"val_{OPT_METRIC}"])


# optimize
study = optuna.create_study(
    storage="sqlite:///data/optuna.db",
    study_name=model_key,
    load_if_exists=True,
    pruner=optuna.pruners.SuccessiveHalvingPruner(),
    direction="maximize",
)

study.set_user_attr("n_data_train", len(data["train"]))
study.set_user_attr("n_data_val", len(data["val"]))

study.optimize(objective, n_trials=NTRIALS)

print("\n=== best params ===")
print(study.best_params)
