"""Evaluate model using k-Fold cross validation on a (ideally large) train set.

We will do K-Fold CV inside the optuna objective
and optimize the mean metric value.

Limitation
----------
Perhaps KF is quite inefficient, and makes pruning (especially successive halving) difficult

"""

import sys

import optuna

sys.path.append(".")
from src import data_functions as datafun
from src import tagger_model, util
from src import torch_metrics as tm
from src import torch_util as tu

NTRIALS = int(sys.argv[1])
VARIANT = sys.argv[2]
assert VARIANT in ("rnn", "gru", "lstm")

K_FOLDS = 4
OVERLAP_NGRAM = 3
OPT_METRIC = "balanced_acc"
MET_FACTOR = tm.METRIC_DIR[OPT_METRIC]

# load data, convert to dataframe
split_idx, split_date = util.load_split_idx()
# Get training data only
data_tr = util.dataset_to_df(util.load_dataset_splits(split_idx)["train"])
study_key = f"tagger_{VARIANT}_split{split_date}_{OPT_METRIC}"


# get a vocab
vocab, token2idx, tag_vocab, tag2idx = util.make_vocab(data_tr)
device = tu.get_dev()
print(f"\n{len(vocab)=} | {len(tag_vocab)=} | {device=}\n")

# --- DATA ---
folds = datafun.simple_folds(data_tr, k=K_FOLDS, shuffle=True, seed=137)
overlaps = [
    datafun.overlap_split_pair(
        s_train["tokens"].to_list(),
        s_test["tokens"].to_list(),
        n=OVERLAP_NGRAM,
        norm="iou",
    )
    for s_train, s_test in folds
]
fold_dsets = [
    {
        "train": tu.SequenceDatasetOLD.from_dataframe(df_tr, token2idx, tag2idx, device="cpu"),
        "test": tu.SequenceDatasetOLD.from_dataframe(df_test, token2idx, tag2idx, device="cpu"),
    }
    for df_tr, df_test in folds
]

print(f"\nTraining {study_key}")


def suggest_params(trial: optuna.Trial):
    """Suggest parameters for the RNNTagger"""

    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 1)
    model_conf = tagger_model.RNNTaggerConfig(
        d_emb_token=trial.suggest_int("d_emb_token", 4, 64, step=4),
        d_emb_tag=trial.suggest_int("d_emb_tag", 4, 64, step=4),
        d_hidden_rnn=trial.suggest_int("d_hidden_rnn", 16, 192, step=16),
        rnn_variant=VARIANT,
        n_rnn_layers=n_rnn_layers,
        mlp_sizes=[trial.suggest_int("mlp_layer_size", 16, 192, step=16)],
        dropout_rnn=trial.suggest_float("dropout_rnn", 0.0, 0.5) if n_rnn_layers > 1 else 0.0,
        # dropout_between=
        # dropout_mlp=
    )
    train_conf = tagger_model.TrainSettings(
        start_lr=trial.suggest_float("start_lr", 1e-4, 1e-2),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.9),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.9),
        bs_train=trial.suggest_int("bs_train", 4, 64, step=4),
    )
    return model_conf, train_conf


def _epoch_callback(trial: optuna.Trial, value: float, step: int):
    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned()


def objective(trial: optuna.Trial):
    # --- PARAMS ---
    model_conf, train_conf = suggest_params(trial)

    # --- MODEL ---
    model = tagger_model.RNNTagger(
        model_conf,
        vocab_sz_token=len(vocab),
        vocab_sz_tag=len(tag_vocab),
    )
    model.to(device)

    # --- TRAINING (one run per fold) ----
    results = []
    # NOTE should we shuffle the order of the folds (see WilcoxonPruner docs)
    for fi, dsets in enumerate(fold_dsets):
        print(f"fold {fi}")
        metrics = model.complete_train_loop(
            train_conf,
            dsets["train"],
            dsets["test"],
        )
        fold_score = max(MET_FACTOR * metrics[f"val_{OPT_METRIC}"])

        # NOTE: no pruning inside of training loop, only reports once per fold
        trial.report(fold_score, fi)
        if trial.should_prune():
            # Return the current predicted value instead of raising `TrialPruned`.
            # This is a workaround to tell the Optuna about the evaluation
            # results in pruned trials.
            return sum(results) / len(results)

        results.append(fold_score)

    # mean result for all folds
    return sum(results) / len(results)


# optimize
study = optuna.create_study(
    storage="sqlite:///data/optuna.db",
    study_name=f"{study_key}_kfold",
    load_if_exists=True,
    pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1),
    direction="maximize",
)
study.set_user_attr(f"fold_overlaps_train_test_{OVERLAP_NGRAM}gram", overlaps)
study.set_user_attr("split_date", split_date)


study.optimize(objective, n_trials=NTRIALS)

print("\n=== best params ===")
print(study.best_params)
