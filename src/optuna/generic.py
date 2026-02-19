import sys
from pathlib import Path

import src.models.rnn_tagger
from src.models import tagger_model

sys.path.append(".")

from src import util
from src.optuna.optuna_experiment import OptunaExperiment


class GenericExperiment(OptunaExperiment):
    """Investigate the impact of multiple unknown token embeddings."""

    def get_model_conf(self, trial):
        n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3, log=True)

        return src.models.rnn_tagger.RNNTaggerConfig(
            d_emb_token=trial.suggest_int("d_emb_token", 8, 64, step=4),
            d_emb_tag=trial.suggest_int("d_emb_tag", 8, 64, step=4),
            d_hidden_rnn=trial.suggest_int("d_hidden_rnn", 16, 192, step=16),
            rnn_variant=self.model_variant,
            n_rnn_layers=n_rnn_layers,
            mlp_sizes=[trial.suggest_int("mlp_layer_size", 16, 192, step=16)],
            dropout_rnn=trial.suggest_float("dropout_rnn", 0.0, 0.5) if n_rnn_layers > 1 else 0.0,
            n_unk_token=trial.suggest_int("n_unk_token", 1, 8),
        )

    def get_train_conf(self, trial) -> tagger_model.TrainSettings:
        return tagger_model.TrainSettings(
            start_lr=trial.suggest_float("start_lr", 1e-4, 1e-2),
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.9),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.9),
            bs_train=trial.suggest_int("bs_train", 4, 64, step=4),
        )


def main():
    if len(sys.argv) != 3:
        print("Usage: ... <n trials> <rnn/gru/lstm>")
        exit(1)

    ntrials = int(sys.argv[1])
    variant = sys.argv[2]

    assert variant in ("rnn", "gru", "lstm")

    split_idx = util.load_split_idx()
    print(f"Loaded split {split_idx.date}")
    data = {
        sk: util.dataset_to_df(v)
        for sk, v in util.load_dataset_splits(
            split_idx.id_to_group, path=Path("./data/dataset.ndjson")
        ).items()
    }

    # create and run experiment
    exp = GenericExperiment(data["train"], data["val"], variant, split_idx)
    print("Experiment:", exp)
    exp.optimize(ntrials)


if __name__ == "__main__":
    main()
