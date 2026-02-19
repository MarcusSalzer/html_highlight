import sys
from pathlib import Path

from optuna.pruners._base import BasePruner

import optuna
import src.models.rnn_tagger
from src.models import tagger_model

sys.path.append(".")

from src import util
from src.optuna.optuna_experiment import OptunaExperiment


class MultipleUnkExperiment(OptunaExperiment):
    """Investigate the impact of multiple unknown token embeddings."""

    def get_model_conf(self, trial):
        return src.models.rnn_tagger.RNNTaggerConfig(
            d_emb_token=32,
            d_emb_tag=32,
            d_hidden_rnn=64,
            rnn_variant=self.model_variant,
            n_rnn_layers=1,
            mlp_sizes=[64, 64],
            dropout_rnn=0.0,
            dropout_between=0.3,
            n_unk_token=trial.suggest_int("n_unk_token", 1, 8),
        )

    def get_train_conf(self, trial) -> tagger_model.TrainSettings:
        return tagger_model.TrainSettings(
            start_lr=1e-3,
            weight_decay=0.0,
            label_smoothing=0.0,
            bs_train=32,
        )

    def get_pruner(self) -> BasePruner:
        return optuna.pruners.NopPruner()

    def get_sampler(self):
        return optuna.samplers.RandomSampler()


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
    exp = MultipleUnkExperiment(data["train"], data["val"], variant, split_idx)
    print("Experiment:", exp)
    exp.optimize(ntrials)


if __name__ == "__main__":
    main()
