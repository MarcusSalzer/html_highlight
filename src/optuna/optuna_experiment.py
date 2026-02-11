from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import polars as pl

import optuna
from src import tagger_model, vocab
from src import torch_util as tu
from src.datamodels.split_index import SplitIndex


@dataclass
class OptunaExperiment:
    """Do some sort of experiment."""

    data_train: pl.DataFrame
    data_valid: pl.DataFrame

    model_variant: Literal["rnn", "gru", "lstm"]
    split_idx: SplitIndex
    opt_metric = "balanced_acc"

    def __str__(self) -> str:
        return self.name

    @property
    def name(self):
        return f"{type(self).__name__}_{self.model_variant}_{self.split_idx.date}_{self.opt_metric}"

    @property
    def device(self):
        return tu.get_dev()

    def _create_study(self):
        study = optuna.create_study(
            storage="sqlite:///data/optuna.db",
            study_name=self.name,
            load_if_exists=True,
            pruner=self.get_pruner(),
            direction="maximize",
        )
        study.set_metric_names([self.opt_metric])
        return study

    @staticmethod
    def epoch_callback(trial: optuna.Trial, value: float, step: int):
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    def _objective(self, trial: optuna.Trial):
        model_conf = self.get_model_conf(trial)
        train_conf = self.get_train_conf(trial)

        # Build vocabs from training data
        vocs = vocab.both_vocabs(self.data_train, n_unknown_token=model_conf.n_unk_token)

        # model
        model = tagger_model.RNNTagger(
            model_conf,
            vocab_sz_token=len(vocs.token),
            vocab_sz_tag=len(vocs.tag),
            n_extra=0,  # EXTRA FEATURES?
        ).to(device=self.device)

        # Data and train

        metrics = model.complete_train_loop(
            train_conf,
            tu.SequenceDataset.from_dataframe(self.data_train, vocs, device=self.device),
            tu.SequenceDataset.from_dataframe(self.data_valid, vocs, device=self.device),
            epoch_cb=lambda ep, mets: self.epoch_callback(
                trial,
                mets[f"val_{self.opt_metric}"],
                ep,
            ),
        )

        return max(metrics[f"val_{self.opt_metric}"])

    def optimize(self, n_trials: int):
        self._create_study().optimize(self._objective, n_trials)

    @abstractmethod
    def get_pruner(self) -> optuna.pruners.BasePruner:
        pass

    @abstractmethod
    def get_model_conf(self, trial: optuna.Trial) -> tagger_model.RNNTaggerConfig:
        pass

    @abstractmethod
    def get_train_conf(self, trial: optuna.Trial) -> tagger_model.TrainSettings:
        pass
