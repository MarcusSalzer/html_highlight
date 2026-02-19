from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer

import mlflow

import src.models.rnn_tagger
from src import torch_util as tu
from src import util, vocab
from src.datamodels.split_index import SplitIndex
from src.datamodels.training import EpochSnapshot
from src.exper.sampler import Sampler
from src.models import tagger_model


@dataclass
class MLflowExperiment:
    name: str
    split_idx: SplitIndex
    filter_lang: set[str]
    model_conf_builder: Callable[[dict], src.models.rnn_tagger.RNNTaggerConfig]
    train_conf_builder: Callable[[dict], tagger_model.TrainSettings]
    sampler: Sampler
    opt_metric: str = "balanced_acc"
    tracking_uri: str = "sqlite:///data/mlflow.db"
    checkpoint_period: int | None = 5

    def __post_init__(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.name)

        # LOAD DATA
        data = {
            sk: util.dataset_to_df(v)
            for sk, v in util.load_dataset_splits(
                self.split_idx.id_to_group,
                path=Path("./data/dataset.ndjson"),
                filter_lang=self.filter_lang,
            ).items()
        }

        assert {"train", "val"}.issubset(data.keys()), f"missing data. Only got:{data.keys()=}"
        self.data_train = data["train"]
        self.data_val = data["val"]

    def __str__(self) -> str:
        datstr = f"train: {self.data_train.shape}, val: {self.data_val.shape}"
        info = ", ".join([self.name, type(self.sampler).__name__, datstr])
        return f"MlflowExperiment({info})"

    def run_single(
        self,
        model_conf,
        train_conf,
        seed: int | None = None,
    ):

        device = tu.get_dev()
        with mlflow.start_run():
            if seed is not None:
                mlflow.log_param("seed", seed)

            # Log configs (flattened)
            mlflow.log_params(model_conf.model_dump())
            mlflow.log_params(train_conf.model_dump())
            mlflow.log_param("filter_lang", self.filter_lang)
            mlflow.log_param("split_idx", self.split_idx.date)

            # Build vocabs
            vocs = vocab.both_vocabs(
                self.data_train,
                n_unknown_token=model_conf.n_unk_token,
            )

            model = src.models.rnn_tagger.RNNTagger(
                model_conf,
                vocab_sz_token=len(vocs.token),
                vocab_sz_tag=len(vocs.tag),
                n_extra=0,
            ).to(device)

            metrics = model.complete_train_loop(
                train_conf,
                tu.SequenceDataset.from_dataframe(self.data_train, vocs, device),
                tu.SequenceDataset.from_dataframe(self.data_val, vocs, device),
                epoch_cb=self._epoch_log,
            )

            best_val = max(metrics[f"val_{self.opt_metric}"])

            mlflow.log_metric(f"best_val_{self.opt_metric}", best_val)

            return best_val

    def _epoch_log(self, snap: EpochSnapshot):

        # Log metrics every epoch
        for k, v in snap.metrics.items():
            mlflow.log_metric(k, v, step=snap.epoch)
        # Log model sometimes
        chk = self.checkpoint_period

        if chk and (snap.epoch % chk) == 0:
            mlflow.log_artifact()

    def run_sampler(self, repeat: int = 1, verbose: bool = True):
        param_sets = self.sampler.generate()

        i = 0
        for params in param_sets:
            for r in range(repeat):
                i += 1
                t0 = default_timer()
                score = self.run_single(
                    model_conf=self.model_conf_builder(params),
                    train_conf=self.train_conf_builder(params),
                    seed=r,
                )
                t_run = default_timer() - t0
                if verbose:
                    print(f"run {i}/{len(param_sets) * repeat} ({t_run:.0f} s) -> {score=:.4f}")
