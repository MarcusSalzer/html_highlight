import sys
from pathlib import Path

sys.path.append(".")
from src import util
from src.exper import sampler
from src.exper.mlflow_experiment import MLflowExperiment
from src.models.rnn_tagger import RNNTaggerConfig
from src.models.tagger_model import TrainSettings

# Keep datasplit for comparing models fairly
SPLIT_PATH = Path("./data/split_index_26-02-14.json")


def build_model_conf(params):
    return RNNTaggerConfig(
        d_emb_token=32,
        d_emb_tag=32,
        d_hidden_rnn=64,
        rnn_variant=params["rnn_variant"],
        n_rnn_layers=1,
        mlp_sizes=[64, 64],
        dropout_rnn=0.0,
        dropout_between=0.3,
        n_unk_token=1,
    )


def build_train_conf(params):
    return TrainSettings(
        start_lr=1e-3,
        weight_decay=0.0,
        label_smoothing=0.0,
        bs_train=32,
        plateau_lr_patience=10,
        stop_patience=20,
        min_lr=5e-5,
    )


if __name__ == "__main__":
    split_idx = util.load_split_idx(SPLIT_PATH)
    print(f"Loaded split {split_idx.date}")

    exp = MLflowExperiment(
        name=f"split-{split_idx.date}",
        split_idx=split_idx,
        filter_lang={"python"},
        model_conf_builder=build_model_conf,
        train_conf_builder=build_train_conf,
        sampler=sampler.GridSampler({"rnn_variant": ["lstm", "gru"]}),
    )
    print(exp)

    exp.run_sampler(repeat=3)
