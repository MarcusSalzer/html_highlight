"""Script for training the sequence tagger model."""

import json
from pathlib import Path

import mlflow
import torch

from html_highlight import mlflow_wrapper, util, vocab
from html_highlight import torch_util as tu
from html_highlight.datamodels.training import EpochSnapshot
from html_highlight.models.rnn_tagger import RNNTagger, RNNTaggerConfig
from html_highlight.models.tagger_model import TrainSettings
from html_highlight.tensor_sequence_dataset import TensorSequenceDataset, get_dl

RETRAIN_FINAL = True

# === File paths ===

PARAMS_FILE = Path("model_params")  # TODO read params from file/or two files?
DATASET_FILE = Path("data/dataset.ndjson")
# Output files go here:
SAVE_PARENT = Path("models_trained")
MEDIA_DIR = Path("media")
# make dirs
SAVE_PARENT.mkdir(exist_ok=True)
MEDIA_DIR.mkdir(exist_ok=True)

DEVICE = tu.get_dev()
SAVE_WAIT = 10

DATALOADER_WORKERS = 0


def _epoch_log(snap: EpochSnapshot, save_dir: Path | None):

    # Log metrics every epoch
    for k, v in snap.metrics.items():
        mlflow.log_metric(k, v, step=snap.epoch)

    # save model state if specified
    for k, (ep_b, _) in snap.best.items():
        if save_dir is not None and snap.epoch > SAVE_WAIT and ep_b == snap.epoch:
            fp = save_dir / f"best_{k}_state.pth"
            torch.save(snap.model.state_dict(), fp)


def main() -> None:
    """Train a model"""

    # ===========  PARAMETERS  ===========
    model_conf = RNNTaggerConfig(
        d_emb_token=64,
        d_emb_tag=64,
        d_hidden_rnn=64,
        rnn_variant="lstm",
        n_rnn_layers=2,
        mlp_sizes=[256, 128],
        bidi=True,
        dropout_rnn=0.1,
        dropout_between=0.2,
        dropout_mlp=0.3,
        n_unk_token=1,
    )

    train_settings = TrainSettings(
        label_smoothing=0.01,
        weight_decay=0.0001,
        start_lr=2e-2,
        bs_train=64,
        loss_weights=None,
        stop_patience=100,
        max_epochs=10_000,
    )

    # ===========     DATA     ===========
    split_idx = util.load_split_idx()

    print(f"Loaded {split_idx}")
    splits = util.load_dataset_splits(split_idx.id_to_group, path=DATASET_FILE)

    if RETRAIN_FINAL:
        print("[NOTE] Retraining model on train+val")
        splits = util.remap_val_train(splits)
        val_set = "test"
    else:
        val_set = "val"

    # precompute deterministic tags
    data = {sk: tu.add_tag_det_col(util.dataset_to_df(v)) for sk, v in splits.items()}
    # get the vocabs
    vocs = vocab.both_vocabs(
        data["train"],  # Build vocabs from training data
        n_unknown_token=1,  # How many unknown tokens to track
    )

    print(f"\n{len(vocs.token)=} | {len(vocs.tag)=} | {DEVICE=}\n")

    dsets = {k: TensorSequenceDataset(**tu.df_to_tensorlists(d, vocs)) for k, d in data.items()}

    for k, d in dsets.items():
        print(f"{k:<8}: {d}")

    # ===========     MODEL & TRAINING     ===========

    # model instance
    model = RNNTagger(model_conf, len(vocs.token), len(vocs.tag), n_extra=None)
    model.to(device=DEVICE)

    # Where to store results
    model_dir = SAVE_PARENT / str(model)
    model_dir.mkdir(exist_ok=True)

    # Save model configuration etc
    (model_dir / "train.json").write_text(train_settings.model_dump_json(indent=2))
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "config": model_conf.model_dump(),
                "vocab": vocs.token.vocab_list,
                "tag_vocab": vocs.tag.vocab_list,
            },
            indent=2,
        )
    )

    print(f"\nTraining {model} on {DEVICE}...\n")
    print(f"Saves results at {model_dir}")

    mlflow_wrapper.init(f"TrainRnnTagger_T{len(dsets['train'])}V{len(dsets[val_set])}")

    with mlflow.start_run():
        # Log configs (flattened)
        mlflow_wrapper.log_params_pydantic([model_conf, train_settings])
        mlflow.log_param("split_idx", split_idx.date)

        # Train the model
        _ = model.complete_train_loop(
            train_settings,
            get_dl(
                dsets["train"],
                shuffle=True,
                bs=train_settings.bs_train,
                n_workers=DATALOADER_WORKERS,
            ),
            get_dl(dsets[val_set], shuffle=False, bs=64, n_workers=DATALOADER_WORKERS),
            verbose=True,
            epoch_cb=lambda snap: _epoch_log(snap, model_dir),
        )

    if torch.cuda.is_available():
        maxmem = torch.cuda.max_memory_allocated()
        print(f"max memory use (cuda): {maxmem / 10**6:.0f} MB")


if __name__ == "__main__":
    main()
