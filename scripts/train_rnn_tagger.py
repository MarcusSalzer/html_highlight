"""Script for training the sequence tagger model."""

import json
import sys
from pathlib import Path

import torch

sys.path.append(".")
from src import plotly_plots, tagger_model, util
from src import torch_util as tu

RETRAIN_FINAL = False

# === File paths ===

PARAMS_FILE = Path("model_params")  # TODO read params from file/or two files?
DATASET_FILE = Path("data/dataset.ndjson")
# Output files go here:
SAVE_DIR = Path("models_trained")
MEDIA_DIR = Path("media")
# make dirs
SAVE_DIR.mkdir(exist_ok=True)
MEDIA_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Train a model"""

    # ===========  PARAMETERS  ===========
    model_conf = tagger_model.RNNTaggerConfig(
        d_emb_token=64,
        d_emb_tag=64,
        d_hidden_rnn=128,
        rnn_variant="gru",
        n_rnn_layers=2,
        mlp_sizes=[128],
        bidi=True,
        dropout_rnn=0.5,
        dropout_between=0.5,
        dropout_mlp=0.3,
    )

    train_settings = tagger_model.TrainSettings(
        label_smoothing=0.05,
        weight_decay=0.001,
        start_lr=1e-2,
        bs_train=32,
        loss_weights=None,
        stop_patience=100,
        max_epochs=3000,
    )

    # ===========     DATA     ===========
    split_idx, split_date = util.load_split_idx()

    print(f"Loaded split {split_date}")
    splits = util.load_dataset_splits(split_idx, path=DATASET_FILE)

    if RETRAIN_FINAL:
        print("[NOTE] Retraining model on train+val")
        splits = util.remap_val_train(splits)
        val_set = "test"
    else:
        val_set = "val"

    data = {sk: util.dataset_to_df(v) for sk, v in splits.items()}
    # get the vocabs
    vocab, token2idx, tag_vocab, tag2idx = util.make_vocab(data["train"])

    device = tu.get_dev()

    print(f"\n{len(vocab)=} | {len(tag_vocab)=} | {device=}\n")

    dsets = {
        k: tu.SequenceDataset.from_dataframe(data[k], token2idx, tag2idx, device="cpu")
        for k in ["train", val_set]
    }
    for k, d in dsets.items():
        print(f"{k}: {d}")

    # ===========     MODEL & TRAINING     ===========

    device = tu.get_dev()

    # model instance
    model = tagger_model.RNNTagger(model_conf, len(vocab), len(tag_vocab))
    model.to(device=device)

    # Where to store results
    model_dir = SAVE_DIR / str(model)
    model_dir.mkdir(exist_ok=True)

    # Save model configuration etc
    (model_dir / "train.json").write_text(json.dumps(dict(train_settings), indent=4))
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "config": dict(model_conf),
                "vocab": vocab,
                "tag_vocab": tag_vocab,
            },
            indent=4,
        )
    )

    print(f"\nTraining {model} on {device}...\n")
    print(f"Saves results at {model_dir}")

    # Train the model
    metrics = model.complete_train_loop(
        train_settings,
        dsets["train"],
        dsets[val_set],
        verbose=True,
        save_dir=model_dir,
    )

    if torch.cuda.is_available():
        maxmem = torch.cuda.max_memory_allocated()
        print(f"max memory use (cuda): {maxmem / 10**6:.0f} MB")

    plot = plotly_plots.train_metrics_single_run(metrics)
    plot_file = MEDIA_DIR / f"{model}_metrics.png"
    plot.write_image(plot_file, width=1200, height=600)
    print(f"saved plot at {plot_file}")


if __name__ == "__main__":
    main()
