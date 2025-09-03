"""CLI for model training"""

import json
from pathlib import Path
import sys
from datetime import datetime
from torch import nn, optim


sys.path.append(".")
from src import types
from src import torch_util, util
from src.cli_util import pick_option
from src import plotly_plots

MODEL_PARAM_DIR = Path("model_params")
DATA_DIR = Path("data")
SAVE_DIR = Path("models_trained")
MEDIA_DIR = Path("media")


def train_interactive() -> None:
    """Train a model"""

    # load hyperparameters
    param_fn = pick_option(
        sorted(MODEL_PARAM_DIR.glob("*.json")),
        "Model parameters?",
    )
    assert param_fn is not None
    with Path(param_fn).open() as f:
        setup = types.ModelTrainSetup(**json.load(f))

    now = datetime.now()
    model_name = f"{Path(param_fn).name.split('.')[0]}_{now.month:02d}{now.day:02d}"

    # load data, convert to dataframe
    split_idx = util.load_split_idx(setup.split)
    data = {
        sk: util.dataset_to_df(v)
        for sk, v in util.load_dataset_splits(split_idx).items()
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
    print(f"\n{model_name}")
    print("\n".join((f"  {k:.<20} {v}" for k, v in setup.constructor.items())))

    meta = types.TrainResultMeta(setup=setup, vocab=vocab, tag_vocab=tag_vocab)

    meta_path = SAVE_DIR / f"{model_name}_meta.json"
    meta_path.parent.mkdir(exist_ok=True, parents=True)
    meta_path.write_text(meta.model_dump_json(indent=2))

    print(f"saved metadata: {meta_path}")

    # model instance
    model_type = model_name.split("_")[0]
    match model_type:
        case "lstm":
            model = torch_util.LSTMTagger(**setup.constructor)  # type: ignore
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    # prepare data

    device = torch_util.get_dev()
    model.to(device=device)

    train_dl = torch_util.data2torch(
        data["train"], setup.train.bs, token2idx, tag2idx, device, model.n_extra
    )
    val_dl = torch_util.data2torch(
        data["val"], 2 * setup.train.bs, token2idx, tag2idx, device, model.n_extra
    )

    # Training details

    opt = optim.Adam(model.parameters(), setup.train.lr)
    lossfn = nn.CrossEntropyLoss()
    lr_s_pars = setup.train.lrs
    lr_s = None
    if lr_s_pars is not None:
        print(f"LR schedule: {lr_s_pars[0]} ({lr_s_pars[1]})")
        if lr_s_pars[0] == "exponential":
            lr_s = optim.lr_scheduler.ExponentialLR(opt, **lr_s_pars[1])

    print(f"\nTraining {model_name} on {device}")
    metrics = torch_util.train_loop(
        model,
        train_dl,
        val_dl,
        setup.train.epochs,
        opt,
        lossfn,
        lr_s,
        name=model_name,
        save_dir=SAVE_DIR,
        reduce_lr_on_plat=setup.train.lrs_plat,
        stop_patience=setup.train.stop_patience,
    )
    meta.metrics = metrics
    meta_path.write_text(meta.model_dump_json(indent=2))

    print(f"updated meta ({meta_path}) with metrics")

    plot = plotly_plots.train_metrics_single_run(metrics)
    plot_file = MEDIA_DIR / f"{model_name}_metrics.png"
    plot.write_image(plot_file, width=1200, height=600)
    plot.write_html(MEDIA_DIR / f"{model_name}_metrics.html")
    plot.show(renderer="browser")
    print(f"saved plot at {plot_file}")


if __name__ == "__main__":
    train_interactive()
