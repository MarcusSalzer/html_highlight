"""CLI for model training"""

from datetime import datetime
from glob import glob
import json
import sys
from os import path as osp

sys.path.append(".")
from torch import optim, nn
from src import torch_util, util

MODEL_PARAM_DIR = "model_params"
DATA_DIR = "data"
SAVE_DIR = "tmp"


def pick_option(options, prompt):
    for i, m in enumerate(options):
        print(f"{i}. {m}")

    res = None
    while res is None:
        try:
            return options[int(input(prompt))]
        except ValueError:
            pass


def train_interactive() -> None:
    """Train a model"""

    # load hyperparameters
    param_file = pick_option(
        sorted(glob(f"{MODEL_PARAM_DIR}/*.json")),
        "Model parameters?",
    )
    with open(param_file) as f:
        params: dict = json.load(f)

    print("\n".join((f"{k}: {v}" for k, v in params["constructor"].items())))

    now = datetime.now()
    model_name = f"{osp.split(param_file)[-1][:-5]}_{now.month:02d}{now.day:02d}"

    # load data, get vocab
    data = util.load_examples_json(split_idx_id=params["split"])
    vocab, token2idx, tag_vocab, tag2idx = util.make_vocab(data["train"])

    print(f"\nVocab size   :{len(vocab)}")
    print(f"tagset size  :{len(tag_vocab)}\n")

    # Update and save metadata
    params["constructor"].update(
        {
            "token_vocab_size": len(vocab),
            "label_vocab_size": len(tag_vocab),
        }
    )
    params.update({"vocab": vocab, "tag_vocab": tag_vocab})
    meta_path = osp.join(SAVE_DIR, model_name + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(params, f)
    print(f"saved metadata: {meta_path}")

    # model instance
    model_type = model_name.split("_")[0]
    match model_type:
        case "lstm":
            model = torch_util.LSTMTagger(**params["constructor"])
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    # prepare data
    train_params: dict = params["train"]
    dev = torch_util.get_dev()
    model.to(device=dev)

    train_dl = torch_util.data2torch(
        data["train"], train_params["bs"], token2idx, tag2idx, dev
    )
    val_dl = torch_util.data2torch(
        data["val"], 2 * train_params["bs"], token2idx, tag2idx, dev
    )

    # Training details

    opt = optim.Adam(model.parameters(), train_params["lr"])
    lossfn = nn.CrossEntropyLoss()
    lr_s_pars = train_params.get("lrs")
    lr_s = None
    if lr_s_pars:
        print(f"LR schedule: {lr_s_pars[0]} ({lr_s_pars[1]})")
        if lr_s_pars[0] == "exponential":
            lr_s = optim.lr_scheduler.ExponentialLR(opt, **lr_s_pars[1])

    print(f"\nTraining {model_name} on {dev}")
    torch_util.train_loop(
        model,
        train_dl,
        val_dl,
        1000,
        opt,
        lossfn,
        lr_s,
        name=model_name,
        save_dir=SAVE_DIR,
    )


if __name__ == "__main__":
    train_interactive()
