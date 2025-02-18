"""CLI for model training"""

from glob import glob
import json

from torch import optim, nn
from src import torch_util, util


MODEL_DIR = "models"
DATA_DIR = "data"


def train_interactive() -> None:
    """Train a model"""
    models_meta = sorted(glob(f"{MODEL_DIR}/*_meta.json"))
    model_names = [m.split("/")[-1].replace("_meta.json", "") for m in models_meta]
    print("Models:")
    for i, m in enumerate(model_names):
        print(f"{i}. {m}: {models_meta[i]}")

    model_idx = int(input("Model?"))
    model_name = model_names[model_idx]

    with open(models_meta[model_idx]) as f:
        metadata = json.load(f)

    print("\n".join((f"{k}: {v}" for k, v in metadata["constructor_params"].items())))

    model_type = model_name.split("_")[0]
    match model_type:
        case "lstm":
            model = torch_util.LSTMTagger(**metadata["constructor_params"])
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    # load data
    split_files = sorted(glob(f"{DATA_DIR}/split_index_*.json"))
    split_ids = [f.split("_")[-1].split(".")[0] for f in split_files]
    print("\nSplits:")
    for i, m in enumerate(split_ids):
        print(f"{i}. {m}")

    split_idx = int(input("Split?"))
    split_id = split_ids[split_idx]

    data = util.load_examples_json(split_idx_id=split_id)

    # get vocab
    vocab = metadata["vocab"]
    tag_vocab = metadata["tag_vocab"]
    token2idx = {t: i for i, t in enumerate(vocab)}
    tag2idx = {t: i for i, t in enumerate(tag_vocab)}

    print(f"\nVocab size   :{len(vocab)}")
    print(f"tagset size  :{len(tag_vocab)}\n")
    # prepare data
    dev = torch_util.get_dev()
    model.to(device=dev)

    bs = int(input("batch-size? "))

    train_dl = torch_util.data2torch(data["train"], bs, token2idx, tag2idx, dev)
    val_dl = torch_util.data2torch(data["val"], 2 * bs, token2idx, tag2idx, dev)

    # Training details

    train_params: dict = metadata.get("train_params")

    lr = train_params.get("lr", 0.001)
    opt = optim.Adam(model.parameters(), lr)
    lossfn = nn.CrossEntropyLoss()
    lr_s = None
    if train_params is not None:
        lr_s_pars = train_params.get("lr_s")
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
        save_dir="./tmp",
    )


if __name__ == "__main__":
    train_interactive()
