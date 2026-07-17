"""Main script for training a model."""

from pathlib import Path

import torch
from clearml import Dataset

from html_highlight import lightning_utils, util, vocab
from html_highlight import torch_util as tu
from html_highlight.cml.cleaml_util import task_init
from html_highlight.models.tagger_lit import (
    RNNTagger,
    TaggerDataModule,
    TaggerLitModule,
    TrainSettings,
    build_trainer,
)
from html_highlight.tensor_sequence_dataset import TensorSequenceDataset


def _main():
    task = task_init("train_tagger")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    lightning_utils.stop_annoying_info_messages()

    # ===========     DATA     ===========
    cml_dset = Dataset.get(dataset_name="examples_split", alias="examples_split")
    print("found dset", cml_dset.name)

    dset_dir = Path(cml_dset.get_local_copy())
    split_idx = util.load_split_idx(dset_dir / "split_index.json")
    print(f"Loaded split index from {split_idx.date}")

    splits = util.load_dataset_splits(split_idx.id_to_group, path=dset_dir / "dataset.ndjson")

    # precompute deterministic tags
    data = {sk: tu.add_tag_det_col(util.dataset_to_df(v)) for sk, v in splits.items()}
    # get the vocabs
    vocs = vocab.both_vocabs(
        data["train"],  # Build vocabs from training data
        n_unknown_token=1,  # How many unknown tokens to track
    )

    print(f"\n{len(vocs.token)=} | {len(vocs.tag)=} | {DEVICE=}\n")

    dsets = {k: TensorSequenceDataset(**tu.df_to_tensorlists(d, vocs)) for k, d in data.items()}

    model = RNNTagger(len(vocs.token), len(vocs.tag))
    train_settings = TrainSettings(max_epochs=10)

    trainer = build_trainer(train_settings.stop_patience, train_settings.max_epochs)

    # train the model
    trainer.fit(
        TaggerLitModule(model, train_settings),
        TaggerDataModule(dsets, train_settings.bs_train),
    )

    task.close()


_main()
