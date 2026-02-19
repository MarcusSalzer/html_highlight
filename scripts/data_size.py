import sys

import polars as pl
import torch

sys.path.append(".")
from src import util

# Check dataset memory use


data_df = util.load_dataset_df()
print(f"{data_df.shape=}")
print(f"est. df size: {data_df.estimated_size(unit='kilobytes'):.1f} kB")

maxlen = data_df.select(pl.col("tokens").list.len().max()).item()

print("longest:", maxlen)
bigtens = torch.zeros([len(data_df), maxlen], dtype=torch.int64)
print(f"A {tuple(bigtens.shape)} i64 tensor uses {bigtens.nbytes / 1000:.1f} kB")
