# %%

import sys
from importlib import reload
from pathlib import Path

sys.path.append("..")
from html_highlight.plots import data_plots, templates
from src import util

reload(templates)
templates.set_plotly_template()


df = util.load_dataset_df(Path("../data/dataset.ndjson"), include_derived=True)
print(f"loaded: {df.shape=}")


# %%

reload(data_plots)
data_plots.langs_pie(df).show()

# %%

reload(data_plots)
data_plots.seq_len_hist(df).show()
# %%

reload(data_plots)
data_plots.seq_len_lang_box(df).show()

# %%

reload(data_plots)
data_plots.seq_len_vs_unique(df).show()
