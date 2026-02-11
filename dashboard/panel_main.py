import sys

import panel as pn

sys.path.append(".")

from src import util

pn.extension(design="material", sizing_mode="stretch_width")
PRIMARY_COLOR = "#3ACB1D"
SECONDARY_COLOR = "#B54300"


@pn.cache
def get_data():
    data = util.load_dataset_parallel()
    return util.dataset_to_df(data)


data = get_data()


variable_widget = pn.widgets.Select(
    name="variable", value="Temperature", options=list(data.columns)
)
window_widget = pn.widgets.IntSlider(name="window", value=30, start=1, end=60)
sigma_widget = pn.widgets.IntSlider(name="sigma", value=10, start=0, end=20)
widgets = pn.Column(variable_widget, window_widget, sigma_widget, sizing_mode="fixed", width=300)

# Template to show
pn.template.MaterialTemplate(
    site="Panel",
    title="Getting Started App",
    sidebar=[variable_widget, window_widget, sigma_widget],
).servable()
