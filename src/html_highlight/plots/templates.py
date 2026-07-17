from plotly import graph_objects as go
from plotly import io as pio


def set_plotly_template(
    base_template="plotly_dark",
    auto_size=False,
    w: int = 600,
    h: int = 300,
    transparent_background=True,
    margin=60,
):
    """Some kind of plot template"""
    plot_temp = pio.templates[base_template]
    layout = plot_temp.layout
    assert isinstance(layout, go.Layout)
    layout.margin = dict.fromkeys(["t", "l", "r", "b"], margin)

    if not auto_size:
        layout.width = w
        layout.height = h
        layout.autosize = False
    if transparent_background:
        layout.paper_bgcolor = "rgba(0,0,0,0.1)"
        layout.plot_bgcolor = "rgba(0,0,0,0)"

    plot_temp.layout = layout
    pio.templates.default = plot_temp
