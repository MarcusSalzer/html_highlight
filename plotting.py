import numpy as np
from plotly import graph_objects as go
from plotly import io as pio


def scatter(x=None, y=None):
    # handle single list input??

    return go.Figure([go.Scatter(x=x, y=yk) for yk in y])
