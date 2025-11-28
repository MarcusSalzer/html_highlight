from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from plotly import graph_objects as go
from plotly import io as pio
from plotly import subplots

pio.templates.default = "plotly_dark"


def train_metrics_single_run(metrics: dict[str, Any], logloss: bool = True):
    fig = subplots.make_subplots(
        rows=1,
        cols=2,
        x_title="Epoch",
        subplot_titles=["Loss" + " (log scale)" * logloss, "Accuracy"],
    )

    # lines
    for mk in metrics:
        if "loss" in mk:
            c = 1
        elif "acc" in mk:
            c = 2
        else:
            print(f"WARNING: unexpected key {mk}, skips")
            continue

        fig.add_trace(
            go.Scatter(y=metrics[mk], name=mk),
            row=1,
            col=c,
        )
        if "acc" in mk:
            # mark maximum
            max_acc_idx = np.argmax(metrics[mk])
            fig.add_annotation(
                text=f"max: {metrics[mk][max_acc_idx]:.2%}",
                x=max_acc_idx,
                y=metrics[mk][max_acc_idx],
                ay=0,
                showarrow=False,
                bgcolor="rgba(0,0,0,0.6)",
                borderpad=3,
                row=1,
                col=2,
            )

    if logloss:
        fig.update_yaxes(go.layout.YAxis(type="log"), row=1, col=1)

    return fig


def heatmap(mat: NDArray, width: int = 400):
    return go.Figure(
        go.Heatmap(z=mat),
        go.Layout(
            yaxis=go.layout.YAxis(scaleanchor="x"),
            width=width,
            height=width - 20,
            margin=dict(t=20, l=20, r=20, b=20),
        ),
    )


def heatmaps_simple(
    arrs: Sequence[NDArray],
    titles: list[str] | None = None,
    width: int = 400,
    bg: str = "rgba(0,0,0,0)",
):
    fig = subplots.make_subplots(rows=1, cols=len(arrs), subplot_titles=titles)

    for i, z in enumerate(arrs):
        fig.add_trace(
            go.Heatmap(z=z),
            row=1,
            col=i + 1,
        )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(scaleanchor="x", visible=False)

    fig.update_layout(
        go.Layout(
            width=width,
            height=width // len(arrs),
            margin=dict(t=40, l=20, r=20, b=20),
            plot_bgcolor=bg,
            paper_bgcolor=bg,
        ),
    )
    return fig
