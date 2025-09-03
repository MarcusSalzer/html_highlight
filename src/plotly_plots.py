from typing import Any, Sequence
import numpy as np
from plotly import graph_objects as go, io as pio, subplots


pio.templates.default = "plotly_dark"


def train_metrics_single_run(metrics: dict[str, Any]):
    fig = subplots.make_subplots(
        rows=1, cols=2, x_title="Epoch", subplot_titles=["Loss", "Accuracy"]
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
                row=1,
                col=2,
            )

    return fig
