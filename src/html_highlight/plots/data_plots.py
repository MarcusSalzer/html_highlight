import polars as pl
from plotly import graph_objects as go


def langs_pie(df: pl.DataFrame):
    lang_counts = df.group_by("lang").len().sort("len", descending=True)

    return go.Figure(
        data=[
            go.Pie(
                labels=lang_counts["lang"].to_list(),
                values=lang_counts["len"].to_list(),
                hole=0.4,
                textposition="inside",
            )
        ],
        layout=go.Layout(
            title="Dataset Distribution by Language",
            uniformtext=go.layout.Uniformtext(minsize=12, mode="hide"),
        ),
    )


def seq_len_hist(df: pl.DataFrame):
    return go.Figure(
        data=[
            go.Histogram(
                x=df["seq_len"].to_list(),
                nbinsx=50,
            )
        ],
        layout=go.Layout(
            title="Sequence Length Distribution",
            xaxis=dict(title="Sequence Length (tokens)"),
            yaxis=dict(title="Count"),
            bargap=0.05,
        ),
    )


def seq_len_lang_box(df: pl.DataFrame):
    return go.Figure(
        data=[
            go.Box(
                y=df.filter(pl.col("lang") == lang)["seq_len"].to_list(),
                name=lang,
                boxmean=True,
            )
            for lang in df["lang"].unique().sort().to_list()
        ],
        layout=go.Layout(
            title="Sequence Length Distribution per Language",
            yaxis=dict(title="Sequence Length"),
            xaxis=dict(title="Language"),
        ),
    )


def lang_difficulty_heat(df: pl.DataFrame):
    pivot = (
        df.group_by(["lang", "difficulty"])
        .len()
        .pivot(
            on=["difficulty"],
            values="len",
            index="lang",
            aggregate_function="first",
        )
        .fill_null(0)
    )

    langs = pivot["lang"].to_list()
    difficulties = [c for c in pivot.columns if c != "lang"]

    z = [pivot.select(diff).to_series().to_list() for diff in difficulties]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=langs,
                y=difficulties,
            )
        ],
        layout=go.Layout(
            title="Language vs Difficulty Distribution",
            xaxis=dict(title="Language"),
            yaxis=dict(title="Difficulty"),
        ),
    )

    return fig


def seq_len_vs_unique(df: pl.DataFrame):
    fig = go.Figure(
        layout=go.Layout(
            title="Sequence Length vs Unique Tags",
            xaxis=dict(title="Sequence Length"),
            yaxis=dict(title="Number of Unique Tags"),
        ),
    )
    for lang in df["lang"].unique().sort().to_list():
        df_la = df.filter(lang=lang)
        fig.add_trace(
            go.Scatter(
                x=df_la["seq_len"].to_list(),
                y=df_la["unique_tags"].list.len().to_list(),
                mode="markers",
                name=lang,
            )
        )

    return fig
