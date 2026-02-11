import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

texts = [
    "First document full text ...",
    "Second document full text ...",
    "Third document full text ...",
]

# fake embeddings for demo
rng = np.random.default_rng(0)
embeddings = rng.normal(size=(len(texts), 2))

DATA = {
    "texts": texts,
    "points": [
        {
            "id": i,
            "x": float(embeddings[i, 0]),
            "y": float(embeddings[i, 1]),
            "preview": texts[i][:120] + "…",
        }
        for i in range(len(texts))
    ],
}


def embedding_scatter(points):
    df = pd.DataFrame(points)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        custom_data=["id"],
        hover_data={"preview": True, "x": False, "y": False},
    )

    fig.update_layout(
        clickmode="event",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    fig = embedding_scatter(DATA["points"])
    fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    return f"""
    <html>
      <head>
      <script src="https://cdn.plot.ly/plotly-3.3.0.min.js" charset="utf-8"></script>
        <style>
          body {{ font-family: sans-serif; margin: 0; }}
          #container {{ display: flex; height: 100vh; }}
          #plot {{ flex: 1; }}
          #detail {{
            width: 40%;
            border-left: 1px solid #ccc;
            padding: 1em;
            overflow: auto;
          }}
        </style>
      </head>

      <body>
        <div id="container">
          <div id="plot">{fig_html}</div>
          <div id="detail">
            <em>Click a point to view full text</em>
          </div>
        </div>

        <script>
          const plot = document.querySelector('.plotly-graph-div');

          plot.on('plotly_click', function(e) {{
          console.log("AA");
            const docId = e.points[0].customdata[0];
            fetch(`/doc/${{docId}}`)
              .then(r => r.text())
              .then(html => {{
                document.getElementById('detail').innerHTML = html;
              }});
          }});
        </script>
      </body>
    </html>
    """


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def doc(doc_id: int):
    text = DATA["texts"][doc_id]
    return f"<pre>{text}</pre>"
