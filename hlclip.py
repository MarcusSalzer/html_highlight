# mypy: disable-error-code="import-untyped"

"""Import and use `hlclip.hlclip()` to highlight clipboard"""

from datetime import datetime
import time

import pyperclip as pc

from src import inference
from src.html_process import format_html
from src.text_process import process
from src.util import MAP_TAGS

infer = inference.Inference("model_inference")


def hlclip():
    """Highlight source code in clipboard"""
    try:
        in_text = pc.paste()
    except Exception:
        in_text = "clipboard problem?"

    if not in_text:
        in_text = "Copy something!"

    save_to_history(in_text)

    # first, run deterministic process
    t0 = time.time()
    tokens, tags_det = process(in_text)
    t_det = time.time() - t0

    tags_pred = infer.run(tokens, tags_det)
    t_infer = time.time() - (t0 + t_det)
    print(
        f"Time: deterministic {t_det * 1000:.1f} | inference {t_infer * 1000:.1f} (ms)"
    )

    # combine with deterministic tags
    tags: list[str] = []
    for td, tp in zip(tags_det, tags_pred):
        if td == "uk":
            tags.append(tp)
        else:
            tags.append(td)

    tags = [MAP_TAGS.get(t, t) for t in tags]

    out_text = format_html(tokens, tags)

    print(" ".join(tags))
    pc.copy(out_text)


def save_to_history(text: str):
    name = datetime.now().isoformat()
    with open("data/hlclip_history/" + name + ".txt", "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    hlclip()
