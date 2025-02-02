from datetime import datetime
from datatools.benchmark import SequentialTimer
from src.util import MAP_TAGS

timer = SequentialTimer()

import pyperclip as pc  # noqa: E402

timer.add("import pyperclip")

from src.text_process import process, format_html  # noqa: E402

timer.add("import text_process")

from src import inference  # noqa: E402

timer.add("import inference")

print(timer)


def hlclip():
    """Highlight source code in clipboard"""
    try:
        in_text = pc.paste()
    except Exception:
        in_text = "clipboard problem?"

    save_to_history(in_text)

    tokens, tags_det = process(in_text)
    tags_pred = inference.run(tokens, tags_det)

    # combine with deterministic tags
    tags = []
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
