from datatools.benchmark import SequentialTimer

timer = SequentialTimer()

import pyperclip as pc  # noqa: E402

timer.add("import pyperclip")
# from src.text_functions import highlight_code

# # highlight code in clipboard
# try:
#     in_text = pc.paste()
# except Exception:
#     in_text = "clipboard problem?"

# out_text, classes = highlight_code(in_text)

# print(out_text)
# pc.copy(out_text)
# print()

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

    tokens, tags_det = process(in_text)
    tags_pred = inference.run(tokens, tags_det)

    # combine with deterministic tags
    tags = []
    for td, tp in zip(tags_det, tags_pred):
        if td == "uk":
            tags.append(tp)
        else:
            tags.append(td)

    out_text = format_html(tokens, tags)

    print(out_text)
    pc.copy(out_text)


if __name__ == "__main__":
    hlclip()
