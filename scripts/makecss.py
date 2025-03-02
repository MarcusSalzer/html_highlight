from src import util


def make_css(
    tags: list,
    exclude: list[str] = ["ws", "id", "nl", "brcl", "brop", "<unk>"],
    max_br=4,
):
    doc = "body {\n  color: white;\n  background-color: black;\n}\n"
    doc = 'pre {\n  font-family: "Comic Mono", monospace;'
    doc += (
        "mark {\n  color: white;\n  background-color: rgba(129, 129, 129, 0.378);\n}\n"
    )
    doc += ".error {\n  text-decoration: underline wavy red 1px;\n}\n"

    for k in range(max_br):
        tags.append(f"br{k}")

    for t in tags:
        if t in exclude:
            continue
        doc += f".{t}" + " {\n  color: white;\n}\n"

    with open("_style.css", "w", encoding="utf-8") as f:
        f.write(doc)


if __name__ == "__main__":
    data = util.load_examples_json(verbose=False)
    tags = data["tags"].explode().unique().to_list()
    make_css(sorted(tags))
