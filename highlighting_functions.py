import regex as re


class Highlighter:
    patterns_find = {}
    patterns_repl = {}

    def __init__(self, patterns_find, patterns_repl):
        self.patterns_find = patterns_find
        self.patterns_repl = patterns_repl

    def pre_process(self, text: str):
        # convert html reserved characters
        text = re.sub(r"&", r"&amp;", text)
        text = re.sub(r"<", r"&lt;", text)
        text = re.sub(r">", r"&gt;", text)
        text = re.sub(r'"', r"&quot;", text)
        text = re.sub(r"'", r"&apos;", text)
        # preserve line-breaks
        text = re.sub(r"\n", r"<br>\n", text)
        # preserve indentation
        text = re.sub(r"  ", r"&nbsp; ", text)
        return text

    def find_variables(self, text: str):
        """Finds variables in code.

        ## Returns
        variables: list[str]
            List of variable names

        pattern_variable: str
            regex pattern for finding variables"""

        variables = re.findall(r"\b[^\s]+ ?=", text)
        variables = set(map(lambda s: re.sub(" ?=", "", s), variables))
        pattern_variable = r"\b(" + "|".join(variables) + r")\b"

        return variables, pattern_variable

    def syntax_highlight(self, text: str):
        patterns = self.patterns_find

        tagged = {}
        # tag with placeholders
        for tag in patterns.keys():
            tagged[tag] = []
            while True:
                m = re.search(patterns[tag], text)
                if m:
                    tagged[tag].append(f"""<span class="{tag}">{m.group()}</span>""")
                    text = text[: m.start()] + f"£££{tag}£££" + text[m.end() :]
                else:
                    break

            # fix parantheses
            for i, t in enumerate(tagged[tag]):
                tagged[tag][i] = re.sub(r"\(</span>", r"</span>(", t)

        # replace placeholders
        for tag in tagged.keys():
            for i, t in enumerate(tagged[tag]):
                m = re.search(f"£££{tag}£££", text)
                text = text[: m.start()] + t + text[m.end() :]

        css_link = '<link rel="stylesheet" type="text/css" href="_highlight_style.css">'
        html_text = f"""<div class="code-snippet">{text}</div>"""
        final_html = f"""
        <head>
            {css_link}
        </head>
        <body>
            {html_text}
        </body>
        """
        return final_html

    def process(self, text: str):
        """Preprocesses, finds variables and highlights."""

        text = self.pre_process(text)
        variables, p_variable = self.find_variables(text)
        if len(variables) > 0:
            self.patterns_find["variable"] = p_variable

        text = self.syntax_highlight(text)
        return text
