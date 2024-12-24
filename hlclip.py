import pyperclip as pc
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

from src.text_process import process, format_html

try:
    in_text = pc.paste()
except Exception:
    in_text = "clipboard problem?"

# TODO CSS!
out_text = format_html(*process(in_text))

print(out_text)
pc.copy(out_text)
print()
