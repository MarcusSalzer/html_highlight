import pyperclip as pc
from text_functions import highlight_code

# highlight code in clipboard
try:
    in_text = pc.paste()
except Exception:
    in_text = "clipboard problem?"

out_text, classes = highlight_code(in_text)

print(out_text)
pc.copy(out_text)
print()
