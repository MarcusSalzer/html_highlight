# from pathlib import Path
# import random

# from rich.console import Console
# import sys

# sys.path.append(".")
# from src import cli_util

# console = Console()

# src_folder = Path("data/hlclip_history")

# files = list(src_folder.iterdir())


# cli_util.clearCLI()

# print("INCOMPLETE...")
# while True:
#     console.print("-" * 60, style="dim")

#     file = random.choice(files)
#     console.print(f"{file.name}\n", style="black on white", highlight=False)

#     text = file.read_text()
#     print(text)

#     r = input("Language? (delete/skip): ").lower()
#     if r in ("del", "delete"):
#         file.unlink()
#         files = list(src_folder.iterdir())
#         cli_util.clearCLI()

#         console.print("deleted!", style="red")
#     if r in ("s", "skip"):
#         cli_util.clearCLI()
#         console.print("skip!", style="blue")
