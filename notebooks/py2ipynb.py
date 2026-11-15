"""Преобразува notebook_src.py (формат с # %% клетки) в .ipynb."""
import re, sys, nbformat
src, out = sys.argv[1], sys.argv[2]
cells, cur, kind = [], [], None
def flush():
    if kind is None: return
    text = "\n".join(cur).strip("\n")
    if kind == "md":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(text))
for line in open(src, encoding="utf-8").read().split("\n"):
    if line.startswith("# %%"):
        flush(); cur = []; kind = "md" if "[markdown]" in line else "code"
    else:
        cur.append(line)
flush()
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
nbformat.write(nb, out)
print(len(cells), "клетки ->", out)
