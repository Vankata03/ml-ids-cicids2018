"""Изчислява номерата на главите, разделите, фигурите, таблиците и формулите без LaTeX.

build_docx.py и presentation/render_formulas.py четат номерата на препратките от
report/latex/main.aux, който се създава от XeLaTeX. Когато XeLaTeX не е наличен, този
скрипт обхожда главите в реда от main.tex и записва същите \\newlabel редове:

    python report/make_aux.py

Номерацията следва main.tex и preamble.tex: фигурите, таблиците и формулите имат обща
номерация за целия доклад (counterwithout), приложенията получават букви, а главата с
източниците запазва номер 6.
"""
from __future__ import annotations

import re
from pathlib import Path

LT = Path(__file__).resolve().parent / "latex"
EQ_ENVS = ("equation", "align", "gather", "multline")
TOKEN = re.compile(
    r"\\(?P<cmd>chapter|section|subsection)(?P<star>\*?)(?=[\[{])"
    r"|\\begin\{(?P<env>figure|table|longtable|" + "|".join(EQ_ENVS) + r")(?P<envstar>\*?)\}"
    r"|\\label\{(?P<label>[^}]+)\}"
    r"|\\input\{(?P<input>[^}]+)\}"
    r"|(?P<appendix>\\appendix)(?![A-Za-z])"
    r"|\\setcounter\{chapter\}\{(?P<setch>\d+)\}"
    r"|(?P<arabic>\\renewcommand\{\\thechapter\}\{\\arabic\{chapter\}\})"
)


def strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in s.split("\n"))


def expand(path: Path) -> str:
    """Текстът на файла с рекурсивно вмъкнати \\input файлове."""
    s = strip_comments(path.read_text(encoding="utf-8"))

    def sub(m):
        f = LT / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        return expand(f) if f.exists() else ""
    return re.sub(r"\\input\{([^}]+)\}", sub, s)


def labels() -> dict[str, str]:
    s = expand(LT / "main.tex")
    s = s[s.index(r"\begin{document}"):]
    ch = sec = sub = fig = tab = eq = 0
    appendix = False
    current, out = "", {}

    def chap() -> str:
        return "ABCDEFGH"[ch - 1] if appendix else str(ch)

    for m in TOKEN.finditer(s):
        if m.group("appendix"):
            appendix, ch = True, 0
        elif m.group("arabic"):
            appendix = False
        elif m.group("setch"):
            ch = int(m.group("setch"))
        elif m.group("cmd"):
            if m.group("star"):
                continue
            if m.group("cmd") == "chapter":
                ch += 1
                sec = sub = 0
                current = chap()
            elif m.group("cmd") == "section":
                sec, sub = sec + 1, 0
                current = f"{chap()}.{sec}"
            else:
                sub += 1
                current = f"{chap()}.{sec}.{sub}"
        elif m.group("env"):
            if m.group("envstar"):
                continue
            env = m.group("env")
            if env == "figure":
                fig += 1
                current = str(fig)
            elif env in ("table", "longtable"):
                tab += 1
                current = str(tab)
            else:
                eq += 1
                current = str(eq)
        elif m.group("label"):
            out[m.group("label")] = current
    return out


def main():
    lab = labels()
    lines = [rf"\newlabel{{{k}}}{{{{{v}}}{{0}}{{}}{{}}{{}}}}" for k, v in lab.items()]
    (LT / "main.aux").write_text("\\relax\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("written", LT / "main.aux", len(lab), "labels")


if __name__ == "__main__":
    main()
