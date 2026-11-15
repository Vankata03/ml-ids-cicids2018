"""Генерира доклада във формат Word (.docx) от LaTeX главите и резултатите.

Етапи:
  1. LaTeX главите -> Markdown (стойностите \\val, препратките и номерата се
     разрешават от report/latex/generated/values.tex и report/latex/main.aux);
  2. всеки абзац получава номера на използваните източници във формат [1], [2];
     източниците се номерират по реда на първото им цитиране;
  3. pandoc -> .docx (формулите стават редактируеми формули на Word);
  4. python-docx: стилове, номера на формулите, таблици, заглавна страница,
     съдържание, горен и долен колонтитул.

Изисква pandoc, python-docx и LibreOffice (за номерата на страниците в съдържанието).
Изпълнение (след python report/build_report.py): python report/build_docx.py
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
LT = ROOT / "report" / "latex"
FIG = ROOT / "results" / "figures"
BUILD = ROOT / "report" / "_docx_build"
OUT = ROOT / "report" / "Доклад_ML_IDS_CSE-CIC-IDS2018.docx"

# Оформление по шаблона на ТУ-София: Times New Roman 12 pt, черен текст, A4
NAVY, ACCENT, MUTED, ORANGE = "000000", "000000", "404040", "000000"
BODY_FONT, HEAD_FONT = "Times New Roman", "Times New Roman"
LOGO = ROOT / "report" / "assets" / "tu_sofia_logo.png"
THIRD_MEMBER = "Тихомир Божков, Fak No. 961326014"  # попълва се с име и факултетен номер
RUNNING_TITLE = "ML откриване на прониквания в трафика на център за данни (CSE-CIC-IDS2018)"

CHAPTERS = ["00_abstract", "01_intro", "02_methodology", "03_framework", "04_plan", "05_results",
            "06_analysis", "07_conclusion", "09_appendix"]
APPENDIX_LETTERS = "АБВГДЕЖЗ"

MACROS = r"""
\newcommand{\vect}[1]{\mathbf{#1}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\txt}[1]{\text{#1}}
\newcommand{\argmax}{\mathop{\mathrm{arg\,max}}}
\newcommand{\argmin}{\mathop{\mathrm{arg\,min}}}
\newcommand{\sign}{\operatorname{sign}}
\newcommand{\bm}[1]{\boldsymbol{#1}}
"""

# ---------------------------------------------------------------------------
# Източници: ключови думи -> източник (за цитирането след всеки абзац)
# ---------------------------------------------------------------------------
KEYWORDS = [
    (r"ciciot2023|105 (реални )?iot|набора? от данни|canadian institute", "neto2023"),
    (r"изтегл|официалн(ия|ата) (сървър|дистрибуция|сайт)|регистрация", "cic2023web"),
    (r"mirai|ботнет", "antonakakis2017"),
    (r"откриване на прониквания|\bids\b|intrusion detection", "sommer2010"),
    (r"машинно обучение за|data mining|ml-базиран", "buczak2016"),
    (r"\biat\b|артефакт|изтичане на информация|leakage", "arp2022"),
    (r"\bddos\b|отказ от услуга|наводн", "mirkovic2004"),
    (r"сканиран|портове|nmap|разузнава", "lyon2009"),
    (r"kdd|nsl-kdd", "tavallaee2009"),
    (r"cic-ids2017|cicids2017", "sharafaldin2018"),
    (r"\biat\b", "engelen2021"),
    (r"джини|решаващ(о|и)( дърв)|cart|дърво", "breiman1984"),
    (r"random forest|случайн(а|ата) гора|\brf\b|bagging|mdi", "breiman2001"),
    (r"стандартиз|z-оценка|преобучение|bias|компромис между отместване", "hastie2009"),
    (r"\bsvm\b|опорни(те)? вектори|максимален марж|хиперравнина", "cortes1995"),
    (r"libsvm|\bsvc\b|smo", "chang2011"),
    (r"\brbf\b|ядро|ядрото|\\gamma|γ", "scholkopf2002"),
    (r"\bpca\b|главни(те)? компоненти|главна компонента|собствен", "jolliffe2016"),
    (r"t-sne", "vandermaaten2008"),
    (r"кръстосана валидация|\bcv\b|фолд|gridsearch", "kohavi1995"),
    (r"\broc\b|\bauc\b", "fawcett2006"),
    (r"precision[-–—]recall|\bpr\b крив|средна(та)? точност|\bap\b", "saito2015"),
    (r"базов(ата|а) честота|естествено(то)? съотношение|реалистичн", "axelsson2000"),
    (r"bootstrap|доверителн", "efron1993"),
    (r"mcnemar", "mcnemar1947"),
    (r"mcnemar|сравняване на (два )?класификатор|статистически значим", "dietterich1998"),
    (r"пермутационн", "fisher2019"),
    (r"scikit-learn|sklearn|pipeline|gridsearchcv|standardscaler|joblib", "pedregosa2011"),
    (r"хранилище|run_all|registry|experiment\.py|програмн(ия|ата) (код|реализация)|\.csv|\.json|журнал|"
     r"seed|възпроизвод|тестове", "repo"),
]
# Източник по подразбиране за абзац без съвпадение, по глава
CHAPTER_DEFAULT = {
    "00_abstract": ["neto2023"], "01_intro": ["sommer2010"], "02_methodology": ["hastie2009"],
    "03_framework": ["pedregosa2011", "repo"], "04_plan": ["repo"], "05_results": ["neto2023", "repo"],
    "06_analysis": ["neto2023"], "07_conclusion": ["neto2023"], "09_appendix": ["repo"],
}
MAX_REFS = 3


# ---------------------------------------------------------------------------
# Помощни функции за LaTeX
# ---------------------------------------------------------------------------
def brace(s: str, i: int) -> tuple[str, int]:
    """Връща съдържанието на {...}, започващо от позиция i, и позицията след него."""
    while s[i] in " \n":
        i += 1
    if s[i] != "{":
        raise ValueError(s[i:i + 40])
    depth, j = 0, i
    while True:
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1


def bracket(s: str, i: int) -> tuple[str | None, int]:
    if i < len(s) and s[i] == "[":
        j = s.index("]", i)
        return s[i + 1:j], j + 1
    return None, i


def sub_cmd(s: str, name: str, fn, nargs: int = 1, opt: bool = False) -> str:
    """Заменя \\name[opt]{a1}...{an} с fn(opt, a1, ..., an)."""
    pat = re.compile(r"\\" + name + r"(?![A-Za-z])\*?")
    out, pos = [], 0
    while True:
        m = pat.search(s, pos)
        if not m:
            out.append(s[pos:])
            return "".join(out)
        out.append(s[pos:m.start()])
        j = m.end()
        o = None
        if opt:
            o, j = bracket(s, j)
        args = []
        try:
            for _ in range(nargs):
                a, j = brace(s, j)
                args.append(a)
        except (ValueError, IndexError):
            out.append(s[m.start():m.end()])
            pos = m.end()
            continue
        out.append(fn(o, *args) if opt else fn(*args))
        pos = j


def env_blocks(s: str, env: str):
    pat = re.compile(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", re.S)
    return pat


def strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in s.split("\n"))


def load_values() -> dict:
    txt = (LT / "generated" / "values.tex").read_text(encoding="utf-8")
    vals = {}
    for m in re.finditer(r"\\csname val@(.+?)\\endcsname", txt):
        v, _ = brace(txt, m.end())
        vals[m.group(1)] = v
    return vals


def load_labels() -> dict:
    txt = (LT / "main.aux").read_text(encoding="utf-8", errors="replace")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}", txt)}


# ---------------------------------------------------------------------------
# LaTeX -> Markdown
# ---------------------------------------------------------------------------
class Converter:
    def __init__(self):
        self.vals = load_values()
        self.labels = load_labels()
        self.blocks: dict[str, str] = {}      # маркер -> готов Markdown
        self.eq_count = 0
        self.fig_count = 0
        self.tab_count = 0
        self.ch = 0
        self.sec = 0
        self.subsec = 0
        self.appendix = False

    def val(self, k: str) -> str:
        v = self.vals.get(k, f"??{k}??")
        m = re.fullmatch(r"\$(.*)\$", v.strip())
        return r"\ensuremath{" + m.group(1) + "}" if m else v

    # --- маркери --------------------------------------------------------
    def mark(self, md: str) -> str:
        key = f"BLOCKMARK{len(self.blocks):04d}X"
        self.blocks[key] = md
        return f"\n\n{key}\n\n"

    # --- препратки ------------------------------------------------------
    def refname(self, key: str, cap: bool) -> str:
        n = self.labels.get(key, "?")
        kind = key.split(":")[0]
        words = {"fig": "фиг.", "tab": "табл.", "eq": "формула", "sec": "раздел", "ch": "глава",
                 "app": "приложение", "subsec": "раздел"}
        w = words.get(kind, "")
        if kind == "eq":
            txt = f"{w} ({n})"
        elif kind == "ch" and not n.isdigit():
            txt = f"приложение {n}"
        else:
            txt = f"{w} {n}"
        return txt[0].upper() + txt[1:] if cap else txt

    def cref(self, arg: str, cap: bool) -> str:
        keys = [k.strip() for k in arg.split(",")]
        if len(keys) > 1 and all(k.startswith("eq:") for k in keys):
            w = "Формули" if cap else "формули"
            return f"{w} " + " и ".join(f"({self.labels.get(k, '?')})" for k in keys)
        parts = [self.refname(k, cap and i == 0) for i, k in enumerate(keys)]
        return " и ".join(parts)

    # --- текст -----------------------------------------------------------
    def inline(self, s: str) -> str:
        """Общи замени, валидни навсякъде (вкл. надписи и клетки)."""
        for _ in range(3):
            s = sub_cmd(s, "val", lambda k: self.val(k))
        s = sub_cmd(s, "Cref", lambda a: self.cref(a, True))
        s = sub_cmd(s, "cref", lambda a: self.cref(a, False))
        s = sub_cmd(s, "ref", lambda a: self.labels.get(a, "?"))
        s = sub_cmd(s, "cite", lambda a: " CITE" + a.replace(" ", "").replace(",", "Q") + "ETIC")
        s = sub_cmd(s, "num", lambda a: re.sub(r"(\d)(?=(\d{3})+(?!\d))", "\\1\u00a0", a.replace(".", ",")))
        s = sub_cmd(s, "path", lambda a: r"\texttt{" + a.replace("_", r"\_").replace("<", r"\textless{}").replace(">", r"\textgreater{}") + "}")
        s = sub_cmd(s, "fitbox", lambda a: a)
        s = sub_cmd(s, "textsf", lambda a: a)
        s = sub_cmd(s, "color", lambda a: "")
        s = sub_cmd(s, "cellcolor", lambda a: "")
        s = sub_cmd(s, "rowcolor", lambda a: "")
        s = sub_cmd(s, "hd", lambda a: a)
        s = sub_cmd(s, "makecell", lambda a: re.sub(r"\\\\(\[[^\]]*\])?", " ", a))
        s = sub_cmd(s, "vspace", lambda a: "")
        s = sub_cmd(s, "thispagestyle", lambda a: "")
        s = sub_cmd(s, "addcontentsline", lambda a, b, c: "", nargs=3)
        s = sub_cmd(s, "markboth", lambda a, b: "", nargs=2)
        s = sub_cmd(s, "label", lambda a: "")
        for w in ["headrow", "centering", "noindent", "medskip", "smallskip", "bigskip", "clearpage", "newpage",
                  "hfill", "scriptsize", "footnotesize", "small", "sffamily", "bfseries", "Large", "large",
                  "normalsize", "par", "raggedright", "arraybackslash", "itshape"]:
            s = re.sub(r"\\" + w + r"(?![A-Za-z])\s?", " " if w == "par" else "", s)
        return s

    def cell(self, s: str) -> str:
        def clean(x):
            for k, v in {r"$\chi^2$": "χ²", r"$\Delta$": "Δ", r"$\theta$": "θ"}.items():
                x = x.replace(k, v)
            x = re.sub(r"\$([A-Za-z])\$", r"\1", x)
            x = x.replace(r"$+$", "+").replace(r"$-$", "−").replace(r"$^{\times}$", " ×").replace(r"$\geq$", "≥")
            x = re.sub(r"\\ensuremath\{(-?\d+)\{,\}(\d+)\}", r"\1,\2", x)
            return re.sub(r"\$(-?\d+)\{,\}(\d+)\$", r"\1,\2", x)          # $0{,}9079$ -> 0,9079
        s = clean(self.inline(clean(s))).strip()
        return self.to_md(s).replace("\n", " ").replace("|", "\\|").strip()

    def to_md(self, latex: str) -> str:
        r = subprocess.run(["pandoc", "-f", "latex+raw_tex", "-t", "markdown-raw_tex-simple_tables-multiline_tables-grid_tables",
                            "--wrap=none"], input=MACROS + latex, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("pandoc: " + r.stderr[:2000] + "\n" + latex[:500])
        return r.stdout.strip()

    # --- таблици ---------------------------------------------------------
    def parse_tabular(self, tex: str) -> tuple[list[str], list[list[str]], str]:
        m = re.search(r"\\begin\{(tabularx?)\}", tex)
        env = m.group(1)
        j = m.end()
        if env == "tabularx":
            _, j = brace(tex, j)
        _, j = brace(tex, j)
        end = tex.index(r"\end{" + env + "}", j)
        body = tex[j:end]
        after = tex[end + len(r"\end{" + env + "}"):]
        body = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", body)
        body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
        rows = []
        for raw in re.split(r"\\\\(?:\[[^\]]*\])?", self._protect(body)):
            raw = self._unprotect(raw).strip()
            if not raw:
                continue
            cells = re.split(r"(?<!\\)&", self._protect(raw))
            out = []
            for c in cells:
                c = self._unprotect(c)
                mc = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}", c)
                if mc:
                    inner, _ = brace(c, mc.end())
                    out.append(inner)
                    out += [""] * (int(mc.group(1)) - 1)
                    continue
                mr = re.match(r"\s*\\multirow\{[^}]*\}\{[^}]*\}", c)
                if mr:
                    c, _ = brace(c, mr.end())
                out.append(c)
            rows.append([self.cell(c) for c in out])
        note = ""
        k = after.find("{\\footnotesize")
        if k >= 0:
            note = self.inline(brace(after, k)[0]).strip()
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        return rows[0], rows[1:], note

    def _protect(self, s: str) -> str:
        # \\ и & вътре в {...} (напр. \makecell) не разделят редове/клетки
        out, depth = [], 0
        i = 0
        while i < len(s):
            c = s[i]
            if c == "\\" and i + 1 < len(s):
                pair = s[i:i + 2]
                if depth > 0 and pair == "\\\\":
                    out.append("\x01\x01")
                else:
                    out.append(pair)
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == "&" and depth > 0:
                c = "\x02"
            out.append(c)
            i += 1
        return "".join(out)

    @staticmethod
    def _unprotect(s: str) -> str:
        return s.replace("\x01\x01", "\\\\").replace("\x02", "&")

    def md_table(self, header, rows, caption) -> str:
        lens = [max([len(header[k])] + [len(r[k]) for r in rows]) for k in range(len(header))]
        tot = sum(lens) or 1
        dashes = [max(3, round(60 * l / tot)) for l in lens]
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("-" * d for d in dashes) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines) + f"\n\n: {caption}"

    def table_env(self, inner: str) -> str:
        self.tab_count += 1
        cap, lab = "", None
        mc = re.search(r"\\caption", inner)
        cap, _ = brace(inner, mc.end())
        ml = re.search(r"\\label\{(tab:[^}]+)\}", inner)
        lab = ml.group(1) if ml else None
        num = self.labels.get(lab, str(self.tab_count))
        interp = ""
        mi = re.search(r"\\interprettab", inner)
        if mi:
            interp, _ = brace(inner, mi.end())
            inner = inner[:mi.start()]
        mi2 = re.search(r"\\input\{generated/([^}]+)\}", inner)
        tex = (LT / "generated" / (mi2.group(1) + ".tex")).read_text(encoding="utf-8") if mi2 else inner
        header, rows, note = self.parse_tabular(tex)
        caption = self.to_md(self.inline(cap)).replace("\n", " ")
        md = self.md_table(header, rows, f"Таблица {num}. {caption}")
        if note:
            md += "\n\n::: {custom-style=\"Table Note\"}\n" + self.to_md(note) + "\n:::"
        out = self.mark(md)
        if interp:
            out += self.interp_block("Как се чете таблицата.", interp)
        return out

    # --- фигури -----------------------------------------------------------
    def figure_env(self, inner: str) -> str:
        self.fig_count += 1
        imgs = []
        if "tikzpicture" in inner:
            imgs.append((str(BUILD / "architecture.png"), 1.0))
        for m in re.finditer(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", inner):
            opt = m.group(1) or ""
            w = re.search(r"width=([\d.]+)\\textwidth", opt)
            imgs.append((str(FIG / m.group(2)), float(w.group(1)) if w else 0.9))
        sub = len(re.findall(r"\\begin\{subfigure\}", inner))
        main = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", inner, flags=re.S)
        mc = list(re.finditer(r"\\caption(?![a-z])", main))
        cap, _ = brace(main, mc[-1].end())
        ml = re.search(r"\\label\{(fig:[^}]+)\}", main)
        num = self.labels.get(ml.group(1) if ml else "", str(self.fig_count))
        interp = ""
        mi = re.search(r"\\interpret(?![a-z])", main)
        if mi:
            interp, _ = brace(main, mi.end())
        caption = self.to_md(self.inline(cap)).replace("\n", " ")
        if sub:
            subcaps = [brace(b, b.index(r"\caption") + 8)[0] for b in re.findall(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", inner, re.S)]
            caption += " (" + "; ".join(f"{chr(1072 + i)}) {self.inline(c)}" for i, c in enumerate(subcaps)) + ")"
            row = " ".join(f"![]({p}){{width=7.9cm}}" for p, _ in imgs)
            md = row
        else:
            md = "\n\n".join(f"![]({p}){{width={min(16.0, 16.0 * w / 0.9):.1f}cm}}" for p, w in imgs)
        md = "::: {custom-style=\"Figure\"}\n" + md + "\n:::\n\n::: {custom-style=\"Image Caption\"}\n" \
             + f"Фиг. {num}. {caption}\n:::"
        out = self.mark(md)
        if interp:
            out += self.interp_block("Какво показва фигурата.", interp)
        return out

    def interp_block(self, lead: str, text: str) -> str:
        body = self.to_md(self.inline(text)).replace("\n\n", " ")
        return self.mark("::: {custom-style=\"Interpretation\"}\n**" + lead + "** " + body + "\n:::")

    # --- формули ----------------------------------------------------------
    def equation(self, body: str) -> str:
        ml = re.search(r"\\label\{(eq:[^}]+)\}", body)
        num = self.labels.get(ml.group(1)) if ml else None
        body = re.sub(r"\\label\{[^}]+\}", "", body).strip()
        return self.mark(f"$${self.inline_math(body)}$$ EQNO{num}X")

    def align(self, body: str) -> str:
        out = []
        for row in re.split(r"\\\\(?:\[[^\]]*\])?\s*\n", body):
            row = row.strip()
            if not row:
                continue
            ml = re.search(r"\\label\{(eq:[^}]+)\}", row)
            num = self.labels.get(ml.group(1)) if ml else None
            row = re.sub(r"\\label\{[^}]+\}", "", row).replace("&", "").strip().rstrip("\\").strip()
            out.append(self.mark(f"$${self.inline_math(row)}$$ EQNO{num}X"))
        return "".join(out)

    def inline_math(self, s: str) -> str:
        s = sub_cmd(s, "val", lambda k: self.vals.get(k, k).strip("$"))
        s = sub_cmd(s, "txt", lambda a: r"\text{" + a + "}")
        s = s.replace("\\Bigl", "\\left").replace("\\Bigr", "\\right")
        return re.sub(r"\s*\n\s*", " ", s)

    # --- кутии --------------------------------------------------------------
    def box(self, title: str | None, body: str, style: str) -> str:
        body = self.inline(body).strip()
        md = self.to_md(body)
        lead = f"**{title}.** " if title else ""
        return self.mark(f"::: {{custom-style=\"{style}\"}}\n{lead}{md}\n:::")

    # --- заглавия ----------------------------------------------------------
    def heading(self, level: str, star: bool, title: str) -> str:
        title = self.to_md(self.inline(title)).replace("\n", " ")
        if level == "chapter":
            if star:
                return f"\n\n# {title} {{.unnumbered}}\n\n"
            self.ch += 1
            self.sec = self.subsec = 0
            if self.appendix:
                letter = APPENDIX_LETTERS[self.ch - 1]
                self.chap_label = letter
                return f"\n\n# Приложение {letter}. {title}\n\n"
            self.chap_label = str(self.ch)
            return f"\n\n# {self.ch}. {title}\n\n"
        if level == "section":
            if star:
                return f"\n\n## {title}\n\n"
            self.sec += 1
            self.subsec = 0
            return f"\n\n## {self.chap_label}.{self.sec}. {title}\n\n"
        self.subsec += 1
        return f"\n\n### {self.chap_label}.{self.sec}.{self.subsec}. {title}\n\n"

    # --- глава --------------------------------------------------------------
    def chapter(self, name: str) -> str:
        s = strip_comments((LT / "chapters" / f"{name}.tex").read_text(encoding="utf-8"))
        if name == "09_appendix":
            self.appendix = True
            self.ch = 0
        s = s.replace(r"\begin{otherlanguage}{english}", "").replace(r"\end{otherlanguage}", "")
        s = re.sub(r"\{\\sffamily\\Large\\bfseries\\color\{navy\}\s*Abstract\}\\par\\smallskip", r"\\section*{Abstract}", s)
        s = re.sub(r"\\noindent\\textsf\{\\textbf\{\\color\{navy\}(.*?)\}\}", r"\\textbf{\1}", s)
        # плаващи обекти и формули -> маркери
        s = env_blocks(s, "figure").sub(lambda m: self.figure_env(m.group(1)), s)
        s = env_blocks(s, "table").sub(lambda m: self.table_env(m.group(1)), s)
        s = env_blocks(s, "equation").sub(lambda m: self.equation(m.group(1)), s)
        s = env_blocks(s, "align").sub(lambda m: self.align(m.group(1)), s)
        s = re.sub(r"\\begin\{keybox\}(?:\[(.*?)\])?(.*?)\\end\{keybox\}", lambda m: self.box(m.group(1), m.group(2), "Key Box"), s, flags=re.S)
        s = re.sub(r"\\begin\{notebox\}(?:\[(.*?)\])?(.*?)\\end\{notebox\}", lambda m: self.box(m.group(1), m.group(2), "Note Box"), s, flags=re.S)
        s = sub_cmd(s, "interprettab", lambda a: self.interp_block("Как се чете таблицата.", a))
        s = sub_cmd(s, "interpret", lambda a: self.interp_block("Какво показва фигурата.", a))
        s = re.sub(r"\\begin\{lstlisting\}(\[[^\]]*\])?\n?(.*?)\\end\{lstlisting\}",
                   lambda m: self.mark("```" + ("bash" if m.group(1) and "bash" in m.group(1) else "python") + "\n" + m.group(2).rstrip() + "\n```"), s, flags=re.S)
        # заглавия
        def head(m):
            lvl, star = m.group(1), bool(m.group(2))
            t, _ = brace(s_ref[0], m.end())
            return lvl, star, t
        out, pos = [], 0
        pat = re.compile(r"\\(chapter|section|subsection)(\*?)")
        s_ref = [s]
        while True:
            m = pat.search(s, pos)
            if not m:
                out.append(s[pos:])
                break
            out.append(s[pos:m.start()])
            t, j = brace(s, m.end())
            out.append(self.mark(self.heading(m.group(1), bool(m.group(2)), t).strip()))
            pos = j
        s = "".join(out)
        s = sub_cmd(s, "paragraph", lambda a: r"\textbf{" + a.rstrip() + ("" if a.rstrip().endswith((".", ":", "?", "!")) else ".") + "} ")
        s = self.inline(s)
        md = self.to_md(s)
        md = re.sub(r"(BLOCKMARK\d{4}X)", lambda m: "\n\n" + self.blocks[m.group(1)] + "\n\n", md)
        md = re.sub(r"(BLOCKMARK\d{4}X)", lambda m: "\n\n" + self.blocks[m.group(1)] + "\n\n", md)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip() + "\n"


# ---------------------------------------------------------------------------
# Цитиране след всеки абзац
# ---------------------------------------------------------------------------
def bibliography() -> dict:
    txt = strip_comments((LT / "chapters" / "08_references.tex").read_text(encoding="utf-8"))
    items = {}
    for m in re.finditer(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", txt, re.S):
        items[m.group(1)] = m.group(2).strip()
    return items


def cite_paragraphs(chapters: list[tuple[str, str]], bib: dict) -> tuple[str, list[str]]:
    order: list[str] = []

    def num(k):
        if k not in order:
            order.append(k)
        return order.index(k) + 1

    out_all = []
    for name, md in chapters:
        blocks = re.split(r"\n\s*\n", md)
        res = []
        for b in blocks:
            res.append(cite_block(b, name, num))
        out_all.append("\n\n".join(res))
    missing = [k for k in bib if k not in order]
    return "\n\n".join(out_all), order, missing


def refs_for(text: str, chapter: str) -> list[str]:
    explicit = []
    for m in re.finditer(r"CITE([a-z0-9Q]+)ETIC", text):
        explicit += m.group(1).split("Q")
    low = re.sub(r"CITE[a-z0-9Q]+ETIC", "", text).lower()
    low = re.sub(r"\(.*?\.png\)", "", low)
    kw = [ref for pat, ref in KEYWORDS if re.search(pat, low)]
    keys = []
    for k in explicit + kw:
        if k not in keys:
            keys.append(k)
    if not keys:
        keys = list(CHAPTER_DEFAULT[chapter])
    return keys[:MAX_REFS] if len(explicit) <= MAX_REFS else explicit


def cite_block(b: str, chapter: str, num) -> str:
    s = b.strip()
    if not s:
        return b
    skip = (s.startswith("#") or s.startswith("BLOCK") or s.startswith("```") or s.startswith("$$")
            or s.startswith("|") or s.startswith(": Таблица") or "custom-style=\"Figure\"" in s
            or "custom-style=\"Image Caption\"" in s or "custom-style=\"Table Note\"" in s)
    if s.startswith("|"):
        return re.sub(r"\s*CITE([a-z0-9Q]+)ETIC", lambda m: " " + ", ".join(f"[{n}]" for n in sorted(num(k) for k in m.group(1).split("Q"))), b)
    if s.startswith("::: {custom-style=\"Image Caption\"}") or skip:
        return re.sub(r"\s*CITE[a-z0-9Q]+ETIC", "", b)
    if s.startswith("|"):
        return b
    keys = refs_for(s, chapter)
    s = re.sub(r"\s*CITE[a-z0-9Q]+ETIC", "", s)
    # номерата се присвояват по реда на първо споменаване и се подреждат възходящо в групата
    tag = ", ".join(f"[{n}]" for n in sorted(num(k) for k in keys))
    if s.endswith(":::"):
        lines = s.split("\n")
        lines[-2] = lines[-2].rstrip() + " " + tag
        return "\n".join(lines)
    return s.rstrip() + " " + tag


# ---------------------------------------------------------------------------
# Сглобяване и pandoc
# ---------------------------------------------------------------------------
def render_architecture():
    tex = strip_comments((LT / "chapters" / "03_framework.tex").read_text(encoding="utf-8"))
    m = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, re.S)
    pre = (ROOT / "presentation" / "render_formulas.py").read_text(encoding="utf-8")
    preamble = re.search(r'PRE = r"""(.*?)"""', pre, re.S).group(1).replace("varwidth=24cm", "").replace(r"\begin{document}", r"\setsansfont{PT Sans}\newfontfamily\cyrillicfontsf{PT Sans}\begin{document}")
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "architecture.tex").write_text(preamble + m.group(0) + "\n\\end{document}\n", encoding="utf-8")
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "architecture.tex"], cwd=BUILD,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("architecture: " + r.stdout[-1500:])
    subprocess.run(["pdftoppm", "-png", "-r", "250", "-singlefile", "architecture.pdf", "architecture"], cwd=BUILD, check=True)


def latex_to_md_text(s: str) -> str:
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "markdown", "--wrap=none"], input=s, capture_output=True, text=True)
    return r.stdout.strip()


def build_markdown() -> tuple[str, list[str]]:
    conv = Converter()
    chapters = []
    for name in CHAPTERS:
        chapters.append((name, conv.chapter(name)))
        print("converted", name)
    bib = bibliography()
    body, order, missing = cite_paragraphs(chapters, bib)
    if missing:
        print("ВНИМАНИЕ: нецитирани източници:", missing)
    refs = ["# 6. Информационни източници", ""]
    for i, k in enumerate(order, 1):
        entry = latex_to_md_text(bib[k].replace("~", " "))
        refs.append(f"::: {{custom-style=\"Bibliography\"}}\n[{i}]\u2003{entry}\n:::\n")
    # същият номериран списък за слайда „Информационни източници“ в презентацията
    plain = [re.sub(r"[*_`\\]", "", latex_to_md_text(bib[k].replace("~", " "))).strip() for k in order]
    (ROOT / "presentation" / "references.json").write_text(json.dumps(plain, ensure_ascii=False, indent=1), encoding="utf-8")
    # заключение и приложенията са преди списъка с източници, който е последен
    md = body + "\n\n" + "\n".join(refs)
    md = md.replace("$^{\\times}$", "×")
    return md, order


def make_reference_doc() -> Path:
    ref = BUILD / "reference.docx"
    subprocess.run(["pandoc", "-o", str(ref), "--print-default-data-file", "reference.docx"], check=True)
    return ref


def run_pandoc(md: str, ref: Path) -> Path:
    src = BUILD / "report.md"
    src.write_text(MACROS.strip() + "\n\n" + md, encoding="utf-8")
    out = BUILD / "body.docx"
    subprocess.run(["pandoc", str(src), "-f", "markdown+tex_math_dollars", "-o", str(out),
                    f"--reference-doc={ref}", "--resource-path", str(ROOT)], check=True)
    return out


# ---------------------------------------------------------------------------
# Пост-обработка с python-docx
# ---------------------------------------------------------------------------
def set_font(style, name, size=None, bold=None, color=None, italic=None):
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), name)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    if color:
        style.font.color.rgb = RGBColor.from_string(color)


def para_border(pf_element, side="left", color=ACCENT, sz=18, space=8, fill=None):
    ppr = pf_element.get_or_add_pPr()
    bdr = ppr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = OxmlElement("w:pBdr")
        ppr.append(bdr)
    el = OxmlElement(f"w:{side}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), str(space))
    el.set(qn("w:color"), color)
    bdr.append(el)
    if fill:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        ppr.append(shd)


class _Styles:
    def __init__(self, styles):
        self.by = {x.name: x for x in styles}

    def __getitem__(self, k):
        return self.by[k]

    def __iter__(self):
        return iter(self.by.values())


def style_document(doc: Document):
    """Стилове по шаблона: Times New Roman 12 pt, двустранно подравняване, отстъп 1,25 cm."""
    st = _Styles(doc.styles)
    names = [s.name for s in st]
    for name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
        if name in names:
            set_font(st[name], BODY_FONT, 12, color="000000")
    for name in ["Body Text", "First Paragraph"]:
        pf = st[name].paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.space_before, pf.space_after = Pt(0), Pt(3)
        pf.line_spacing = 1.15
        pf.first_line_indent = Cm(1.25)
    st["Compact"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    st["Compact"].paragraph_format.space_after = Pt(2)
    sizes = {"Heading 1": 16, "Heading 2": 14, "Heading 3": 13}
    for name, size in sizes.items():
        s = st[name]
        set_font(s, HEAD_FONT, size, True, "000000", False)
        pf = s.paragraph_format
        pf.space_before = Pt({"Heading 1": 24, "Heading 2": 16, "Heading 3": 12}[name])
        pf.space_after = Pt({"Heading 1": 12, "Heading 2": 6, "Heading 3": 4}[name])
        pf.line_spacing = 1.5 if name != "Heading 3" else 1.15
        pf.keep_with_next = True
        pf.first_line_indent = Cm(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    st["Heading 1"].font.all_caps = True
    st["Heading 1"].paragraph_format.page_break_before = True
    for name in ["Title", "Subtitle"]:
        if name in names:
            set_font(st[name], HEAD_FONT, color="000000")
    # надписи: удебелени, центрирани, черни (фигура отдолу, таблица отгоре)
    for name, size in [("Image Caption", 12), ("Table Caption", 11)]:
        s = st[name]
        set_font(s, HEAD_FONT, size, True, "000000", False)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.space_before = Pt(4 if name == "Image Caption" else 10)
        s.paragraph_format.space_after = Pt(10 if name == "Image Caption" else 4)
        s.paragraph_format.keep_with_next = name == "Table Caption"
    s = st["Figure"]
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.paragraph_format.keep_with_next = True
    s.paragraph_format.space_before = Pt(8)
    s.paragraph_format.first_line_indent = Cm(0)
    s = st["Interpretation"]
    set_font(s, BODY_FONT, 12, color="000000", italic=False)
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s.paragraph_format.first_line_indent = Cm(1.25)
    s.paragraph_format.space_after = Pt(8)
    s.paragraph_format.line_spacing = 1.15
    s = st["Table Note"]
    set_font(s, BODY_FONT, 10, color="404040", italic=False)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.first_line_indent = Cm(0)
    for name in ("Key Box", "Note Box"):
        s = st[name]
        set_font(s, BODY_FONT, 12, color="000000")
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        s.paragraph_format.left_indent = Cm(0.5)
        s.paragraph_format.right_indent = Cm(0.5)
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.space_before = Pt(6)
        s.paragraph_format.space_after = Pt(8)
        s.paragraph_format.line_spacing = 1.15
        para_border(s.element, "left", "000000", 6, 8)
    s = st["Bibliography"]
    set_font(s, BODY_FONT, 11, color="000000")
    s.paragraph_format.left_indent = Cm(1.0)
    s.paragraph_format.first_line_indent = Cm(-1.0)
    s.paragraph_format.space_after = Pt(3)
    s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if "Source Code" in names:
        sc = st["Source Code"]
        sc.paragraph_format.space_before = Pt(4)
        sc.paragraph_format.space_after = Pt(8)
        sc.paragraph_format.first_line_indent = Cm(0)
        para_border(sc.element, "left", "808080", 6, 6, "F4F4F4")
    if "Verbatim Char" in names:
        set_font(st["Verbatim Char"], "Courier New", 9)
    for name in ["Hyperlink"]:
        if name in names:
            st[name].font.color.rgb = RGBColor.from_string("1F3F8F")


def number_equations(doc: Document):
    sect = doc.sections[0]
    width = sect.page_width - sect.left_margin - sect.right_margin
    n = 0
    paras = doc.paragraphs
    for i, p in enumerate(paras):
        omp = p._p.find(qn("m:oMathPara"))
        if omp is None:
            continue
        m = re.search(r"EQNO(\d+|None)X", p.text)
        if not m and i + 1 < len(paras):
            m = re.fullmatch(r"\s*EQNO(\d+|None)X\s*", paras[i + 1].text)
            if m:
                paras[i + 1]._p.getparent().remove(paras[i + 1]._p)
        if not m:
            continue
        n += 1
        # премахване на маркера и интервалите около него
        for r in p.runs:
            if "EQNO" in r.text or r.text.strip() == "":
                r._r.getparent().remove(r._r)
        omath = omp.find(qn("m:oMath"))
        idx = list(p._p).index(omp)
        p._p.remove(omp)
        tab1 = OxmlElement("w:r"); t = OxmlElement("w:tab"); tab1.append(t)
        p._p.insert(idx, tab1)
        p._p.insert(idx + 1, omath)
        if m.group(1) != "None":
            run = p.add_run("\t(" + m.group(1) + ")")
            run.font.name = BODY_FONT
        p.paragraph_format.tab_stops.add_tab_stop(int(width / 2), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_together = True
    return n


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def set_col_widths(t, total: int, size: float = 9):
    """Ширини на колоните по съдържанието (алгоритъм min/max като при HTML таблиците)."""
    rows = t.rows
    ncols = len(t.columns)
    numeric = re.compile(r"^[\d\s,.%+−\-–×≥\[\];()]+$")
    need, desire = [3.0] * ncols, [3.0] * ncols
    for j in range(ncols):
        for row in rows:
            try:
                txt = re.sub(r"\s+", " ", row.cells[j].text.strip())
            except IndexError:
                continue
            toks = [txt] if numeric.match(txt or "x") and len(txt) <= 18 else txt.split(" ")
            need[j] = max(need[j], min(max((len(w) for w in toks), default=1), 20) + 3)
            desire[j] = max(desire[j], len(txt) + 3)
    C = total / (Pt(size) * 0.58)          # приблизителен брой знаци в реда
    if sum(desire) <= C:
        w = [d * C / sum(desire) for d in desire]
    elif sum(need) >= C:
        w = [n * C / sum(need) for n in need]
    else:
        k = (C - sum(need)) / (sum(desire) - sum(need))
        w = [n + (d - n) * k for n, d in zip(need, desire)]
    tw = [int(total * x / sum(w)) for x in w]
    grid = t._tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
        if j < ncols:
            gc.set(qn("w:w"), str(tw[j] // 635))
    tblpr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblpr.append(lay)
    for row in rows:
        for j, c in enumerate(row.cells):
            if j < ncols:
                c.width = tw[j]


def style_tables(doc: Document):
    """Таблици по шаблона: пълна мрежа с тънки черни линии, удебелен заглавен ред."""
    sect = doc.sections[0]
    width = sect.page_width - sect.left_margin - sect.right_margin
    for t in doc.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblpr = t._tbl.tblPr
        for el in tblpr.findall(qn("w:tblW")):
            tblpr.remove(el)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), "5000"); tw.set(qn("w:type"), "pct"); tblpr.append(tw)
        for el in tblpr.findall(qn("w:tblBorders")):
            tblpr.remove(el)
        borders = OxmlElement("w:tblBorders")
        for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
            borders.append(e)
        tblpr.append(borders)
        ncols = len(t.columns)
        size = 10 if ncols <= 6 else 9
        set_col_widths(t, int(width), size)
        for i, row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); trpr.append(cs)
            if i == 0:
                th = OxmlElement("w:tblHeader"); trpr.append(th)
            for c in row.cells:
                if i == 0:
                    shade(c, "F2F2F2")
                for p in c.paragraphs:
                    p.paragraph_format.space_before = Pt(1.5)
                    p.paragraph_format.space_after = Pt(1.5)
                    p.paragraph_format.first_line_indent = Cm(0)
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(size)
                        r.font.name = BODY_FONT
                        r.font.color.rgb = RGBColor(0, 0, 0)
                        if i == 0:
                            r.font.bold = True


def add_field(paragraph, instr: str, placeholder: str = ""):
    def fc(t):
        r = OxmlElement("w:r"); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), t); r.append(f); return r
    paragraph._p.append(fc("begin"))
    r = OxmlElement("w:r"); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r.append(it)
    paragraph._p.append(r)
    paragraph._p.append(fc("separate"))
    run = paragraph.add_run(placeholder)
    paragraph._p.append(fc("end"))
    return run


def header_footer(doc: Document):
    """A4, полета 2 / 2 / 2,5 / 2 cm; номер на страницата долу вдясно, без номер на заглавната страница."""
    sect = doc.sections[0]
    sect.page_height, sect.page_width = Cm(29.7), Cm(21.0)
    sect.left_margin, sect.right_margin = Cm(2.5), Cm(2.0)
    sect.top_margin, sect.bottom_margin = Cm(2.0), Cm(2.0)
    sect.header_distance = sect.footer_distance = Cm(1.25)
    sect.different_first_page_header_footer = True
    sect.header.paragraphs[0].text = ""
    fp = sect.footer.paragraphs[0]
    fp.text = ""
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.first_line_indent = Cm(0)
    run = add_field(fp, "PAGE", "1")
    run.font.size, run.font.name = Pt(11), BODY_FONT


def insert_before(anchor, text="", style=None, **fmt):
    p = anchor.insert_paragraph_before(text, style)
    return p


def title_page(doc: Document, first):
    """Заглавна страница по шаблона: университет с герб и линия, вид на работата, тема, данни за авторите."""
    def P(text="", size=12, bold=False, space_after=0, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0):
        p = first.insert_paragraph_before()
        p.alignment = align
        pf = p.paragraph_format
        pf.space_after, pf.space_before, pf.first_line_indent, pf.line_spacing = Pt(space_after), Pt(space_before), Cm(0), 1.0
        if text:
            r = p.add_run(text)
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, BODY_FONT
            r.font.color.rgb = RGBColor(0, 0, 0)
        return p
    # герб и наименование на университета на един ред, линия отдолу
    p = P(align=WD_ALIGN_PARAGRAPH.LEFT, space_after=6)
    if LOGO.exists():
        p.add_run().add_picture(str(LOGO), height=Cm(1.75))
    r = p.add_run("   ТЕХНИЧЕСКИ УНИВЕРСИТЕТ – СОФИЯ")
    r.font.size, r.font.bold, r.font.name = Pt(16), True, BODY_FONT
    r.font.color.rgb = RGBColor(0, 0, 0)
    para_border(p._p, "bottom", "000000", 6, 4)
    P(space_after=80)
    P("КУРСОВ ПРОЕКТ", 20, True, space_after=24)
    p = P(space_after=8)
    r = p.add_run("Тема: "); r.font.size, r.font.name = Pt(16), BODY_FONT
    r = p.add_run("„Експериментално изследване на откриване на проникване в мрежов трафик на център за данни, базирано на машинно обучение“")
    r.font.size, r.font.bold, r.font.name = Pt(16), True, BODY_FONT
    P("Атаки Distributed Denial of Service (DDoS), Brute Force (BruteForce) и Botnet (Bot) · Random Forest, SVM и XGBoost · реален набор от данни CSE-CIC-IDS2018", 12, space_after=60)
    rows = [("Дисциплина:", "Интелигентни системи за сигурност на екосистемите на големите данни и Интернет на нещата"),
            ("Изготвили:", "Иван Терзиев, Fak No. 961326004"),
            ("", "Кристиян Януцев, Fak No. 961326003"),
            ("", THIRD_MEMBER),
            ("Факултет:", "ФПМИ"),
            ("Специалност:", "АГМПД"),
            ("ОКС:", "магистър")]
    tbl = doc.add_table(rows=len(rows), cols=2)
    first._p.addprevious(tbl._tbl)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    for gc, wcm in zip(tbl._tbl.tblGrid.findall(qn("w:gridCol")), (3.6, 12.4)):
        gc.set(qn("w:w"), str(int(wcm * 567)))
    for i, (k, v) in enumerate(rows):
        for j, txt in enumerate((k, v)):
            c = tbl.cell(i, j)
            c.width = Cm(3.6) if j == 0 else Cm(12.4)
            c.text = txt
            for pp in c.paragraphs:
                pp.paragraph_format.first_line_indent = Cm(0)
                pp.paragraph_format.space_after = Pt(6)
                for rr in pp.runs:
                    rr.font.size, rr.font.name, rr.font.bold = Pt(12), BODY_FONT, j == 0
    P(space_after=90)
    P("София 2026", 14, True)
    pb = first.insert_paragraph_before()
    pb.add_run().add_break(WD_BREAK.PAGE)


def _fld(t):
    r = OxmlElement("w:r"); f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), t); r.append(f); return r


def _instr(text):
    r = OxmlElement("w:r"); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
    it.text = text; r.append(it); return r


def bookmark_headings(doc: Document) -> dict:
    """Поставя отметки _TocNNNN върху заглавията от ниво 1 и 2 (като в Word)."""
    marks, n = {}, 0
    for p in doc.paragraphs:
        if p.style.name in ("Heading 1", "Heading 2") and p.text.strip():
            n += 1
            name = f"_Toc{100000 + n}"
            st = OxmlElement("w:bookmarkStart"); st.set(qn("w:id"), str(900000 + n)); st.set(qn("w:name"), name)
            en = OxmlElement("w:bookmarkEnd"); en.set(qn("w:id"), str(900000 + n))
            ppr = p._p.find(qn("w:pPr"))
            (ppr.addnext if ppr is not None else (lambda e: p._p.insert(0, e)))(st)
            p._p.append(en)
            marks.setdefault(p.text.strip(), name)
    return marks


def toc_styles(doc: Document):
    """Стилове TOC Heading, TOC 1 и TOC 2 (вградените стилове на Word за съдържание)."""
    from docx.enum.style import WD_STYLE_TYPE
    sect = doc.sections[0]
    width = sect.page_width - sect.left_margin - sect.right_margin
    names = {s.name for s in doc.styles}
    for name, size, bold, indent, before in (("toc 1", 11, True, 0, 4), ("toc 2", 11, False, 0.6, 0)):
        st = doc.styles[name] if name in names else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = doc.styles["Normal"]
        set_font(st, BODY_FONT, size, bold=bold, color="000000", italic=False)
        st.font.all_caps = name == "toc 1"
        pf = st.paragraph_format
        pf.left_indent, pf.first_line_indent = Cm(indent), Cm(0)
        pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(1), 1.0
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        st.element.get_or_add_pPr()
        ui = OxmlElement("w:uiPriority"); ui.set(qn("w:val"), "39")
        st.element.find(qn("w:pPr")).addprevious(ui)
    name = "TOC Heading"
    st = doc.styles[name] if name in names else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    st.base_style = doc.styles["Normal"]
    st.font.name, st.font.size, st.font.bold, st.font.italic, st.font.all_caps = BODY_FONT, Pt(14), True, False, False
    set_font(st, BODY_FONT, 14, bold=True, color="000000", italic=False)
    st.paragraph_format.space_before = Pt(0)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    st.paragraph_format.first_line_indent = Cm(0)
    st.paragraph_format.space_after = Pt(12)


def toc(doc: Document, before, entries: list[tuple[int, str, str]], marks: dict):
    """Истинско съдържание на Word (Table of Contents): поле TOC, стилове TOC 1/TOC 2,
    хипервръзки към заглавията и полета PAGEREF за страниците. В Word се обновява с
    десен бутон -> Update Field (или References -> Update Table)."""
    toc_styles(doc)
    h = before.insert_paragraph_before("Съдържание", style="TOC Heading")
    h.paragraph_format.page_break_before = True
    paras = []
    for i, (level, text, page) in enumerate(entries):
        p = before.insert_paragraph_before(style="toc 1" if level == 1 else "toc 2")
        if i == 0:
            p._p.append(_fld("begin")); p._p.append(_instr(' TOC \\o "1-2" \\h \\z \\u ')); p._p.append(_fld("separate"))
        anchor = marks.get(text)
        hl = OxmlElement("w:hyperlink")
        if anchor:
            hl.set(qn("w:anchor"), anchor); hl.set(qn("w:history"), "1")
        def run(t):
            r = OxmlElement("w:r"); x = OxmlElement("w:t"); x.set(qn("xml:space"), "preserve"); x.text = t; r.append(x); return r
        hl.append(run(text))
        r = OxmlElement("w:r"); r.append(OxmlElement("w:tab")); hl.append(r)
        if anchor:
            hl.append(_fld("begin")); hl.append(_instr(f" PAGEREF {anchor} \\h ")); hl.append(_fld("separate"))
            hl.append(run(page)); hl.append(_fld("end"))
        else:
            hl.append(run(page))
        p._p.append(hl)
        paras.append(p)
    paras[-1]._p.append(_fld("end"))
    # обвивка като в Word (References -> Table of Contents): блок „Table of Contents“
    sdt = OxmlElement("w:sdt"); pr = OxmlElement("w:sdtPr")
    dpo = OxmlElement("w:docPartObj"); g = OxmlElement("w:docPartGallery"); g.set(qn("w:val"), "Table of Contents")
    dpo.append(g); dpo.append(OxmlElement("w:docPartUnique")); pr.append(dpo); sdt.append(pr)
    content = OxmlElement("w:sdtContent"); sdt.append(content)
    h._p.addprevious(sdt)
    for el in [h._p] + [q._p for q in paras]:
        content.append(el)


def no_italic(doc: Document):
    """Без курсив никъде: стилове, текстови тирове и променливите във формулите (m:sty = p)."""
    for st in doc.styles:
        try:
            if st.font.italic:
                st.font.italic = False
        except Exception:
            pass
    body = doc.element.body
    M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    for tag in ("w:i", "w:iCs"):
        for e in body.iter(qn(tag)):
            e.set(qn("w:val"), "0")
    for e in doc.styles.element.iter(qn("w:i")):
        e.set(qn("w:val"), "0")
    for e in doc.styles.element.iter(qn("w:iCs")):
        e.set(qn("w:val"), "0")
    for mr in body.iter(f"{{{M}}}r"):
        rpr = mr.find(f"{{{M}}}rPr")
        if rpr is None:
            rpr = OxmlElement("m:rPr"); mr.insert(0, rpr)
        for old in rpr.findall(f"{{{M}}}sty"):
            rpr.remove(old)
        if rpr.find(f"{{{M}}}nor") is not None:
            continue
        if rpr.find(f"{{{M}}}scr") is not None:      # калиграфски букви: само прав стил
            sty = OxmlElement("m:sty"); sty.set(qn("m:val"), "p"); rpr.find(f"{{{M}}}scr").addnext(sty)
            continue
        # „нормален текст“ в формулата: прав шрифт и в Word, и в LibreOffice/PDF
        rpr.append(OxmlElement("m:nor"))
        wr = mr.find(qn("w:rPr"))
        if wr is None:
            wr = OxmlElement("w:rPr"); rpr.addnext(wr)
        f = OxmlElement("w:rFonts")
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            f.set(qn(a), "Cambria Math")
        wr.insert(0, f)
        i0 = OxmlElement("w:i"); i0.set(qn("w:val"), "0"); wr.append(i0)


def headings(doc: Document) -> list[tuple[int, str]]:
    out = []
    for p in doc.paragraphs:
        if p.style.name in ("Heading 1", "Heading 2") and p.text.strip() and p.text != "Съдържание":
            out.append((1 if p.style.name == "Heading 1" else 2, p.text.strip()))
    return out


def postprocess(body: Path, pages: dict | None) -> Path:
    doc = Document(str(body))
    style_document(doc)
    header_footer(doc)
    neq = number_equations(doc)
    style_tables(doc)
    # картинки: ширина не повече от полезната
    first = doc.paragraphs[0]
    # абстрактът е първото заглавие: заглавната страница е преди него
    title_page(doc, first)
    doc.paragraphs[[i for i, p in enumerate(doc.paragraphs) if p.style.name == "Heading 1"][0]].paragraph_format.page_break_before = False
    # съдържанието е след абстракта (преди „Увод“)
    hs = headings(doc)
    entries = [(lvl, txt, (pages or {}).get(txt, "")) for lvl, txt in hs if txt not in ("Абстракт", "Abstract")]
    intro = next(p for p in doc.paragraphs if p.style.name == "Heading 1" and p.text.startswith("Увод"))
    marks = bookmark_headings(doc)
    toc(doc, intro, entries, marks)
    no_italic(doc)
    for p in doc.paragraphs:
        if p.style.name == "Heading 1" and p.text.strip() == "Abstract":
            p.paragraph_format.page_break_before = False
    out = BUILD / "final.docx"
    doc.core_properties.title = "Експериментално изследване на откриване на проникване в мрежов трафик на център за данни, базирано на машинно обучение"
    doc.core_properties.author = "Иван Терзиев, Кристиян Януцев" + ("" if THIRD_MEMBER.startswith("[") else ", " + THIRD_MEMBER.split(",")[0])
    doc.core_properties.language = "bg-BG"
    doc.save(str(out))
    print("формули:", neq)
    return out


def page_numbers(docx: Path, hs: list[tuple[int, str]]) -> dict:
    """Рендира документа с LibreOffice и намира страницата на всяко заглавие."""
    pdf_dir = BUILD / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pdf_dir), str(docx)],
                   capture_output=True, check=True, timeout=600)
    pdf = pdf_dir / (docx.stem + ".pdf")
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    texts = []
    for k in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(k), "-l", str(k), "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        texts.append(re.sub(r"\s+", " ", t).lower())
    pages, start = {}, 2
    for _, h in hs:
        key = re.sub(r"\s+", " ", h)[:40].lower()
        for k in range(start, n):
            if key in texts[k] and "съдържание" not in texts[k][:60] and texts[k].count("......") < 3:
                pages[h] = str(k + 1)
                start = k
                break
    return pages


def force_times(path: Path):
    """Целият документ с Times New Roman: всички w:rFonts (без формулите с Cambria Math)
    и шрифтовете на темата (заглавия и основен текст)."""
    import zipfile
    tmp = path.with_suffix(".tmp.docx")
    def fix_rfonts(m):
        tag = m.group(0)
        if "Cambria Math" in tag:
            return tag
        return '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                x = data.decode("utf-8")
                x = re.sub(r"<w:rFonts\b[^>]*/>", fix_rfonts, x)
                if item.filename.startswith("word/theme/"):
                    x = re.sub(r'(<a:(?:latin|ea|cs) typeface=")[^"]*(")', r"\1Times New Roman\2", x)
                    x = re.sub(r'(<a:font script="[^"]*" typeface=")[^"]*(")', r"\1Times New Roman\2", x)
                data = x.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)


def main():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    render_architecture()
    md, order = build_markdown()
    ref = make_reference_doc()
    body = run_pandoc(md, ref)
    first = postprocess(body, None)
    hs = headings(Document(str(body)))
    pages = page_numbers(first, [h for h in hs if h[1] not in ("Абстракт", "Abstract")])
    final = postprocess(body, pages)
    shutil.copy(final, OUT)
    force_times(OUT)
    subprocess.run(["zip", "-d", str(OUT), "*/"], capture_output=True)
    # PDF версията на доклада е същият Word документ, превърнат с LibreOffice
    pdf_dir = BUILD / "pdf_out"
    pdf_dir.mkdir(exist_ok=True)
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pdf_dir), str(OUT)],
                   capture_output=True, check=True, timeout=600)
    shutil.copy(pdf_dir / (OUT.stem + ".pdf"), OUT.with_suffix(".pdf"))
    print("written", OUT, "източници:", len(order))


if __name__ == "__main__":
    main()
