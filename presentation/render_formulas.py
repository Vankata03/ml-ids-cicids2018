"""Рендира формулите и архитектурната схема за презентацията чрез XeLaTeX -> PNG.

Номерата на формулите се вземат от report/latex/main.aux, така че съвпадат
с номерацията в доклада.  Изпълнение: python presentation/render_formulas.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "formulas"
AUX = ROOT / "report" / "latex" / "main.aux"
BUILD = OUT / "_build"

PRE = r"""\documentclass[border=6pt,varwidth=24cm]{standalone}
\usepackage{amsmath,mathtools}
\usepackage{fontspec}
\usepackage[math-style=upright,bold-style=upright]{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{polyglossia}\setdefaultlanguage{bulgarian}
\setmainfont{Carlito}\newfontfamily\cyrillicfont{Carlito}
\usepackage{xcolor}\definecolor{navy}{HTML}{14213D}\definecolor{accent}{HTML}{1D5FA8}
\definecolor{muted}{HTML}{52514E}\definecolor{orange}{HTML}{EB6834}\definecolor{boxbg}{HTML}{EEF3F8}
\usepackage{tikz}\usetikzlibrary{arrows.meta}
\newcommand{\vect}[1]{\symbf{#1}}\newcommand{\R}{\symbb{R}}
\DeclareMathOperator*{\argmax}{arg\,max}
\newcommand{\tg}[1]{\tag*{\color{muted}\small(#1)}}
\begin{document}\color{navy}
"""

FORMULAS = {
    "select": r"""\begin{minipage}{12.5cm}
\begin{align*}
s_j^2&=\tfrac{1}{n-1}\textstyle\sum_i(x_{ij}-\bar{x}_j)^2,\qquad \text{премахва се при } s_j^2\le 10^{-8} \tg{EQ:var}\\[4pt]
I_j&=\frac{1}{|\mathcal{R}_j|}\sum_{r\in\mathcal{R}_j}\operatorname{Gain}(r),\qquad \mathcal{J}^{*}=\operatorname{top}_k\{\tilde I_j\} \tg{EQ:gain-imp}\\[4pt]
z_{ij}&=\frac{x_{ij}-\bar{x}_j}{s_j} \tg{EQ:zscore}
\end{align*}\end{minipage}""",
    "xgb": r"""\begin{minipage}{12.5cm}
\begin{align*}
F(\vect{x})&=\sum_{k=1}^{K}\nu\,f_k(\vect{x}),\qquad \hat{p}(\vect{x})=\sigma\bigl(F(\vect{x})\bigr) \tg{EQ:xgb-model}\\[4pt]
\mathcal{L}^{(k)}&=\sum_{i}l\bigl(y_i,F^{(k-1)}(\vect{x}_i)+f_k(\vect{x}_i)\bigr)+\gamma T+\tfrac12\lambda\textstyle\sum_t w_t^2 \tg{EQ:xgb-obj}\\[4pt]
\operatorname{Gain}&=\tfrac12\Bigl[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Bigr]-\gamma \tg{EQ:xgb-gain}
\end{align*}\end{minipage}""",
    "onese": r"""\begin{minipage}{12.5cm}
\begin{align*}
\overline{\operatorname{F1}}(\symbf{\theta})&=\frac{1}{K}\sum_{k=1}^{K}\operatorname{F1}_k(\symbf{\theta}),\qquad \operatorname{SE}(\symbf{\theta})=\frac{s_{\operatorname{F1}}(\symbf{\theta})}{\sqrt{K}} \tg{EQ:cv}\\[4pt]
\symbf{\theta}^{*}&=\operatorname*{arg\,min}_{\symbf{\theta}}\bigl\{c(\symbf{\theta}):\ \overline{\operatorname{F1}}(\symbf{\theta})\ge\overline{\operatorname{F1}}(\hat{\symbf{\theta}})-\operatorname{SE}(\hat{\symbf{\theta}})\bigr\} \tg{EQ:one-se}
\end{align*}\end{minipage}""",
    "rf": r"""\begin{minipage}{12.5cm}
\begin{align*}
G(S_v)&=1-\sum_{c}p_{v,c}^2 \tg{EQ:gini}\\[4pt]
\hat{p}(\vect{x})&=\frac{1}{B}\sum_{b=1}^{B}\hat{p}_b(\vect{x}),\qquad \hat{y}=\mathbf{1}\left[\hat{p}(\vect{x})>0{,}5\right] \tg{EQ:rf}\\[4pt]
\operatorname{Var}\bigl[\hat{p}\bigr]&=\rho\,\sigma^2+\frac{1-\rho}{B}\,\sigma^2 \tg{EQ:rf-var}
\end{align*}\end{minipage}""",
    "svm": r"""\begin{minipage}{12.5cm}
\begin{align*}
&\min_{\vect{w},b,\symbf{\xi}}\ \tfrac12\|\vect{w}\|^2+C\sum_{i}\xi_i\quad\text{при}\quad t_i\bigl(\vect{w}^{\top}\phi(\vect{x}_i)+b\bigr)\ge 1-\xi_i \tg{EQ:svm-primal}\\[4pt]
&K(\vect{x},\vect{x}')=\exp\bigl(-\gamma\,\|\vect{x}-\vect{x}'\|^2\bigr) \notag\\[4pt]
&f(\vect{x})=\sum_{i:\,\alpha_i>0}\alpha_i t_i K(\vect{x}_i,\vect{x})+b,\qquad \hat{y}=\mathbf{1}[f(\vect{x})>0] \tg{EQ:rbf}
\end{align*}\end{minipage}""",
    "metrics": r"""\begin{minipage}{12.5cm}
\begin{align*}
\text{Accuracy}&=\frac{TP+TN}{TP+TN+FP+FN} \tg{EQ:accuracy}\\[3pt]
\text{Precision}&=\frac{TP}{TP+FP} \tg{EQ:precision}\\[3pt]
\text{Recall (Sensitivity)}&=\frac{TP}{TP+FN} \tg{EQ:sensitivity}\\[3pt]
\text{F1}&=\frac{2\,TP}{2\,TP+FP+FN} \tg{EQ:f1}\\[3pt]
\text{Specificity}&=\frac{TN}{TN+FP} \tg{EQ:spec}
\end{align*}\end{minipage}""",
    "baserate": r"""\begin{minipage}{11cm}
\begin{equation*}
\text{Precision}(\pi)=\frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)} \tg{EQ:baserate}
\end{equation*}\end{minipage}""",
    "stats": r"""\begin{minipage}{12.5cm}
\begin{align*}
\text{CI}_{95\%}(M)&=\bigl[\hat{M}^{*}_{(0{,}025)},\ \hat{M}^{*}_{(0{,}975)}\bigr] \tg{EQ:bootstrap}\\[4pt]
\chi^2&=\frac{(|b-c|-1)^2}{b+c}\sim\chi^2_1 \tg{EQ:mcnemar}\\[4pt]
D_j&=\sup_t\bigl|F_{1,j}(t)-F_{0,j}(t)\bigr| \tg{EQ:ks}
\end{align*}\end{minipage}""",
}


def eq_numbers() -> dict:
    if not AUX.exists():
        return {}
    txt = AUX.read_text(encoding="utf-8", errors="replace")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{eq:([^}@]+)\}\{\{(\d+)\}", txt)}


def render(name: str, body: str):
    BUILD.mkdir(parents=True, exist_ok=True)
    tex = BUILD / f"{name}.tex"
    tex.write_text(PRE + body + "\n\\end{document}\n", encoding="utf-8")
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=BUILD,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{name}: {r.stdout[-1500:]}")
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", f"{name}.pdf", str(OUT / name)], cwd=BUILD, check=True)
    from PIL import Image, ImageChops
    im = Image.open(OUT / f"{name}.png").convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if box:
        im.crop((max(box[0] - 20, 0), max(box[1] - 20, 0), min(box[2] + 20, im.width), min(box[3] + 20, im.height))).save(OUT / f"{name}.png")


def main():
    nums = eq_numbers()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in FORMULAS.items():
        body = re.sub(r"EQ:([A-Za-z0-9\-]+)", lambda m: nums.get(m.group(1), "?"), body)
        render(name, body)
        print("rendered", name)
    shutil.rmtree(BUILD, ignore_errors=True)


if __name__ == "__main__":
    main()
