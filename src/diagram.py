"""Схема на експерименталния конвейер (фигура за доклада и презентацията)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import json

from . import config as C
from .registry import ALGORITHMS, ATTACKS

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#8a8984"
ML, OP = "#2a78d6", "#eb6834"


def box(ax, x, y, w, h, title, lines, color=EDGE, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.55 - i * 0.27, ln, ha="center", va="top", fontsize=8, color=INK2)


def arrow(ax, x1, y1, x2, y2, color=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=color))


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0.1, 6.0, "Експериментален конвейер: от суровите CSV файлове до метриките", fontsize=12,
            fontweight="bold", color=INK, va="top")

    w, h, y = 2.15, 1.8, 3.5
    st = {}
    sp = C.RESULTS_DIR / "dataset_stats.json"
    if sp.exists():
        st = json.loads(sp.read_text(encoding="utf-8"))
    n = lambda v: f"{v:,}".replace(",", " ") if isinstance(v, int) else "?"
    ld, cl = st.get("load", {}), st.get("clean", {})
    box(ax, 0.1, y, w, h, "1. Данни",
        [C.DATASET_2018, C.DATASET_2017, f"{ld.get('files', '?')} CSV файла (по ден)",
         f"{n(ld.get('rows_read_total'))} записа", f"{ld.get('n_features_raw', '?')} признака"])
    box(ax, 2.55, y, w, h, "2. Почистване", ["NaN / ±Inf", "дубликати", f"→ {n(cl.get('rows_after'))} записа"])
    box(ax, 5.0, y, w, h, "3. Бинарни задачи", [f"{k} срещу Benign" for k in ATTACKS] + ["балансирани 1:1"])
    box(ax, 7.45, y, w, h, "4. Разделяне", ["стратифицирано", "70% обучение", "30% тест"])
    for x in (2.25, 4.7, 7.15):
        arrow(ax, x, y + h / 2, x + 0.3, y + h / 2)

    # ML phase
    ax.add_patch(FancyBboxPatch((0.1, 0.25), 6.9, 2.95, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=ML, alpha=0.06, ec=ML, lw=1.2))
    ax.text(0.3, 3.05, "Фаза на машинно обучение (70%)", fontsize=10, fontweight="bold", color=ML, va="top")
    bw = 1.55
    xs = [0.3, 2.0, 3.7, 5.4]
    algos = [a.name_bg for a in ALGORITHMS.values()] + ["GridSearchCV, 5-CV"]
    if C.REDUCTION == "select":
        titles = ["FeatureFilter", "Подбор", "StandardScaler", "Класификатор"]
        lines = [["нулева дисперсия", "(константни", "признаци)"],
                 ["топ " + str(C.SELECT_K) + " признака", "по важност (XGB),", "само от обучението"],
                 ["средна 0,", "станд. откл. 1", ""], algos]
    else:
        titles = ["FeatureFilter", "StandardScaler", "PCA", "Класификатор"]
        lines = [["нулева дисперсия,", f"|r| > {C.CORR_THRESHOLD}", ""],
                 ["средна 0,", "станд. откл. 1", ""],
                 [f"{int(C.PCA_VARIANCE*100)}% дисперсия", "компоненти", ""], algos]
    for x, t, l in zip(xs, titles, lines):
        box(ax, x, 0.75, bw, 1.75, t, l, color=ML)
    for x in xs[:-1]:
        arrow(ax, x + bw, 1.62, x + bw + 0.15, 1.62, color=ML)
    ax.text(3.55, 0.42, "един sklearn Pipeline, обучаван само върху обучаващите данни → joblib",
            ha="center", fontsize=8.5, color=INK2)

    # Operational phase
    ax.add_patch(FancyBboxPatch((7.3, 0.25), 4.55, 2.95, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=OP, alpha=0.06, ec=OP, lw=1.2))
    ax.text(7.5, 3.05, "Оперативна фаза (30%)", fontsize=10, fontweight="bold", color=OP, va="top")
    box(ax, 7.5, 0.75, 1.95, 1.75, "Записан модел", ["класификация на", f"пакети по {C.OPERATIONAL_BATCH:,}".replace(",", " "), "невиждани записа"], color=OP)
    box(ax, 9.75, 0.75, 1.95, 1.75, "Метрики", ["TP, TN, FP, FN", "Precision, Accuracy,", "Sensitivity, F1, ROC"], color=OP)
    arrow(ax, 9.45, 1.62, 9.75, 1.62, color=OP)

    arrow(ax, 8.5, y, 3.0, 3.2, color=ML)
    arrow(ax, 8.9, y, 9.3, 3.2, color=OP)
    arrow(ax, 7.0, 1.62, 7.5, 1.62, color=INK2)

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / "00_pipeline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(pipeline_diagram())
