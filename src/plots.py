"""Графично представяне на резултатите (всички надписи са на български).

Цветово кодиране, еднакво във всички графики:
  Random Forest -> син,  SVM -> оранжев;  атака -> червен, нормален трафик -> сив.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .registry import ALGORITHMS

ALGO_COLOR = {"RF": "#2a78d6", "SVM": "#eb6834", "XGB": "#2e9e6a"}
ATTACK_COLOR = "#e34948"
BENIGN_COLOR = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


# 1. Разпределение на класовете --------------------------------------------------
def class_distribution(task_stats: dict):
    n = len(task_stats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.8), squeeze=False)
    axes = axes[0]
    for ax, (attack, st) in zip(axes, task_stats.items()):
        cats = ["Оригинален набор", "Балансирана извадка"]
        att = [st["attack_rows_original"], st["per_class_sample"]]
        ben = [st["benign_rows_original"], st["per_class_sample"]]
        x = np.arange(2)
        w = 0.36
        b1 = ax.bar(x - w / 2 - 0.01, ben, w, color=BENIGN_COLOR, label=f"Нормален трафик ({C.BENIGN_LABEL})")
        b2 = ax.bar(x + w / 2 + 0.01, att, w, color=ATTACK_COLOR, label="Атака")
        ax.set_yscale("log")
        ax.set_xticks(x, cats)
        ax.set_title(f"{attack} срещу нормален трафик")
        ax.set_ylabel("Брой записи (лог. скала)")
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(_fmt_int(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, color=INK2, xytext=(0, 2), textcoords="offset points")
        ax.grid(axis="x", visible=False)
    axes[0].legend(handles=[b1, b2], labels=[f"Нормален трафик ({C.BENIGN_LABEL})", "Атака"],
                   loc="upper center", bbox_to_anchor=(n / 2 + 0.05 * (n - 1), -0.12), ncol=2)
    fig.suptitle(f"Разпределение на класовете в {C.DATASET_NAME} и след балансиране", fontweight="bold", y=1.02)
    return _save(fig, "01_class_distribution.png")


# 2. PCA: кумулативна обяснена дисперсия ----------------------------------------
def pca_variance(curves: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    styles = {"DDoS": "-", "BruteForce": ":", "Bot": "--"}
    for attack, evr in curves.items():
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, C.PCA_VARIANCE) + 1)
        ax.plot(np.arange(1, len(cum) + 1), cum * 100, styles.get(attack, "-"), lw=2, color=INK,
                label=f"{attack}: {k} компонента за {int(C.PCA_VARIANCE*100)}%")
        ax.plot([k], [cum[k - 1] * 100], "o", ms=7, color=ATTACK_COLOR, mec="white", mew=1.5)
    ax.axhline(C.PCA_VARIANCE * 100, color=INK2, lw=1, ls=":")
    ax.text(1, C.PCA_VARIANCE * 100 + 1, f"праг {int(C.PCA_VARIANCE*100)}%", color=INK2, fontsize=8.5)
    ax.set_xlabel("Брой главни компоненти")
    ax.set_ylabel("Кумулативна обяснена дисперсия, %")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")
    ax.set_title("PCA - кумулативна обяснена дисперсия (след филтриране на признаците)")
    return _save(fig, "02_pca_explained_variance.png")


# 3. Корелационна матрица ------------------------------------------------------------
def correlation_heatmap(X: pd.DataFrame, attack: str, dropped: list, idx: int):
    corr = X.corr().abs()
    big = corr.shape[0] > 45
    fig, ax = plt.subplots(figsize=(14, 12.5) if big else (11, 9.5))
    sns.heatmap(corr, cmap="Blues", vmin=0, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "|коефициент на корелация на Пиърсън|", "shrink": 0.7},
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=4.6 if big else 6.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        if lbl.get_text() in dropped:
            lbl.set_color(ATTACK_COLOR)
            lbl.set_fontweight("bold")
    ax.set_title(f"Корелация между признаците - задача {attack}\n"
                 f"(в червено: признаци, премахнати при |r| > {C.CORR_THRESHOLD} или нулева дисперсия)")
    ax.grid(False)
    return _save(fig, f"03{'abcdef'[idx]}_correlation_{attack}.png")


# 4. Матрици на объркване -------------------------------------------------------------
def confusion_matrices(results: pd.DataFrame):
    attacks = list(dict.fromkeys(results.attack))
    algos = list(dict.fromkeys(results.algo))
    fig, axes = plt.subplots(len(attacks), len(algos), figsize=(4.4 * len(algos), 4 * len(attacks)), squeeze=False)
    order = [results[(results.attack == a) & (results.algo == g)].iloc[0] for a in attacks for g in algos]
    for ax, r in zip(axes.ravel(), order):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(pct, annot=False, cmap="Blues", vmin=0, vmax=100, cbar=False, ax=ax,
                    linewidths=2, linecolor="white")
        names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                col = "white" if pct[i, j] > 55 else INK
                ax.text(j + 0.5, i + 0.42, f"{names[i][j]} = {_fmt_int(cm[i, j])}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=col)
                ax.text(j + 0.5, i + 0.62, f"{pct[i, j]:.3f}% от реалния клас", ha="center", va="center",
                        fontsize=8.5, color=col)
        ax.set_xticklabels(["Нормален", "Атака"])
        ax.set_yticklabels(["Нормален", "Атака"], rotation=90, va="center")
        ax.set_xlabel("Предсказан клас")
        ax.set_ylabel("Реален клас")
        ax.set_title(f"{r.model}")
        ax.grid(False)
    fig.suptitle("Матрици на объркване в оперативната фаза (тестови данни, 30%)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_confusion_matrices.png")


# 5. Precision / Accuracy / Sensitivity ----------------------------------------------
def metric_bars(results: pd.DataFrame, metrics=("Precision", "Accuracy", "Sensitivity"), fname="05_metrics.png",
                title="Precision, Accuracy и Sensitivity на моделите"):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.4), sharey=True)
    attacks = list(dict.fromkeys(results.attack))
    lo = max(0.0, np.floor(results[list(metrics)].min().min() * 100 - 1) / 100)
    for ax, met in zip(np.atleast_1d(axes), metrics):
        x = np.arange(len(attacks))
        algos = list(dict.fromkeys(results.algo))
        w = 0.8 / len(algos)
        for k, algo in enumerate(algos):
            vals = [results[(results.attack == a) & (results.algo == algo)][met].iloc[0] for a in attacks]
            bars = ax.bar(x + (k - (len(algos) - 1) / 2) * (w + 0.01), [v - lo for v in vals], w, bottom=lo,
                          color=ALGO_COLOR[algo], label=ALGORITHMS[algo].name_bg)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v*100:.2f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                            fontsize=6.5, rotation=90 if len(algos) > 2 else 0, color=INK, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x, attacks)
        ax.set_title(met)
        ax.set_ylim(lo, 1.0 + (1 - lo) * 0.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].legend(loc="upper center", bbox_to_anchor=(1.5 if len(metrics) > 1 else 0.5, -0.1), ncol=len(algos))
    fig.suptitle(title + (f" (оста започва от {lo*100:.0f}%)" if lo > 0 else ""), fontweight="bold", y=1.02)
    return _save(fig, fname)


# 6. ROC криви ----------------------------------------------------------------------------
def roc_curves(rocs: dict, aucs: dict):
    attacks = list(dict.fromkeys(a for a, _ in rocs))
    algos = list(dict.fromkeys(g for _, g in rocs))
    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4.3), squeeze=False)
    for ax, attack in zip(axes[0], attacks):
        for algo in algos:
            fpr, tpr = rocs[(attack, algo)]
            ax.plot(np.clip(fpr, 1e-6, 1), tpr, lw=2, color=ALGO_COLOR[algo],
                    label=f"{ALGORITHMS[algo].name_bg} (AUC = {aucs[(attack, algo)]:.5f})")
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("Процент фалшиви аларми FPR (лог. скала)")
        ax.set_ylabel("Sensitivity (TPR)")
        ax.set_title(attack)
        ax.legend(loc="upper left")
    fig.suptitle("ROC криви: " + ", ".join(ALGORITHMS[g].name_bg for g in algos), fontweight="bold", y=1.02)
    return _save(fig, "06_roc_curves.png")


# 7. Важност на признаците -------------------------------------------------------------
def feature_importance(imps: dict, top: int = 15):
    fig, axes = plt.subplots(1, len(imps), figsize=(5.5 * len(imps), 5), squeeze=False)
    for ax, (attack, s) in zip(axes[0], imps.items()):
        s = s.sort_values(ascending=True).tail(top)
        ax.barh(s.index, s.values, color=ALGO_COLOR["RF"], height=0.7)
        ax.set_title(attack)
        ax.set_xlabel("Важност (средно намаление на Gini)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle(f"Топ {top} признака според Random Forest (пълен вектор на признаците)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "07_feature_importance.png")


# 8. Време за обучение и класификация: пълен срещу редуциран вектор ---------------
def timing(results: pd.DataFrame, baseline: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(max(11, 1.2 * len(results)), 8))
    labels = [f"{r.algo}\n{r.attack}" for _, r in results.iterrows()]
    x = np.arange(len(labels))
    w = 0.36
    for ax, col, ttl in [(axes[0], "fit_time_s", "Време за обучение, s"),
                         (axes[1], "predict_time_per_record_us", "Време за класификация на 1 запис, µs")]:
        full = [baseline[(baseline.attack == r.attack) & (baseline.algo == r.algo)][col].iloc[0] for _, r in results.iterrows()]
        red = results[col].tolist()
        ax.bar(x - w / 2 - 0.01, full, w, color="#c3c2b7", label="Пълен вектор (без редукция)")
        ax.bar(x + w / 2 + 0.01, red, w, color=[ALGO_COLOR[a] for a in results.algo], label=C.REDUCTION_LABEL)
        for xi, (f, r_) in enumerate(zip(full, red)):
            ax.annotate(f"{f:.3g}", (xi - w / 2, f), ha="center", va="bottom", fontsize=7.5, color=INK2, xytext=(0, 2), textcoords="offset points")
            ax.annotate(f"{r_:.3g}", (xi + w / 2, r_), ha="center", va="bottom", fontsize=7.5, color=INK, xytext=(0, 2), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.set_title(ttl + " (лог. скала)")
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.suptitle("Ефект на редукцията на размерността върху времето", fontweight="bold", y=1.02)
    return _save(fig, "08_timing.png")


# 9. Ефект на редукцията върху качеството (E1) -----------------------------------------
def reduction_effect(results: pd.DataFrame, baseline: pd.DataFrame, metric: str = "F1"):
    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(results)), 4.4))
    labels = [f"{ALGORITHMS[r.algo].name_bg}\n{r.attack}" for _, r in results.iterrows()]
    x = np.arange(len(labels))
    full = np.array([baseline[(baseline.attack == r.attack) & (baseline.algo == r.algo)][metric].iloc[0] for _, r in results.iterrows()])
    red = results[metric].to_numpy()
    lo = max(0.0, np.floor(min(full.min(), red.min()) * 100 - 1) / 100)
    w = 0.36
    ax.bar(x - w / 2 - 0.01, full - lo, w, bottom=lo, color="#c3c2b7", label="Пълен вектор")
    ax.bar(x + w / 2 + 0.01, red - lo, w, bottom=lo, color=[ALGO_COLOR[a] for a in results.algo], label=C.REDUCTION_LABEL)
    for xi in range(len(x)):
        ax.annotate(f"{full[xi]*100:.3f}%", (xi - w / 2, full[xi]), ha="center", va="bottom", fontsize=7.5, color=INK2, xytext=(0, 2), textcoords="offset points")
        ax.annotate(f"{red[xi]*100:.3f}%", (xi + w / 2, red[xi]), ha="center", va="bottom", fontsize=7.5, color=INK, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, labels)
    ax.set_ylim(lo, 1 + (1 - lo) * 0.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title(f"{metric}: пълен срещу редуциран вектор на признаците (E1)")
    return _save(fig, "09_reduction_effect.png")
