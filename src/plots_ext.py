"""Графики към разширените анализи (фигури 12-23). Стилът е общ с src/plots.py."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .plots import ALGO_COLOR, ATTACK_COLOR, BENIGN_COLOR, GRID, INK, INK2, _fmt_int, _save
from .registry import ALGORITHMS, ATTACKS

PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=1)


def _name(algo):
    return ALGORITHMS[algo].name_bg



def _present(df: pd.DataFrame) -> list:
    """Атаките от регистъра, за които има резултати (в реда на регистъра)."""
    have = set(df["attack"])
    return [a for a in ATTACKS if a in have]

# 12. KS разделимост -----------------------------------------------------------------------
def separability(sep: pd.DataFrame, top: int = 15):
    attacks = _present(sep)
    fig, axes = plt.subplots(1, len(attacks), figsize=(11, 5.2), sharex=True)
    for ax, a in zip(np.atleast_1d(axes), attacks):
        s = sep[sep.attack == a].sort_values("KS_D").tail(top)
        ax.barh(s.feature, s.KS_D, color=ATTACK_COLOR, height=0.7)
        for y, v in enumerate(s.KS_D):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, 1.1)
        ax.set_title(f"{a} срещу нормален трафик")
        ax.set_xlabel("KS статистика D (0 = еднакви, 1 = напълно разделими)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle(f"Едномерна разделимост на признаците: топ {top} по тест на Колмогоров-Смирнов",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "12_ks_separability.png")


# 13. Разпределения на най-разделимите признаци --------------------------------------------
def feature_distributions(df: pd.DataFrame, sep: pd.DataFrame, per_attack: int = 4, n: int = 20_000):
    attacks = _present(sep)
    fig, axes = plt.subplots(len(attacks), per_attack, figsize=(12, 3.2 * len(attacks)))
    rng = np.random.default_rng(C.SEED)
    lbl = df[C.LABEL_COLUMN]
    ben_idx = rng.choice(df.index[(lbl == C.BENIGN_LABEL).to_numpy()], n, replace=False)
    for r, a in enumerate(attacks):
        mask = lbl.map(ATTACKS[a]).to_numpy()
        idx = df.index[mask]
        att_idx = rng.choice(idx, min(n, len(idx)), replace=False)
        feats = sep[sep.attack == a].sort_values("KS_D", ascending=False).feature.head(per_attack)
        for c, f in enumerate(feats):
            ax = axes[r, c]
            vb = np.log10(1 + np.clip(df.loc[ben_idx, f].to_numpy(), 0, None))
            va = np.log10(1 + np.clip(df.loc[att_idx, f].to_numpy(), 0, None))
            bins = np.linspace(min(vb.min(), va.min()), max(vb.max(), va.max()) + 1e-9, 50)
            ax.hist(vb, bins=bins, color=BENIGN_COLOR, alpha=0.65, density=True, label="Нормален")
            ax.hist(va, bins=bins, color=ATTACK_COLOR, alpha=0.6, density=True, label=a)
            d = sep[(sep.attack == a) & (sep.feature == f)].KS_D.iloc[0]
            ax.set_title(f"{f}  (D = {d:.2f})", fontsize=10)
            ax.set_xlabel("log10(1 + стойност)", fontsize=8.5)
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            if c == 0:
                ax.set_ylabel(a, fontsize=11, fontweight="bold", color=INK)
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Разпределения на четирите най-разделими признака за всяка атака (плътност)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "13_feature_distributions.png")


# 14. 2D проекции --------------------------------------------------------------------------
def projections(proj: dict):
    attacks = list(proj)
    fig, axes = plt.subplots(len(attacks), 2, figsize=(10, 4.6 * len(attacks)))
    for r, a in enumerate(attacks):
        p = proj[a]
        for c, (k, ttl) in enumerate([("pca", "PCA (линейна)"), ("tsne", "t-SNE (нелинейна)")]):
            ax = axes[r, c]
            for cls, col, nm in [(0, BENIGN_COLOR, "Нормален"), (1, ATTACK_COLOR, a)]:
                q = p[p.y == cls]
                ax.scatter(q[f"{k}1"], q[f"{k}2"], s=4, alpha=0.45, color=col, label=nm, linewidths=0)
            if k == "pca":
                ttl += f": {p.pca_var1.iloc[0]*100:.1f}% + {p.pca_var2.iloc[0]*100:.1f}% дисперсия"
            ax.set_title(f"{a} - {ttl}", fontsize=10.5)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.legend(markerscale=4, loc="best", fontsize=8.5)
    fig.suptitle(f"Двумерни проекции на {_fmt_int(len(proj[attacks[0]]))} записа за всяка задача",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "14_projections_2d.png")


# 15. Чувствителност към хиперпараметрите -------------------------------------------------------
_PNAME = {"n_estimators": "дървета", "max_depth": "дълб.", "min_samples_leaf": "лист", "max_features": "призн.",
          "C": "C", "gamma": "γ", "learning_rate": "η", "min_child_weight": "мин. тегло"}


def _cfg_label(row, cols):
    parts = []
    for c in cols:
        v = row[c]
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ("None", "nan"):
            v = "∞"
        elif isinstance(v, float) and v.is_integer():
            v = int(v)
        parts.append(f"{_PNAME.get(c.replace('param_clf__', ''), c)} {v}")
    return ", ".join(parts)


def hp_sensitivity(cv: dict, top: int = 8):
    attacks = list(dict.fromkeys(a for _, a in cv))
    algos = list(dict.fromkeys(g for g, _ in cv))
    fig, axes = plt.subplots(len(algos), len(attacks), figsize=(5.6 * len(attacks), 3.6 * len(algos)), squeeze=False)
    for r_, algo in enumerate(algos):
        for c, a in enumerate(attacks):
            r = cv[(algo, a)].copy()
            cols = [k for k in r.columns if k.startswith("param_clf__")]
            r["cfg"] = [_cfg_label(row, cols) for _, row in r.iterrows()]
            r = r.sort_values("mean_test_score").tail(top)
            ax = axes[r_, c]
            ax.barh(r.cfg, r.mean_test_score, xerr=r.std_test_score, color=ALGO_COLOR[algo], height=0.65,
                    error_kw={"ecolor": INK2, "lw": 1})
            lo = r.mean_test_score.min() - 3 * r.std_test_score.max()
            ax.set_xlim(lo, min(1.0, r.mean_test_score.max() + 3 * r.std_test_score.max()))
            ax.set_title(f"{_name(algo)} - {a}: CV F1, топ {top} (± ст. откл.)")
            ax.tick_params(axis="y", labelsize=7.5)
            ax.grid(axis="y", visible=False)
    fig.suptitle("Чувствителност към хиперпараметрите (5-кратна кръстосана валидация)", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "15_hyperparameter_sensitivity.png")


# 16. Криви на обучение ------------------------------------------------------------------------
def learning_curves(lc: pd.DataFrame):
    attacks = _present(lc)
    fig, axes = plt.subplots(2, len(attacks), figsize=(5.5 * len(attacks), 7.4), squeeze=False)
    for c, a in enumerate(attacks):
        for algo in ALGORITHMS:
            d = lc[(lc.attack == a) & (lc.algo == algo)]
            axes[0, c].plot(d.n_train, d.F1, "o-", color=ALGO_COLOR[algo], lw=2, ms=5, label=_name(algo))
            axes[1, c].plot(d.n_train, d.fit_time_s, "o-", color=ALGO_COLOR[algo], lw=2, ms=5, label=_name(algo))
        axes[0, c].set_title(f"{a}: F1 спрямо размера на обучението")
        axes[0, c].yaxis.set_major_formatter(PCT)
        axes[1, c].set_title(f"{a}: време за обучение")
        axes[1, c].set_yscale("log")
        axes[1, c].set_ylabel("секунди (лог. скала)")
        for ax in axes[:, c]:
            ax.set_xscale("log")
            ax.set_xlabel("Брой обучаващи записи (лог. скала)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Криви на обучение (тест върху 10 000 невиждани записа)", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "16_learning_curves.png")


# 17. Брой запазени измерения: подбор на признаци срещу PCA ---------------------------------------
def pca_sweep(ps: pd.DataFrame, chosen: dict):
    attacks = list(dict.fromkeys(ps.attack))
    fig, axes = plt.subplots(2, len(attacks), figsize=(5.5 * len(attacks), 7.6), squeeze=False)
    if "method" not in ps.columns:
        ps = ps.assign(method="pca")
    style = {"select": ("o-", "подбор на признаци"), "pca": ("s:", "PCA")}
    for c, a in enumerate(attacks):
        for algo in ALGORITHMS:
            col = ALGO_COLOR[algo]
            d = ps[(ps.attack == a) & (ps.algo == algo)]
            if not len(d):
                continue
            for meth, (ls, _) in style.items():
                red = d[d.reduced & (d.method == meth)]
                axes[0, c].plot(red.n_components, red.F1, ls, color=col, lw=1.8, ms=4.5,
                                label=f"{_name(algo)}, {style[meth][1]}")
                axes[1, c].plot(red.n_components, red.predict_us_per_record, ls, color=col, lw=1.8, ms=4.5)
            full = d[~d.reduced]
            if len(full):
                axes[0, c].axhline(full.F1.iloc[0], color=col, ls="--", lw=1)
        k = chosen.get(a)
        if k:
            for ax in axes[:, c]:
                ax.axvline(k, color=INK2, ls=":", lw=1)
            axes[0, c].text(k + 0.4, axes[0, c].get_ylim()[0], f" избрано: {k}", fontsize=8, color=INK2, va="bottom")
        axes[0, c].set_title(f"{a}: F1 спрямо броя измерения")
        axes[0, c].yaxis.set_major_formatter(PCT)
        axes[1, c].set_title(f"{a}: време за класификация на запис")
        axes[1, c].set_yscale("log")
        axes[1, c].set_ylabel("µs (лог. скала)")
        for ax in axes[:, c]:
            ax.set_xlabel("Брой запазени измерения (признаци или компоненти)")
    axes[0, 0].legend(loc="lower right", title="пунктир = без редукция", title_fontsize=8, fontsize=7.5)
    fig.suptitle("Компромис на редукцията: подбор на признаци и PCA срещу качество и скорост",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "17_pca_components.png")


# 18. Доверителни интервали ------------------------------------------------------------------------
def bootstrap_forest(ci: pd.DataFrame, metrics=("Precision", "Accuracy", "Sensitivity", "F1")):
    attacks = _present(ci)
    fig, axes = plt.subplots(1, len(attacks), figsize=(5.5 * len(attacks), 4.4), squeeze=False)
    axes = axes[0]
    algos = list(ALGORITHMS)
    for ax, a in zip(axes, attacks):
        d = ci[ci.attack == a]
        ys = np.arange(len(metrics))
        for k, algo in enumerate(ALGORITHMS):
            q = d[d.algo == algo].set_index("metric").loc[list(metrics)]
            off = (k - (len(algos) - 1) / 2) * 0.24
            ax.errorbar(q.value, ys + off, xerr=[q.value - q.ci_low, q.ci_high - q.value], fmt="o",
                        color=ALGO_COLOR[algo], ms=6, capsize=4, lw=1.6, label=_name(algo))
        ax.set_yticks(ys, metrics)
        ax.invert_yaxis()
        _sp = float(d.ci_high.max() - d.ci_low.min())
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=3 if _sp < 0.002 else 2 if _sp < 0.02 else 1))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(f"{a}: стойност и 95% доверителен интервал")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle(f"Bootstrap доверителни интервали ({C.BOOTSTRAP_ROUNDS} повторни извадки на тестовия набор)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "18_bootstrap_ci.png")


# 19. Precision-Recall при естествено съотношение ----------------------------------------------------
def pr_curves(curves: dict, ops: pd.DataFrame):
    attacks = _present(ops)
    fig, axes = plt.subplots(1, len(attacks), figsize=(5.5 * len(attacks), 4.6), squeeze=False)
    for ax, a in zip(axes[0], attacks):
        base = None
        for algo in ALGORITHMS:
            cv = curves.get((a, algo))
            if cv is None:
                continue
            base = cv["base_rate"]
            ax.plot(cv["recall"], cv["precision"], lw=2, color=ALGO_COLOR[algo],
                    label=f"{_name(algo)} (AP = {cv['average_precision']:.3f})")
            dp = cv["default_point"]
            ax.plot(dp["recall"], dp["precision"], "D", color=ALGO_COLOR[algo], ms=8, mec="white", mew=1.5)
        if base is not None:
            ax.axhline(base, color=INK2, ls=":", lw=1)
            ax.text(0.01, base + 0.015, f"случаен класификатор: {base*100:.1f}%", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.03)
        ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision")
        ax.set_title(f"{a} при естествено съотношение")
        ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Precision-Recall криви (ромб = праг по подразбиране)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "19_precision_recall_natural.png")


# 20. Подвидове на атаките ---------------------------------------------------------------------------------
def ddos_subtypes(sub: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6.4))
    algos = list(dict.fromkeys(sub.algo))
    labels = sub[sub.algo == algos[0]].sort_values("n").label.tolist()
    y = np.arange(len(labels))
    h = 0.8 / len(algos)
    for k, algo in enumerate(algos):
        d = sub[sub.algo == algo].set_index("label").loc[labels]
        off = (k - (len(algos) - 1) / 2) * h
        ax.barh(y + off, d.missed, h, color=ALGO_COLOR[algo], label=_name(algo))
        for yi, (m, n, r) in enumerate(zip(d.missed, d.n, d.detection_rate)):
            ax.text(m + 0.15, yi + off, f"{m} от {_fmt_int(n)} ({r*100:.2f}%)", va="center",
                    fontsize=7.5, color=INK2)
    ax.set_yticks(y, [l.replace("DDoS-", "") for l in labels])
    ax.set_xlabel("Брой пропуснати записи (FN) в тестовия набор")
    ax.set_xlim(0, max(sub.missed.max() * 1.8, 3))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Пропуснати записи по разновидност на атаката (в скоби: дял на откритите)")
    return _save(fig, "20_ddos_subtypes.png")


# 21. Пермутационна важност ---------------------------------------------------------------------------
def permutation_importance(pi: pd.DataFrame, top: int = 10):
    attacks = _present(pi)
    algos = list(ALGORITHMS)
    fig, axes = plt.subplots(len(attacks), len(algos), figsize=(5 * len(algos), 4.0 * len(attacks)), squeeze=False)
    for r, a in enumerate(attacks):
        for c, algo in enumerate(algos):
            ax = axes[r, c]
            d = pi[(pi.attack == a) & (pi.algo == algo)].sort_values("importance_mean").tail(top)
            ax.barh(d.feature, d.importance_mean, xerr=d.importance_std, color=ALGO_COLOR[algo], height=0.65,
                    error_kw={"ecolor": INK2, "lw": 1})
            ax.set_title(f"{_name(algo)} - {a}")
            ax.set_xlabel("Спад на F1 при разбъркване на признака")
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle(f"Пермутационна важност на входните признаци (топ {top}, {C.PERMUTATION_REPEATS} повторения)",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "21_permutation_importance.png")


# 22. Пренос между атаките -----------------------------------------------------------------------------
def cross_attack(ca: pd.DataFrame):
    piv = ca.assign(model=ca.algo.map(_name) + " / обучен за " + ca.train_attack).pivot_table(
        index="model", columns="test_attack", values="detection_rate")
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(piv) + 1.5))
    sns.heatmap(piv, annot=True, fmt=".2%", cmap="Reds", vmin=0, vmax=1, ax=ax, cbar=False,
                linewidths=2, linecolor="white", annot_kws={"fontsize": 10, "fontweight": "bold"})
    ax.set_xlabel("Атака в тестовите записи")
    ax.set_ylabel("")
    ax.set_title("Пренос между атаките: дял на откритите атакуващи записи")
    ax.grid(False)
    return _save(fig, "22_cross_attack.png")


# 23. Стабилност по фолдове ------------------------------------------------------------------------------
def cv_folds(summary: dict):
    names = list(summary)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(names) + 1.6))
    for i, nm in enumerate(names):
        algo, a = nm.split("_", 1)
        sc = summary[nm]["cv_fold_scores"]
        ax.scatter(sc, [i] * len(sc), color=ALGO_COLOR[algo], s=40, zorder=3, alpha=0.85)
        m = np.mean(sc)
        ax.plot([m, m], [i - 0.25, i + 0.25], color=INK, lw=2)
        ax.text(1.0 + 0.04 * max(1.0 - min(min(s["cv_fold_scores"]) for s in summary.values()), 0.002), i, f"{m*100:.2f}% ± {np.std(sc)*100:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), [f"{_name(n.split('_')[0])} - {n.split('_', 1)[1]}" for n in names])
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("F1 на валидиращия фолд")
    ax.grid(axis="y", visible=False)
    ax.set_title("Стабилност: F1 в 5-те фолда на кръстосаната валидация (черта = средна стойност)")
    lo = min(min(summary[nm]["cv_fold_scores"]) for nm in names)
    span = max(1.0 - lo, 0.002)
    ax.set_xlim(lo - 0.08 * span, 1.0 + 0.45 * span)  # място за надписите вдясно
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=2 if span < 0.02 else 1))
    ax.set_xticks([t for t in ax.get_xticks() if t <= 1.0 + 1e-12])
    ax.set_xlim(lo - 0.08 * span, 1.0 + 0.45 * span)
    return _save(fig, "23_cv_folds.png")
