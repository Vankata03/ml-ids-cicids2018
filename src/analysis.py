"""Разширени анализи (A1-A9) върху вече обучените модели и реалните данни.

A1 профил на данните и едномерна разделимост на признаците (тест на Колмогоров-Смирнов);
A2 2D проекции на пространството на признаците (PCA и t-SNE);
A3 криви на обучение (качество и време спрямо размера на обучаващата извадка);
A4 анализ на броя PCA компоненти (компромис качество/скорост на редукцията);
A5 избор на праг на решение при естествено съотношение на класовете (Precision-Recall);
A6 откриване по подвидове на атаките;
A7 пренос между атаките (модел за една атака върху другата);
A8 пермутационна важност на входните признаци за двата алгоритъма;
A9 статистическо сравнение на алгоритмите (bootstrap интервали, McNemar) - в src/stats.py.

Всички функции работят само с оригиналните записи на CSE-CIC-IDS2018.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from . import config as C
from .evaluate import compute_metrics, scores_of
from .registry import ALGORITHMS, ATTACKS
from .train import _subsample, fit_final, n_reduced

log = logging.getLogger(__name__)


def _stratified(X, y, n, seed=C.SEED):
    return _subsample(X, y, n, seed)


# ---------------------------------------------------------------------------
# A1. Профил на данните
# ---------------------------------------------------------------------------
def data_profile(df: pd.DataFrame, sample: int = 50_000, seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Описателна статистика по групи (нормален трафик и всяка атака) и KS разделимост.

    Връща (profile, separability):
      profile       - медиана, средна, стандартно отклонение и квартили на всеки признак по група;
      separability  - KS статистика D между атаката и нормалния трафик за всеки признак
                      (0 = еднакви разпределения, 1 = напълно разделими).
    """
    rng = np.random.default_rng(seed)
    lbl = df[C.LABEL_COLUMN]
    groups = {"Нормален": lbl == C.BENIGN_LABEL}
    for key, spec in ATTACKS.items():
        m = lbl.map(spec)
        if m.sum() > 0:  # атака без налични записи се пропуска
            groups[key] = m
    feats = [c for c in df.columns if c != C.LABEL_COLUMN]
    samples = {}
    rows = []
    for g, mask in groups.items():
        idx = df.index[mask.to_numpy()]
        idx = rng.choice(idx, min(sample, len(idx)), replace=False)
        samples[g] = df.loc[idx, feats]
        d = samples[g].describe(percentiles=[0.25, 0.5, 0.75]).T
        d.insert(0, "group", g)
        d["n_total"] = int(mask.sum())
        rows.append(d)
    profile = pd.concat(rows).rename_axis("feature").reset_index()
    sep = []
    for key in [k for k in ATTACKS if k in samples]:
        for f in feats:
            a, b = samples[key][f].to_numpy(), samples["Нормален"][f].to_numpy()
            ks = ks_2samp(a, b)
            sep.append({"attack": key, "feature": f, "KS_D": float(ks.statistic),
                        "median_attack": float(np.median(a)), "median_benign": float(np.median(b))})
    return profile, pd.DataFrame(sep)


# ---------------------------------------------------------------------------
# A2. 2D проекции
# ---------------------------------------------------------------------------
def projection_2d(pipe, X, y, n: int = C.PROJECTION_SAMPLES, seed: int = C.SEED) -> pd.DataFrame:
    """Проектира стратифицирана извадка в 2D след филтъра и мащабирането на модела."""
    Xs, ys = _stratified(X, y, n, seed)
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(pipe.named_steps["filter"].transform(Xs))  # филтър + мащабиране
    p = PCA(n_components=2, random_state=seed).fit(Z)
    P = p.transform(Z)
    T = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
             random_state=seed).fit_transform(Z)
    return pd.DataFrame({"y": np.asarray(ys), "pca1": P[:, 0], "pca2": P[:, 1],
                         "tsne1": T[:, 0], "tsne2": T[:, 1],
                         "pca_var1": p.explained_variance_ratio_[0],
                         "pca_var2": p.explained_variance_ratio_[1]})


# ---------------------------------------------------------------------------
# A3. Криви на обучение
# ---------------------------------------------------------------------------
def learning_curve(algo: str, params: dict, task: dict, sizes=C.LEARNING_CURVE_SIZES) -> pd.DataFrame:
    X_tr, y_tr = task["X_train"], task["y_train"]
    X_te, y_te = _stratified(task["X_test"], task["y_test"], C.SWEEP_TEST_MAX, C.SEED + 7)
    limit = ALGORITHMS[algo].train_max or len(X_tr)
    out = []
    for n in sizes:
        if n > limit:
            continue
        pipe, t_fit, n_fit = fit_final(algo, params, X_tr, y_tr, train_max=n)
        t0 = time.perf_counter()
        y_pred = pipe.predict(X_te)
        t_pred = time.perf_counter() - t0
        m = compute_metrics(np.asarray(y_te), y_pred)
        out.append({"algo": algo, "n_train": n_fit, "F1": m["F1"], "Accuracy": m["Accuracy"],
                    "Precision": m["Precision"], "Sensitivity": m["Sensitivity"],
                    "fit_time_s": t_fit, "predict_us_per_record": t_pred / len(X_te) * 1e6})
        log.info("  крива на обучение %s n=%d: F1=%.4f, %.1f s", algo, n_fit, m["F1"], t_fit)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# A4. Брой PCA компоненти
# ---------------------------------------------------------------------------
def pca_sweep(algo: str, params: dict, task: dict, ks=C.PCA_SWEEP, method: str = "pca") -> pd.DataFrame:
    X_tr, y_tr = task["X_train"], task["y_train"]
    X_te, y_te = _stratified(task["X_test"], task["y_test"], C.SWEEP_TEST_MAX, C.SEED + 7)
    out = []
    configs = [(k, True) for k in ks] + [(None, False)]
    for k, reduced in configs:
        pipe, t_fit, n_fit = fit_final(algo, params, X_tr, y_tr, reduced=reduced,
                                       n_components=k, train_max=C.SWEEP_TRAIN_MAX, method=method)
        n_feat = n_reduced(pipe) if reduced else X_tr.shape[1]
        t0 = time.perf_counter()
        y_pred = pipe.predict(X_te)
        t_pred = time.perf_counter() - t0
        out.append({"algo": algo, "n_components": int(n_feat), "reduced": reduced,
                    "explained": (float(np.sum(pipe.named_steps["pca"].explained_variance_ratio_))
                                  if reduced and "pca" in pipe.named_steps else
                                  float(pipe.named_steps["select"].importances_[pipe.named_steps["select"].support_].sum())
                                  if reduced else 1.0),
                    "F1": f1_score(y_te, y_pred), "fit_time_s": t_fit,
                    "predict_us_per_record": t_pred / len(X_te) * 1e6, "n_train": n_fit})
        log.info("  %s %s k=%s: F1=%.4f", method, algo, n_feat, out[-1]["F1"])
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# A5. Праг на решение при естествено съотношение
# ---------------------------------------------------------------------------
def threshold_analysis(y_true, scores, default_threshold: float, algo: str,
                       targets=C.TARGET_PRECISIONS) -> tuple[pd.DataFrame, dict]:
    """Precision-Recall крива и работни точки за целеви нива на Precision.

    default_threshold е прагът, който моделът използва по подразбиране
    (0.5 за вероятност при RF, 0 за разстояние до хиперравнината при SVM).
    """
    y_true = np.asarray(y_true)
    prec, rec, thr = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    rows = []
    for t in targets:
        ok = np.where(prec[:-1] >= t)[0]
        if len(ok):
            i = ok[np.argmax(rec[:-1][ok])]
            rows.append({"algo": algo, "target_precision": t, "precision": float(prec[i]),
                         "recall": float(rec[i]), "threshold": float(thr[i]),
                         "alarms_per_10k": float(np.mean(scores >= thr[i]) * 1e4)})
        else:
            rows.append({"algo": algo, "target_precision": t, "precision": np.nan,
                         "recall": 0.0, "threshold": np.nan, "alarms_per_10k": 0.0})
    pred = (np.asarray(scores) >= default_threshold).astype(int)
    m = compute_metrics(y_true, pred)
    # крива, разредена до ~400 точки за графиката
    step = max(1, len(prec) // 400)
    curve = {"precision": prec[::step].tolist(), "recall": rec[::step].tolist(),
             "average_precision": float(ap),
             "default_point": {"precision": m["Precision"], "recall": m["Sensitivity"]},
             "base_rate": float(y_true.mean())}
    return pd.DataFrame(rows), curve


def recall_points(y_true, scores, algo: str, targets=C.TARGET_RECALLS) -> pd.DataFrame:
    """Работни точки, които гарантират зададена Sensitivity (Recall).

    Пропуснатата атака (FN) е най-нежеланата грешка за IDS. За всяко целево
    ниво на Recall се избира най-високият праг, при който то е постигнато,
    и се отчита цената: Precision, Specificity и брой аларми на 10 000 записа.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.sort(np.unique(s))[::-1]
    pos, neg = y.sum(), (1 - y).sum()
    rows = []
    for t in targets:
        # най-високият праг θ с Recall(θ) >= t
        thr = np.quantile(s[y == 1], 1 - t, method="lower")
        pred = s >= thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        rows.append({"algo": algo, "target_recall": t, "threshold": float(thr),
                     "recall": tp / pos, "precision": tp / max(tp + fp, 1),
                     "specificity": 1 - fp / neg, "TP": tp, "FP": fp,
                     "alarms_per_10k": float(pred.mean() * 1e4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A6. Подвидове на атаките
# ---------------------------------------------------------------------------
def subtype_detection(labels, y_true, y_pred, algo: str) -> pd.DataFrame:
    df = pd.DataFrame({"label": np.asarray(labels), "y": np.asarray(y_true), "pred": np.asarray(y_pred)})
    att = df[df.y == 1]
    g = att.groupby("label")["pred"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "detected"})
    g["missed"] = g["n"] - g["detected"]
    g["detection_rate"] = g["detected"] / g["n"]
    g.insert(0, "algo", algo)
    return g.reset_index().sort_values("detection_rate")


# ---------------------------------------------------------------------------
# A7. Пренос между атаките
# ---------------------------------------------------------------------------
def cross_attack(pipes: dict, tasks: dict) -> pd.DataFrame:
    """Всеки модел се прилага върху тестовите записи на всяка атака.

    Диагоналът (обучение = тест) повтаря основните резултати; извъндиагоналните
    клетки показват дали модел, обучен за една атака, разпознава другата."""
    rows = []
    for (train_attack, algo), pipe in pipes.items():
        for test_attack, task in tasks.items():
            X, y = task["X_test"], np.asarray(task["y_test"])
            Xa = X[y == 1]
            Xb = X[y == 0]
            Xa = Xa.sample(min(len(Xa), C.SWEEP_TEST_MAX), random_state=C.SEED)
            Xb = Xb.sample(min(len(Xb), C.SWEEP_TEST_MAX), random_state=C.SEED)
            det = float(np.mean(pipe.predict(Xa)))
            fpr = float(np.mean(pipe.predict(Xb)))
            rows.append({"train_attack": train_attack, "algo": algo, "test_attack": test_attack,
                         "detection_rate": det, "false_alarm_rate": fpr,
                         "n_attack": len(Xa), "n_benign": len(Xb)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A8. Пермутационна важност
# ---------------------------------------------------------------------------
def permutation_importances(pipe, X, y, algo: str, attack: str,
                            n: int = C.PERMUTATION_SAMPLES, repeats: int = C.PERMUTATION_REPEATS) -> pd.DataFrame:
    """Спад на F1 при разбъркване на всеки входен признак (Breiman, 2001; Fisher et al., 2019).

    Изчислява се върху целия конвейер, т.е. спрямо оригиналните 39 признака,
    което прави RF и SVM пряко сравними въпреки PCA."""
    Xs, ys = _stratified(X, y, n, C.SEED + 11)
    t0 = time.perf_counter()
    r = permutation_importance(pipe, Xs, ys, scoring="f1", n_repeats=repeats,
                               random_state=C.SEED, n_jobs=1)
    log.info("  пермутационна важност %s-%s: %.1f s", algo, attack, time.perf_counter() - t0)
    return pd.DataFrame({"attack": attack, "algo": algo, "feature": Xs.columns,
                         "importance_mean": r.importances_mean, "importance_std": r.importances_std}
                        ).sort_values("importance_mean", ascending=False)
