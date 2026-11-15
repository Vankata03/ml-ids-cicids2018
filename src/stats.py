"""Статистическа оценка на резултатите.

* bootstrap_ci  - 95% доверителни интервали на метриките чрез непараметричен
                  bootstrap върху тестовите предсказания (Efron & Tibshirani, 1993);
* mcnemar       - тест на McNemar за сравнение на два класификатора върху едни
                  и същи тестови записи (Dietterich, 1998);
* paired_bootstrap_diff - доверителен интервал за разликата в метрика между два модела.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st
from sklearn.metrics import roc_auc_score

from . import config as C

_METRICS = ("Precision", "Accuracy", "Sensitivity", "Specificity", "F1")


def _metrics_from_counts(tp, tn, fp, fn) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        sens = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        spec = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = np.where(prec + sens > 0, 2 * prec * sens / (prec + sens), 0.0)
    return {"Precision": prec, "Accuracy": acc, "Sensitivity": sens, "Specificity": spec, "F1": f1}


def bootstrap_ci(y_true, y_pred, scores=None, rounds: int = C.BOOTSTRAP_ROUNDS,
                 alpha: float = 0.05, seed: int = C.SEED) -> dict:
    """Перцентилни (1-alpha) доверителни интервали на метриките.

    Матрицата на объркване на всяка повторна извадка се изчислява векторно
    чрез претеглено броене (bincount), което позволява 1000 повторения за секунди.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    code = 2 * y_true + y_pred            # 0=TN, 1=FP, 2=FN, 3=TP
    counts = np.empty((rounds, 4))
    aucs = []
    for r in range(rounds):
        idx = rng.integers(0, n, n)
        counts[r] = np.bincount(code[idx], minlength=4)
        if scores is not None and r < min(rounds, 200):   # AUC е по-скъп: 200 повторения
            yt = y_true[idx]
            if yt.min() != yt.max():
                aucs.append(roc_auc_score(yt, np.asarray(scores)[idx]))
    tn, fp, fn, tp = counts.T
    boot = _metrics_from_counts(tp, tn, fp, fn)
    point = _metrics_from_counts(*[np.array(v, dtype=float) for v in (
        ((y_true == 1) & (y_pred == 1)).sum(), ((y_true == 0) & (y_pred == 0)).sum(),
        ((y_true == 0) & (y_pred == 1)).sum(), ((y_true == 1) & (y_pred == 0)).sum())])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for m in _METRICS:
        out[m] = {"value": float(point[m]), "ci_low": float(np.percentile(boot[m], lo)),
                  "ci_high": float(np.percentile(boot[m], hi)), "std": float(np.std(boot[m]))}
    if aucs:
        out["ROC_AUC"] = {"value": float(roc_auc_score(y_true, scores)),
                          "ci_low": float(np.percentile(aucs, lo)),
                          "ci_high": float(np.percentile(aucs, hi)), "std": float(np.std(aucs))}
    out["rounds"] = rounds
    return out


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Тест на McNemar: различават ли се значимо грешките на два класификатора.

    b = записи, верни при A и грешни при B; c = обратното. При b + c < 25 се
    използва точният биномен тест, иначе - хи-квадрат с корекция за непрекъснатост.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    both_ok = int(np.sum(ok_a & ok_b))
    both_wrong = int(np.sum(~ok_a & ~ok_b))
    if b + c == 0:
        stat, p, method = 0.0, 1.0, "няма несъгласия"
    elif b + c < 25:
        p = float(st.binomtest(min(b, c), b + c, 0.5).pvalue)
        stat, method = float(min(b, c)), "точен биномен тест"
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(st.chi2.sf(stat, 1))
        method = "хи-квадрат с корекция на Edwards"
    return {"both_correct": both_ok, "only_A_correct": b, "only_B_correct": c,
            "both_wrong": both_wrong, "statistic": float(stat), "p_value": p, "method": method}


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric: str = "F1",
                          rounds: int = C.BOOTSTRAP_ROUNDS, seed: int = C.SEED) -> dict:
    """95% интервал за разликата metric(A) - metric(B) върху едни и същи повторни извадки."""
    y_true, pred_a, pred_b = (np.asarray(v).astype(int) for v in (y_true, pred_a, pred_b))
    n = len(y_true)
    rng = np.random.default_rng(seed)
    ca, cb = 2 * y_true + pred_a, 2 * y_true + pred_b
    diffs = np.empty(rounds)
    for r in range(rounds):
        idx = rng.integers(0, n, n)
        ma = _metrics_from_counts(*_tp_tn_fp_fn(np.bincount(ca[idx], minlength=4)))[metric]
        mb = _metrics_from_counts(*_tp_tn_fp_fn(np.bincount(cb[idx], minlength=4)))[metric]
        diffs[r] = ma - mb
    point = (_metrics_from_counts(*_tp_tn_fp_fn(np.bincount(ca, minlength=4)))[metric]
             - _metrics_from_counts(*_tp_tn_fp_fn(np.bincount(cb, minlength=4)))[metric])
    return {"metric": metric, "diff": float(point), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "share_A_better": float(np.mean(diffs > 0))}


def _tp_tn_fp_fn(cnt):
    tn, fp, fn, tp = (np.array(float(v)) for v in cnt)
    return tp, tn, fp, fn
