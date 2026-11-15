"""Оперативна фаза (тестване/експлоатация) и изчисляване на метриките.

Записаният конвейер (обработка + модел) се зарежда и класифицира тестовите
записи, които не са участвали в обучението, на последователни пакети от
OPERATIONAL_BATCH записа - така, както би работила IDS система върху поток
от мрежови записи.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from . import config as C


def scores_of(pipe, X):
    """Непрекъсната оценка за ROC: вероятност (RF) или разстояние до хиперравнината (SVM)."""
    if hasattr(pipe, "predict_proba") and hasattr(pipe.named_steps["clf"], "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipe.decision_function(X)


def operational_phase(pipe, X, y) -> tuple[dict, np.ndarray, np.ndarray]:
    """Класифицира X на пакети и връща метриките, предсказанията и оценките за ROC."""
    preds = []
    t0 = time.perf_counter()
    for start in range(0, len(X), C.OPERATIONAL_BATCH):
        preds.append(pipe.predict(X.iloc[start:start + C.OPERATIONAL_BATCH]))
    t_pred = time.perf_counter() - t0
    y_pred = np.concatenate(preds)
    scores = scores_of(pipe, X)
    m = compute_metrics(np.asarray(y), y_pred, scores)
    m["predict_time_s"] = t_pred
    m["predict_time_per_record_us"] = t_pred / len(X) * 1e6
    m["n_test"] = int(len(X))
    return m, y_pred, scores


def compute_metrics(y_true, y_pred, scores=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) else 0.0,
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        m["ROC_AUC"] = roc_auc_score(y_true, scores)
    return m


def roc_points(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr
