"""Предварителна обработка на CSE-CIC-IDS2018.

Стъпки (виж плана, раздел 4):
  1. зареждане само на нужните класове (Benign, DDOS attack-*, FTP/SSH-BruteForce, Bot);
  2. почистване: NaN/Inf, дубликати;
  3. формиране на бинарна задача за всяка атака;
  4. балансирана стратифицирана извадка;
  5. стратифицирано разделяне 70/30;
  6-8. филтриране на признаци, мащабиране и PCA се извършват ВЪТРЕ в sklearn
       Pipeline (FeatureFilter -> StandardScaler -> PCA), за да се обучават
       само върху обучаващите данни (без изтичане на информация).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from . import config as C
from .registry import ATTACKS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Зареждане
# ---------------------------------------------------------------------------
def _wanted(label: str) -> bool:
    return label == C.BENIGN_LABEL or any(spec(label) for spec in ATTACKS.values())


def _load_fraction(label: str) -> float:
    for prefix, frac in C.LOAD_FRACTION.items():
        if label.startswith(prefix):
            return frac
    return 1.0


def load_raw(raw_dir: Path = C.RAW_DIR, chunksize: int = 500_000) -> tuple[pd.DataFrame, dict]:
    """Чете всички CSV файлове от raw_dir на части и запазва само нужните класове.

    Връща DataFrame и статистика за оригиналния брой записи по клас
    (преди филтриране), която се използва в доклада.
    """
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"Няма CSV файлове в {raw_dir}. Изтеглете CSE-CIC-IDS2018 (виж data/README.md)."
        )
    parts, original_counts = [], {}
    rng = np.random.default_rng(C.SEED)
    for i, f in enumerate(files, 1):
        for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            chunk = chunk.drop(columns=[c for c in C.DROP_COLUMNS if c in chunk.columns])
            label_col = next((c for c in chunk.columns if c.lower() == C.LABEL_COLUMN), None)
            if label_col is None:
                # Файловете по класове (CSV/CSV/<клас>/*.csv) нямат колона с етикет:
                # етикетът е името на папката, в която се намира файлът.
                chunk[C.LABEL_COLUMN] = C.FOLDER_LABELS.get(f.parent.name, f.parent.name)
            elif label_col != C.LABEL_COLUMN:
                chunk = chunk.rename(columns={label_col: C.LABEL_COLUMN})
            lbl = chunk[C.LABEL_COLUMN].astype(str).str.strip()
            for k, v in lbl.value_counts().items():
                original_counts[k] = original_counts.get(k, 0) + int(v)
            keep = lbl.map(_wanted)
            # DDoS записите са десетки милиони: зареждаме случайна част от тях,
            # като оригиналният им брой се пази в статистиката.
            frac = lbl.map(_load_fraction)
            keep &= rng.random(len(lbl)) < frac.to_numpy()
            sub = chunk.loc[keep].copy()
            sub[C.LABEL_COLUMN] = lbl[keep]
            num_cols = sub.columns.drop(C.LABEL_COLUMN)
            sub[num_cols] = sub[num_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
            parts.append(sub)
        log.info("Прочетен файл %d/%d: %s", i, len(files), f.name)
    df = pd.concat(parts, ignore_index=True)
    stats = {
        "files": len(files),
        "file_names": [f"{f.parent.name}/{f.name}" for f in files],
        "rows_read_total": int(sum(original_counts.values())),
        "original_class_counts": dict(sorted(original_counts.items(), key=lambda x: -x[1])),
        "rows_kept": int(len(df)),
        "n_features_raw": int(df.shape[1] - 1),
        "feature_names_raw": [c for c in df.columns if c != C.LABEL_COLUMN],
    }
    return df, stats


# ---------------------------------------------------------------------------
# 2. Почистване
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    num_cols = df.columns.drop(C.LABEL_COLUMN)
    df = df.replace([np.inf, -np.inf], np.nan)
    n_nan_rows = int(df[num_cols].isna().any(axis=1).sum())
    df = df.dropna()
    n1 = len(df)
    df = df.drop_duplicates()
    n2 = len(df)
    stats = {
        "rows_before": n0,
        "rows_with_nan_or_inf": n_nan_rows,
        "duplicates_removed": n1 - n2,
        "rows_after": n2,
        "class_counts_after_cleaning": df[C.LABEL_COLUMN].value_counts().to_dict(),
    }
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# 3-5. Бинарна задача, балансиране, разделяне
# ---------------------------------------------------------------------------
def build_task(df: pd.DataFrame, attack: str, original_counts: dict, seed: int = C.SEED) -> dict:
    """Формира балансирана бинарна задача (атака=1, нормален=0) и я разделя 70/30.

    Записите, които НЕ влизат в балансираната извадка, остават като „резерв“
    за експеримент E2 (тест при естествено съотношение на класовете).
    """
    rule = ATTACKS[attack]
    lbl = df[C.LABEL_COLUMN]
    is_attack = lbl.map(rule)
    is_benign = lbl == C.BENIGN_LABEL

    # Дубликати на ниво задача: различни оригинални етикети могат да имат еднакъв
    # вектор на признаците (напр. DDoS-SYN_Flood и DDoS-SynonymousIP_Flood).
    # След обединяването в бинарен етикет те стават пълни дубликати и биха могли
    # да попаднат и в обучаващите, и в тестовите данни. Затова се пази по едно копие.
    # Вектор, срещан и като атака, и като нормален трафик, е неразрешимо противоречие
    # и всичките му копия се премахват.
    in_task = (is_attack | is_benign).to_numpy()
    sub = df.loc[in_task]
    key = pd.util.hash_pandas_object(sub.drop(columns=[C.LABEL_COLUMN]), index=False)
    y_task = is_attack.loc[sub.index].astype(int)
    n_labels = y_task.groupby(key.values).transform("nunique")
    conflict = (n_labels > 1).to_numpy()
    dup = key.duplicated().to_numpy() & ~conflict
    keep_idx = sub.index[~conflict & ~dup]
    task_cleaning = {"task_duplicates_removed": int(dup.sum()),
                     "task_conflicting_rows_removed": int(conflict.sum()),
                     "task_conflicting_vectors": int(key[conflict].nunique())}
    keep = pd.Series(False, index=df.index)
    keep.loc[keep_idx] = True
    pos_idx = df.index[is_attack & keep].to_numpy()
    neg_idx = df.index[is_benign & keep].to_numpy()

    rng = np.random.default_rng(seed)
    n = int(min(len(pos_idx), len(neg_idx), C.MAX_PER_CLASS))
    pos_s = rng.choice(pos_idx, n, replace=False)
    neg_s = rng.choice(neg_idx, n, replace=False)
    idx = np.concatenate([pos_s, neg_s])

    X = df.loc[idx].drop(columns=[C.LABEL_COLUMN])
    y = is_attack.loc[idx].astype(int)
    sub_labels = lbl.loc[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=seed
    )

    # Резерв за E2: всички записи, които НЕ са използвани за обучение
    # (включително тестовата част на балансираната извадка).
    train_idx = X_tr.index.to_numpy()
    pos_rest = np.setdiff1d(pos_idx, train_idx)
    neg_rest = np.setdiff1d(neg_idx, train_idx)
    # естествено съотношение по ОРИГИНАЛНИЯ брой записи в набора (преди извадката)
    n_att = sum(v for k, v in original_counts.items() if rule(k))
    n_ben = original_counts.get(C.BENIGN_LABEL, 0)
    natural_ratio = n_att / (n_att + n_ben)

    stats = {
        "attack": attack,
        "attack_labels_included": sorted(lbl[is_attack].unique().tolist()),
        "attack_rows_original": int(n_att),
        "benign_rows_original": int(n_ben),
        "attack_rows_available": int(len(pos_idx)),
        "benign_rows_available": int(len(neg_idx)),
        "natural_attack_share": float(natural_ratio),
        "per_class_sample": n,
        **task_cleaning,
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "attack_subtype_counts_in_sample": sub_labels[y == 1].value_counts().to_dict(),
    }
    return {
        "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te,
        "labels_test": lbl.loc[X_te.index], "labels_train": lbl.loc[X_tr.index],
        "pos_rest": pos_rest, "neg_rest": neg_rest,
        "natural_ratio": natural_ratio, "stats": stats,
    }


def natural_test_set(df: pd.DataFrame, task: dict, attack: str, seed: int = C.SEED):
    """E2: тестов набор от НЕИЗПОЛЗВАНИ записи в естественото съотношение атака/нормален.

    Ако резервът на единия клас не стига, размерът се намалява пропорционално,
    така че съотношението да се запази.
    """
    rng = np.random.default_rng(seed + 1)
    p = task["natural_ratio"]
    pos_rest, neg_rest = task["pos_rest"], task["neg_rest"]
    total = C.NATURAL_TEST_SIZE
    n_pos, n_neg = int(round(total * p)), int(round(total * (1 - p)))
    scale = min(1.0,
                len(pos_rest) / n_pos if n_pos else 1.0,
                len(neg_rest) / n_neg if n_neg else 1.0)
    n_pos, n_neg = int(n_pos * scale), int(n_neg * scale)
    if n_pos == 0 or n_neg == 0:
        return None, None
    idx = np.concatenate([rng.choice(pos_rest, n_pos, replace=False),
                          rng.choice(neg_rest, n_neg, replace=False)])
    X = df.loc[idx].drop(columns=[C.LABEL_COLUMN])
    lab = df.loc[idx, C.LABEL_COLUMN]
    y = lab.map(ATTACKS[attack]).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 6. Редукция, етап 1: филтриране на признаци (част от Pipeline)
# ---------------------------------------------------------------------------
class FeatureFilter(BaseEstimator, TransformerMixin):
    """Премахва признаци с нулева дисперсия и силно корелирани признаци.

    Всичко се изчислява в fit() само върху обучаващите данни. От всяка двойка
    с |r| > corr_threshold се премахва признакът, който е по-силно корелиран
    средно с останалите (т.е. носи по-малко уникална информация).
    """

    def __init__(self, var_threshold: float = C.VARIANCE_THRESHOLD,
                 corr_threshold: float = C.CORR_THRESHOLD, drop: tuple = ()):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.drop = drop

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.array(X.columns)
        X = X.drop(columns=[c for c in self.drop if c in X.columns])
        var = X.var()
        self.low_variance_ = var.index[var <= self.var_threshold].tolist()
        X = X.drop(columns=self.low_variance_)
        corr = X.corr().abs().fillna(0.0)
        mean_corr = corr.mean()
        to_drop = set()
        cols = corr.columns.tolist()
        for i, a in enumerate(cols):
            if a in to_drop:
                continue
            for b in cols[i + 1:]:
                if b in to_drop:
                    continue
                if corr.loc[a, b] > self.corr_threshold:
                    to_drop.add(a if mean_corr[a] > mean_corr[b] else b)
                    if a in to_drop:
                        break
        self.correlated_ = sorted(to_drop)
        self.selected_ = [c for c in cols if c not in to_drop]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.selected_].to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Подбор на k-те най-информативни признака (обвиващ/вграден метод за редукция).

    Важността се изчислява в fit() само върху обучаващите данни с помощен ансамбъл
    от решаващи дървета (по подразбиране XGBoost, критерий „gain“ - средното подобрение
    на загубата при разклонение по признака). Така редукцията запазва оригиналните,
    интерпретируеми признаци, вместо да ги смесва линейно като PCA.
    """

    def __init__(self, k: int = 20, method: str = "xgb_gain", seed: int = C.SEED):
        self.k = k
        self.method = method
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[1]
        if self.method == "xgb_gain":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist",
                              random_state=self.seed, n_jobs=4).fit(X, y)
            g = m.get_booster().get_score(importance_type="gain")
            imp = np.array([g.get(f"f{i}", 0.0) for i in range(n)])
        elif self.method == "rf_mdi":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                       random_state=self.seed).fit(X, y)
            imp = m.feature_importances_
        else:
            from sklearn.feature_selection import mutual_info_classif
            imp = mutual_info_classif(X, y, random_state=self.seed)
        self.importances_ = imp / imp.sum() if imp.sum() > 0 else imp
        self.support_ = np.sort(np.argsort(-self.importances_, kind="stable")[: min(self.k, n)])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.support_]
