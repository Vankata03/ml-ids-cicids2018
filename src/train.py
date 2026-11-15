"""Фаза на машинно обучение: синтез и обучение на моделите.

За всяка атака се обучават два алгоритъма (Random Forest и SVM) -> 4 модела.
Всеки модел е sklearn Pipeline, който включва цялата предварителна обработка:

    FeatureFilter -> ImportanceSelector(20) -> StandardScaler -> класификатор
(или FeatureFilter -> StandardScaler -> PCA(95%) -> класификатор, вж. config.REDUCTION)

Хиперпараметрите се избират с GridSearchCV (5-кратна стратифицирана
кръстосана валидация, критерий F1) само върху обучаващите данни.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .preprocess import FeatureFilter, ImportanceSelector
from .registry import ALGORITHMS

log = logging.getLogger(__name__)


def make_classifier(algo: str):
    return ALGORITHMS[algo].make()


def make_pipeline(algo: str, reduced: bool = True, drop: tuple = (), n_components=None,
                  method: str | None = None) -> Pipeline:
    """reduced=True: конвейер с редукция на размерността (основните модели).
    method="select": FeatureFilter -> ImportanceSelector(k) -> StandardScaler -> класификатор;
    method="pca":    FeatureFilter -> StandardScaler -> PCA -> класификатор.
    reduced=False: само мащабиране (контролен модел за експеримент E1)."""
    method = method or C.REDUCTION
    steps = []
    if reduced and method == "pca":
        steps += [("filter", FeatureFilter(drop=drop)),
                  ("scaler", StandardScaler()),
                  ("pca", PCA(n_components=n_components or C.PCA_VARIANCE, svd_solver="full", random_state=C.SEED))]
    elif reduced:
        steps += [("filter", FeatureFilter(drop=drop, corr_threshold=C.SELECT_CORR)),
                  ("select", ImportanceSelector(k=n_components or C.SELECT_K, method=C.SELECT_METHOD)),
                  ("scaler", StandardScaler())]
    else:
        steps += [("scaler", StandardScaler())]
    steps.append(("clf", make_classifier(algo)))
    return Pipeline(steps)


def n_reduced(pipe: Pipeline) -> int:
    """Размерност на вектора, който постъпва в класификатора."""
    if "pca" in pipe.named_steps:
        return int(pipe.named_steps["pca"].n_components_)
    if "select" in pipe.named_steps:
        return int(len(pipe.named_steps["select"].support_))
    return int(pipe.named_steps["scaler"].n_features_in_)


def _subsample(X, y, n, seed=C.SEED):
    if len(X) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_s, y_s


def tune(algo: str, X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame, float]:
    """GridSearchCV върху (под)извадка от обучаващите данни. Връща най-добрите параметри."""
    spec = ALGORITHMS[algo]
    X_cv, y_cv = _subsample(X, y, spec.cv_max)
    gs = GridSearchCV(
        make_pipeline(algo), spec.param_grid, scoring=C.CV_SCORING,
        cv=StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.SEED),
        n_jobs=C.N_JOBS, refit=False, return_train_score=True,
    )
    t0 = time.perf_counter()
    gs.fit(X_cv, y_cv)
    t_cv = time.perf_counter() - t0
    res = pd.DataFrame(gs.cv_results_)
    top = res.loc[res["rank_test_score"].idxmin()]
    # Правило „една стандартна грешка“ (Breiman et al., 1984; Hastie et al., 2009):
    # от всички конфигурации, чиято CV оценка е в рамките на една стандартна грешка
    # от най-добрата, се избира най-простата (най-силно регуляризираната). Така се
    # намалява преобучението без статистически значима загуба на точност.
    se = top["std_test_score"] / np.sqrt(C.CV_FOLDS)
    ok = res[res["mean_test_score"] >= top["mean_test_score"] - se]
    best = min(ok.itertuples(), key=lambda r: spec.complexity(r.params))
    best = res.loc[best.Index]
    res["selected"] = res.index == best.name
    res["one_se_threshold"] = top["mean_test_score"] - se
    best_params = best["params"]
    log.info("%s: избрани параметри %s (CV F1 = %.5f ± %.5f; най-висока %.5f; праг 1-SE %.5f; "
             "%d записа, %.1f s)", algo, best_params, best["mean_test_score"], best["std_test_score"],
             top["mean_test_score"], top["mean_test_score"] - se, len(X_cv), t_cv)
    return best_params, res, t_cv


def fit_final(algo: str, params: dict, X, y, reduced: bool = True, drop: tuple = (),
              n_components=None, train_max: int | None = None, method: str | None = None):
    """Обучава финалния модел с избраните параметри и измерва времето за обучение.

    train_max ограничава обучаващата извадка (по подразбиране - лимитът на алгоритъма)."""
    spec = ALGORITHMS[algo]
    limit = train_max if train_max is not None else spec.train_max
    if limit is not None:
        X, y = _subsample(X, y, limit)
    pipe = make_pipeline(algo, reduced=reduced, drop=drop, n_components=n_components, method=method)
    pipe.set_params(**params)
    if spec.parallel_param:
        pipe.set_params(**{spec.parallel_param: C.N_JOBS})
    t0 = time.perf_counter()
    pipe.fit(X, y)
    t_fit = time.perf_counter() - t0
    return pipe, t_fit, len(X)


def reduction_info(pipe: Pipeline) -> dict:
    """Информация за редукцията на размерността в обучения конвейер."""
    if "filter" not in pipe.named_steps:
        return {}
    flt = pipe.named_steps["filter"]
    info = {
        "method": "pca" if "pca" in pipe.named_steps else "select",
        "n_features_in": int(len(flt.feature_names_in_)),
        "dropped_explicit": list(flt.drop),
        "dropped_low_variance": flt.low_variance_,
        "dropped_correlated": flt.correlated_,
        "n_features_after_filter": int(len(flt.selected_)),
        "n_reduced": n_reduced(pipe),
    }
    if "pca" in pipe.named_steps:
        pca = pipe.named_steps["pca"]
        info.update({"selected_features": flt.selected_, "n_pca_components": int(pca.n_components_),
                     "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                     "pca_total_explained": float(np.sum(pca.explained_variance_ratio_))})
    else:
        sel = pipe.named_steps["select"]
        names = np.array(flt.selected_)
        order = np.argsort(-sel.importances_)
        info.update({"selected_features": names[sel.support_].tolist(),
                     "importance_ranking": [{"feature": str(names[i]), "importance": float(sel.importances_[i])}
                                            for i in order],
                     "selected_importance_share": float(sel.importances_[sel.support_].sum())})
    return info
