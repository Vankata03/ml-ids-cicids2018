"""Оркестрация на експеримента: Experiment обхожда регистрите от атаки и алгоритми.

Етапи:
  1. prepare()   - зареждане и почистване на реалните данни, профил на данните (A1);
  2. run_main()  - за всяка двойка (атака, алгоритъм): фаза на машинно обучение
                   (GridSearchCV + финално обучение), запис на модела, оперативна фаза
                   и контролните експерименти E1 (без редукция), E2 (естествено
                   съотношение), E4 (без времевите признаци);
  3. run_extended() - разширените анализи A2-A9 (src/analysis.py, src/stats.py);
  4. save() / figures() - CSV/JSON резултати и всички графики.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import analysis as A
from . import config as C
from . import plots, plots_ext
from . import stats as S
from .diagram import pipeline_diagram
from .evaluate import compute_metrics, operational_phase, roc_points
from .preprocess import FeatureFilter, build_task, clean, load_raw, natural_test_set, save_json
from sklearn.preprocessing import StandardScaler
from .registry import ALGORITHMS, ATTACKS
from .train import fit_final, reduction_info, tune

log = logging.getLogger("experiment")


@dataclass
class ModelRun:
    """Всичко, което е известно за един обучен модел (атака x алгоритъм)."""
    attack: str
    algo: str
    params: dict
    pipe: object
    metrics: dict
    y_pred: np.ndarray
    scores: np.ndarray
    nat_metrics: dict | None = None
    nat_scores: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.algo}_{self.attack}"


class Experiment:
    def __init__(self, attacks=None, algos=None, extended: bool = True):
        self.attacks = list(attacks or ATTACKS)
        self.algos = list(algos or ALGORITHMS)
        self.extended = extended
        self.out = C.RESULTS_DIR
        self.runs: dict[tuple[str, str], ModelRun] = {}
        self.tasks: dict[str, dict] = {}
        self.nat: dict[str, tuple] = {}
        self.rows = {"main": [], "full": [], "natural": [], "no_iat": [], "other_reduction": [], "overfit": []}
        self.red_infos, self.cv_summary, self.cv_tables = {}, {}, {}
        self.rocs, self.aucs, self.pca_curves, self.importances = {}, {}, {}, {}
        self.ext: dict[str, object] = {}

    # ------------------------------------------------------------------ helpers
    def _row(self, attack, algo, m, extra=None):
        r = {"model": f"{ALGORITHMS[algo].name_bg} - {attack}", "attack": attack, "algo": algo}
        r.update(m)
        r.update(extra or {})
        return r

    def _csv(self, df: pd.DataFrame, name: str):
        df.to_csv(self.out / name, index=False)

    # ------------------------------------------------------------------ 1. данни
    def prepare(self):
        log.info("Зареждане на %s от %s", C.DATASET_NAME, C.RAW_DIR)
        df, self.load_stats = load_raw()
        self.df, self.clean_stats = clean(df)
        save_json({"load": self.load_stats, "clean": self.clean_stats}, self.out / "dataset_stats.json")
        log.info("След почистване: %d записа, %d признака", len(self.df), self.df.shape[1] - 1)
        if self.extended:
            log.info("A1: профил на данните и KS разделимост")
            self.profile, self.separability = A.data_profile(self.df)
            self._csv(self.profile, "data_profile.csv")
            self._csv(self.separability, "ks_separability.csv")
        for attack in self.attacks:
            task = build_task(self.df, attack, self.load_stats["original_class_counts"])
            self.tasks[attack] = task
            self.nat[attack] = natural_test_set(self.df, task, attack)
            log.info("Задача %s: %s", attack, {k: v for k, v in task["stats"].items()
                                              if k not in ("attack_labels_included", "attack_subtype_counts_in_sample")})

    # ------------------------------------------------------------------ 2. основни модели
    def run_main(self):
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for idx, attack in enumerate(self.attacks):
            task = self.tasks[attack]
            X_tr, X_te, y_tr, y_te = task["X_train"], task["X_test"], task["y_train"], task["y_test"]
            X_nat, y_nat = self.nat[attack]
            log.info("=" * 70)
            log.info("ЗАДАЧА: %s срещу нормален трафик", attack)
            for algo in self.algos:
                name = f"{algo}_{attack}"
                log.info("-" * 70)
                log.info("МОДЕЛ %s: фаза на машинно обучение", name)
                params, cv_res, t_cv = tune(algo, X_tr, y_tr)
                cv_res = cv_res.drop(columns=["params"])
                cv_res.to_csv(self.out / f"cv_{name}.csv", index=False)
                self.cv_tables[(algo, attack)] = cv_res
                best = cv_res.loc[cv_res["selected"]].iloc[0]
                top = cv_res.loc[cv_res["rank_test_score"].idxmin()]
                self.cv_summary[name] = {
                    "best_params": params, "cv_f1_mean": float(best["mean_test_score"]),
                    "cv_f1_std": float(best["std_test_score"]), "cv_time_s": t_cv,
                    "cv_f1_train_mean": float(best["mean_train_score"]),
                    "cv_f1_top_mean": float(top["mean_test_score"]),
                    "one_se_threshold": float(best["one_se_threshold"]),
                    "selection_rule": "one_standard_error",
                    "cv_fold_scores": [float(best[f"split{k}_test_score"]) for k in range(C.CV_FOLDS)]}

                pipe, t_fit, n_fit = fit_final(algo, params, X_tr, y_tr)
                joblib.dump(pipe, C.MODELS_DIR / f"{name}.joblib", compress=3)
                self.red_infos[name] = reduction_info(pipe)
                ri = self.red_infos[name]
                log.info("Редукция (%s): %d -> %d признака -> %d", ri["method"],
                         ri["n_features_in"], ri["n_features_after_filter"], ri["n_reduced"])

                log.info("МОДЕЛ %s: оперативна фаза", name)
                pipe = joblib.load(C.MODELS_DIR / f"{name}.joblib")  # записаният модел
                m, y_pred, sc = operational_phase(pipe, X_te, y_te)
                run = ModelRun(attack, algo, params, pipe, m, y_pred, sc)
                self.runs[(attack, algo)] = run
                self.rows["main"].append(self._row(attack, algo, m, {
                    "fit_time_s": t_fit, "n_train": n_fit, "n_reduced": ri["n_reduced"],
                    **{f"param_{k.replace('clf__', '')}": v for k, v in params.items()}}))
                self.rocs[(attack, algo)] = roc_points(y_te, sc)
                self.aucs[(attack, algo)] = m["ROC_AUC"]
                log.info("Резултат: %s", {k: m[k] for k in ["TP", "TN", "FP", "FN", "Precision", "Accuracy", "Sensitivity", "F1"]})

                if X_nat is not None:      # E2
                    m2, _, sc2 = operational_phase(pipe, X_nat, y_nat)
                    run.nat_metrics, run.nat_scores = m2, sc2
                    self.rows["natural"].append(self._row(attack, algo, m2, {"attack_share": float(np.mean(y_nat))}))

                # E1: без редукция, същите хиперпараметри
                pipe_full, t_fit_f, n_fit_f = fit_final(algo, params, X_tr, y_tr, reduced=False)
                mf, _, _ = operational_phase(pipe_full, X_te, y_te)
                self.rows["full"].append(self._row(attack, algo, mf, {
                    "fit_time_s": t_fit_f, "n_train": n_fit_f, "n_features": X_tr.shape[1]}))
                if algo == "RF":
                    self.importances[attack] = pd.Series(pipe_full.named_steps["clf"].feature_importances_,
                                                         index=X_tr.columns)
                    Z = StandardScaler().fit_transform(FeatureFilter().fit(X_tr).transform(X_tr))
                    self.pca_curves[attack] = PCA(random_state=C.SEED).fit(Z).explained_variance_ratio_

                # E5: другият метод за редукция (PCA <-> подбор на признаци), същите хиперпараметри
                other = "pca" if C.REDUCTION == "select" else "select"
                pipe_o, t_o, n_o = fit_final(algo, params, X_tr, y_tr, method=other)
                mo, _, _ = operational_phase(pipe_o, X_te, y_te)
                self.rows["other_reduction"].append(self._row(attack, algo, mo, {
                    "method": other, "fit_time_s": t_o, "n_train": n_o, "n_reduced": reduction_info(pipe_o)["n_reduced"]}))
                # диагностика на преобучението: точност върху обучаващите данни
                Xf, yf = X_tr, y_tr
                if ALGORITHMS[algo].train_max and len(X_tr) > ALGORITHMS[algo].train_max:
                    Xf, yf = X_tr.iloc[:ALGORITHMS[algo].train_max], y_tr.iloc[:ALGORITHMS[algo].train_max]
                m_tr = compute_metrics(np.asarray(yf), pipe.predict(Xf))
                self.rows["overfit"].append({"attack": attack, "algo": algo, "train_accuracy": m_tr["Accuracy"],
                                             "train_f1": m_tr["F1"], "test_accuracy": m["Accuracy"], "test_f1": m["F1"],
                                             "cv_f1": self.cv_summary[name]["cv_f1_mean"],
                                             "gap_accuracy": m_tr["Accuracy"] - m["Accuracy"]})

                # E4: без времевите признаци на потока (продължителност, IAT, активност)
                drop = tuple(f for f in X_tr.columns if any(t in f for t in C.TIME_LIKE_PATTERNS))
                if drop:
                    pipe_ni, t_ni, _ = fit_final(algo, params, X_tr, y_tr, drop=drop)
                    mn, _, _ = operational_phase(pipe_ni, X_te, y_te)
                    self.rows["no_iat"].append(self._row(attack, algo, mn, {"fit_time_s": t_ni, "dropped": ",".join(drop)}))

            first = f"{self.algos[0]}_{attack}"
            plots.correlation_heatmap(X_tr, attack, self.red_infos[first]["dropped_low_variance"]
                                      + self.red_infos[first]["dropped_correlated"], idx)

    # ------------------------------------------------------------------ 3. разширени анализи
    def run_extended(self):
        ci_rows, mcn, lc, ps, ops, curves, subs, pis, proj = [], {}, [], [], [], {}, [], [], {}
        rps = []
        for attack in self.attacks:
            task = self.tasks[attack]
            y_te = np.asarray(task["y_test"])
            log.info("=" * 70)
            log.info("РАЗШИРЕНИ АНАЛИЗИ: %s", attack)
            for algo in self.algos:
                run = self.runs[(attack, algo)]
                ci = S.bootstrap_ci(y_te, run.y_pred, run.scores)
                for met, v in ci.items():
                    if isinstance(v, dict):
                        ci_rows.append({"attack": attack, "algo": algo, "metric": met, **v})
                lc.append(A.learning_curve(algo, run.params, task).assign(attack=attack))
                for meth in ("select", "pca"):
                    ps.append(A.pca_sweep(algo, run.params, task, method=meth).assign(attack=attack, method=meth))
                if run.nat_scores is not None:
                    X_nat, y_nat = self.nat[attack]
                    thr0 = ALGORITHMS[algo].threshold
                    tab, cv = A.threshold_analysis(y_nat, run.nat_scores, thr0, algo)
                    ops.append(tab.assign(attack=attack))
                    rps.append(A.recall_points(y_nat, run.nat_scores, algo).assign(attack=attack))
                    curves[(attack, algo)] = cv
                if len(task["stats"]["attack_labels_included"]) > 1:
                    subs.append(A.subtype_detection(task["labels_test"], y_te, run.y_pred, algo).assign(attack=attack))
                pis.append(A.permutation_importances(run.pipe, task["X_test"], task["y_test"], algo, attack))
            # сдвоени сравнения на всички двойки алгоритми (McNemar + bootstrap разлики)
            for i, a in enumerate(self.algos):
                for b in self.algos[i + 1:]:
                    ra, rb = self.runs[(attack, a)], self.runs[(attack, b)]
                    mcn[f"{attack}:{a}-{b}"] = {
                        "attack": attack, "A": a, "B": b, **S.mcnemar(y_te, ra.y_pred, rb.y_pred),
                        "accuracy_diff": S.paired_bootstrap_diff(y_te, ra.y_pred, rb.y_pred, "Accuracy"),
                        "f1_diff": S.paired_bootstrap_diff(y_te, ra.y_pred, rb.y_pred, "F1"),
                        "precision_diff": S.paired_bootstrap_diff(y_te, ra.y_pred, rb.y_pred, "Precision"),
                        "sensitivity_diff": S.paired_bootstrap_diff(y_te, ra.y_pred, rb.y_pred, "Sensitivity")}
            log.info("A2: 2D проекции")
            proj[attack] = A.projection_2d(self.runs[(attack, self.algos[0])].pipe, task["X_train"], task["y_train"])
        log.info("A7: пренос между атаките")
        cross = A.cross_attack({k: r.pipe for k, r in self.runs.items()}, self.tasks)

        self.ext = {
            "ci": pd.DataFrame(ci_rows), "mcnemar": mcn, "lc": pd.concat(lc), "pca_sweep": pd.concat(ps),
            "ops": pd.concat(ops) if ops else pd.DataFrame(), "curves": curves,
            "recall_ops": pd.concat(rps) if rps else pd.DataFrame(),
            "subtypes": pd.concat(subs) if subs else pd.DataFrame(), "perm": pd.concat(pis),
            "proj": proj, "cross": cross}
        self._csv(self.ext["ci"], "bootstrap_ci.csv")
        save_json(mcn, self.out / "mcnemar.json")
        self._csv(self.ext["lc"], "learning_curves.csv")
        self._csv(self.ext["pca_sweep"], "pca_sweep.csv")
        self._csv(self.ext["ops"], "threshold_operating_points.csv")
        self._csv(self.ext["recall_ops"], "threshold_recall_points.csv")
        save_json({f"{a}_{g}": v for (a, g), v in curves.items()}, self.out / "pr_curves.json")
        self._csv(self.ext["subtypes"], "ddos_subtypes.csv")
        self._csv(self.ext["perm"], "permutation_importance.csv")
        self._csv(cross, "cross_attack.csv")
        pd.concat([p.assign(attack=a) for a, p in proj.items()]).to_csv(self.out / "projections_2d.csv", index=False)

    # ------------------------------------------------------------------ 4. запис и графики
    def save(self):
        self.res = pd.DataFrame(self.rows["main"])
        self.base = pd.DataFrame(self.rows["full"])
        self._csv(self.res, "metrics.csv")
        self._csv(self.base, "metrics_full_features.csv")
        if self.rows["natural"]:
            self._csv(pd.DataFrame(self.rows["natural"]), "metrics_natural_ratio.csv")
        if self.rows["no_iat"]:
            self._csv(pd.DataFrame(self.rows["no_iat"]), "metrics_without_IAT.csv")
        if self.rows["other_reduction"]:
            self._csv(pd.DataFrame(self.rows["other_reduction"]), "metrics_other_reduction.csv")
        self._csv(pd.DataFrame(self.rows["overfit"]), "overfitting.csv")
        for attack, s in self.importances.items():
            s.sort_values(ascending=False).to_csv(self.out / f"feature_importance_{attack}.csv", header=["importance"])
        save_json({a: t["stats"] for a, t in self.tasks.items()}, self.out / "task_stats.json")
        save_json(self.red_infos, self.out / "reduction.json")
        save_json(self.cv_summary, self.out / "cv_summary.json")

    def figures(self):
        res, base = self.res, self.base
        pipeline_diagram()
        plots.class_distribution({a: t["stats"] for a, t in self.tasks.items()})
        plots.pca_variance(self.pca_curves)
        plots.confusion_matrices(res)
        plots.metric_bars(res)
        plots.roc_curves(self.rocs, self.aucs)
        plots.feature_importance(self.importances)
        plots.timing(res, base)
        plots.reduction_effect(res, base)
        if self.rows["natural"]:
            plots.metric_bars(pd.DataFrame(self.rows["natural"]), metrics=("Precision", "Sensitivity", "Specificity"),
                              fname="10_natural_ratio.png", title="Тест при естественото съотношение на класовете (E2)")
        if self.rows["no_iat"]:
            plots.metric_bars(pd.DataFrame(self.rows["no_iat"]), metrics=("Precision", "Accuracy", "Sensitivity"),
                              fname="11_without_IAT.png", title="Модели без времевите признаци на потока (E4)")
        plots_ext.cv_folds(self.cv_summary)
        plots_ext.hp_sensitivity(self.cv_tables)
        if self.extended and self.ext:
            e = self.ext
            plots_ext.separability(self.separability)
            plots_ext.feature_distributions(self.df, self.separability)
            plots_ext.projections(e["proj"])
            plots_ext.learning_curves(e["lc"])
            chosen = {a: self.red_infos[f"{self.algos[0]}_{a}"]["n_reduced"] for a in self.attacks}
            plots_ext.pca_sweep(e["pca_sweep"], chosen)
            plots_ext.bootstrap_forest(e["ci"])
            if e["curves"]:
                plots_ext.pr_curves(e["curves"], e["ops"])
            if len(e["subtypes"]):
                plots_ext.ddos_subtypes(e["subtypes"])
            plots_ext.permutation_importance(e["perm"])
            plots_ext.cross_attack(e["cross"])

    def run(self):
        t0 = time.perf_counter()
        self.prepare()
        self.run_main()
        self.save()
        if self.extended:
            self.run_extended()
        self.figures()
        cols = ["model", "TP", "TN", "FP", "FN", "Precision", "Accuracy", "Sensitivity", "F1", "Specificity", "ROC_AUC"]
        log.info("\n%s", self.res[cols].to_string(index=False))
        total = (time.perf_counter() - t0) / 60
        log.info("Общо време: %.1f min", total)
        with open(self.out / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump({"total_minutes": total, "attacks": self.attacks, "algorithms": self.algos,
                       "extended": self.extended}, f, ensure_ascii=False, indent=2)
