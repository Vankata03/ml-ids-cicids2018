"""Генерира доклада (LaTeX -> PDF) от резултатите в results/.

    python report/build_report.py            # генерира числата и таблиците и компилира PDF
    python report/build_report.py --no-pdf   # само генерираните .tex файлове

Всички числа в текста се вземат от results/ чрез макроса \\val{ключ}, затова
докладът винаги съответства на последното изпълнение на експеримента.
Изисква XeLaTeX (TeX Live) и шрифтовете PT Serif / PT Sans.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("IDS_RESULTS_DIR", ROOT / "results")).resolve()
TEX = ROOT / "report" / "latex"
GEN = TEX / "generated"
sys.path.insert(0, str(ROOT))
from src.registry import ALGORITHMS, ATTACKS  # noqa: E402
from src import config as C  # noqa: E402

ALGOS, ATTS = list(ALGORITHMS), list(ATTACKS)
try:
    import xgboost
    XGB_VERSION = xgboost.__version__
except ImportError:  # pragma: no cover
    XGB_VERSION = "--"
NAME = {k: v.name_bg for k, v in ALGORITHMS.items()}
VALUES: dict[str, str] = {}


# ---------------------------------------------------------------- форматиране
def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def num(n) -> str:
    """Цяло число с тънък интервал между хилядите: 3\\,170\\,805."""
    s = f"{int(round(float(n))):,}".replace(",", r"\,")
    return s


def pct(x, d=2) -> str:
    return f"{float(x) * 100:.{d}f}".replace(".", ",") + r"\%"


def dec(x, d=3) -> str:
    return f"{float(x):.{d}f}".replace(".", ",")


def sig(x, d=3) -> str:
    x = float(x)
    if x == 0:
        return "0"
    s = f"{x:.{d}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m.replace('.', ',')}\\cdot10^{{{int(e)}}}"
    return s.replace(".", ",")


def pval(p) -> str:
    p = float(p)
    if p < 1e-300:
        return r"$<10^{-300}$"
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return f"${p / 10**e:.1f}\\cdot10^{{{e}}}$".replace(".", "{,}")
    return f"${p:.4f}$".replace(".", "{,}")


def put(key, value):
    VALUES[key] = str(value)


def table(colspec: str, header: list[str], rows: list[list], width: str | None = None, note: str | None = None) -> str:
    env = "tabularx" if width else "tabular"
    head = " & ".join(r"\hd{" + h + "}" for h in header)
    body = []
    for i, r in enumerate(rows):
        pre = r"\rowcolor{rowalt}" if i % 2 else ""
        body.append(pre + " & ".join(str(c) for c in r) + r"\\")
    w = f"{{{width}}}" if width else ""
    out = [r"\fitbox{%", rf"\begin{{{env}}}{w}{{{colspec}}}", r"\toprule", r"\headrow " + head + r"\\", r"\midrule",
           *body, r"\bottomrule", rf"\end{{{env}}}}}%"]
    if note:
        out.append(r"\par\smallskip{\footnotesize\color{muted}" + note + "}")
    return "\n".join(out) + "\n"


def write(name: str, content: str):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text(content, encoding="utf-8")


def read_json(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------- данни
def data_values():
    ds = read_json("dataset_stats.json")
    ld, cl = ds["load"], ds["clean"]
    put("rows-read", num(ld["rows_read_total"]))
    put("rows-read-mln", dec(ld["rows_read_total"] / 1e6, 2))
    put("n-files", ld["files"])
    put("n-features", ld["n_features_raw"])
    put("nan-rows", num(cl["rows_with_nan_or_inf"]))
    put("dup-rows", num(cl["duplicates_removed"]))
    put("dup-share", pct(cl["duplicates_removed"] / cl["rows_before"], 1))
    put("rows-clean", num(cl["rows_after"]))
    task = read_json("task_stats.json")
    for a, t in task.items():
        put(f"nat-share-{a}", pct(t["natural_attack_share"], 1))
        put(f"train-rows-{a}", num(t["train_rows"]))
        put(f"test-rows-{a}", num(t["test_rows"]))
        put(f"per-class-{a}", num(t["per_class_sample"]))
        put(f"attack-orig-{a}", num(t["attack_rows_original"]))
        put(f"attack-avail-{a}", num(t["attack_rows_available"]))
        put(f"task-dup-{a}", num(t.get("task_duplicates_removed", 0)))
        put(f"task-conf-{a}", num(t.get("task_conflicting_rows_removed", 0)))
        put(f"task-confvec-{a}", num(t.get("task_conflicting_vectors", 0)))
    put("benign-orig", num(task[ATTS[0]]["benign_rows_original"]))
    put("benign-avail", num(task[ATTS[0]]["benign_rows_available"]))

    # таблица с класовете
    orig, after = ld["original_class_counts"], cl["class_counts_after_cleaning"]
    rows = []
    def task_of(lbl):
        if lbl == C.BENIGN_LABEL:
            return "нормален (всички задачи)"
        return next((a for a, s in ATTACKS.items() if s(lbl)), "--")
    for lbl in sorted(orig, key=lambda k: (k != C.BENIGN_LABEL, list(ATTACKS).index(task_of(k)) if task_of(k) in ATTACKS else -1, -orig[k])):
        rows.append([r"\texttt{" + esc(lbl) + "}", task_of(lbl), num(orig[lbl]), num(after.get(lbl, 0)),
                     pct(1 - after.get(lbl, 0) / orig[lbl], 1)])
    rows.append([r"\textbf{Общо}", "", r"\textbf{" + num(ld["rows_read_total"]) + "}",
                 r"\textbf{" + num(cl["rows_after"]) + "}", pct(1 - cl["rows_after"] / cl["rows_before"], 1)])
    write("tab_class_counts", table("@{}llrrr@{}", ["Етикет", "Задача", "Изтеглени", "След почистване", "Премахнати"], rows))

    env = read_json("environment.json")
    rows = [["Процесор", esc(env.get("cpu_model", env.get("processor", ""))) + f" ({env.get('cpu_count')} ядра)"],
            ["Оперативна памет", f"{dec(env.get('ram_gb', 0), 1)} GB"],
            ["Операционна система", esc(env.get("platform", ""))],
            ["Език", f"Python {env['python']}"],
            ["Анализ на данни", f"pandas {env['pandas']}, NumPy {env['numpy']}, SciPy {env.get('scipy', '')}"],
            ["Машинно обучение", f"scikit-learn {env['scikit-learn']} (RandomForestClassifier, SVC/LIBSVM, PCA), XGBoost {XGB_VERSION}"],
            ["Визуализация", "matplotlib, seaborn"],
            ["Документация", "XeLaTeX (този доклад), pptxgenjs (презентация)"],
            ["Тестове", "pytest (8 теста върху реални записи)"]]
    write("tab_environment", table("@{}lL@{}", ["Компонент", "Версия / описание"], rows, width=r"\textwidth"))


def reduction_values():
    red = read_json("reduction.json")
    rows = []
    for a in ATTS:
        r = red[f"{ALGOS[0]}_{a}"]  # филтърът и подборът зависят само от обучаващите данни на задачата
        put(f"filtered-{a}", r["n_features_after_filter"])
        put(f"lowvar-{a}", len(r["dropped_low_variance"]))
        put(f"sel-share-{a}", pct(r.get("selected_importance_share", 0), 1))
        rk = r.get("importance_ranking", [])
        put(f"sel-top-{a}", esc(rk[0]["feature"]) if rk else "--")
        put(f"sel-topv-{a}", pct(rk[0]["importance"], 1) if rk else "--")
        put(f"sel-top3-{a}", esc(", ".join(x["feature"] for x in rk[:3])))
        rows.append([a, f"{r['n_features_in']}", f"{len(r['dropped_low_variance'])}", f"{r['n_features_after_filter']}",
                     f"{r['n_reduced']}", pct(r.get("selected_importance_share", 0), 1),
                     r"\scriptsize\texttt{" + esc(", ".join(r.get("selected_features", []))) + "}"])
    r0 = red[f"{ALGOS[0]}_{ATTS[0]}"]
    put("n-filtered", r0["n_features_after_filter"])
    put("n-selected", r0["n_reduced"])
    put("lowvar-names", esc(", ".join(r0["dropped_low_variance"])))
    fl = sorted({red[k]["n_features_after_filter"] for k in red})
    put("n-filtered-range", "--".join(map(str, [fl[0], fl[-1]])) if len(fl) > 1 else str(fl[0]))
    write("tab_reduction", table(r"@{}lccccc>{\raggedright\arraybackslash}p{6.2cm}@{}",
                                 ["Задача", "Вход", r"Нулева\\дисперсия", r"След\\филтър", "Избрани", r"Дял на\\важността", "Избрани признаци"], rows))
    # контролен метод (PCA) от E5
    oth = csv("metrics_other_reduction.csv")
    if not oth.empty and "n_reduced" in oth:
        for a in ATTS:
            d = oth[oth.attack == a]
            if len(d):
                put(f"pca-{a}", int(d.n_reduced.iloc[0]))


def model_values():
    res = csv("metrics.csv")
    full = csv("metrics_full_features.csv")
    nat = csv("metrics_natural_ratio.csv")
    noiat = csv("metrics_without_IAT.csv")
    cv = read_json("cv_summary.json")
    rows, rows2 = [], []
    for _, r in res.iterrows():
        k = f"{r.algo}-{r.attack}"
        for m in ["TP", "TN", "FP", "FN"]:
            put(f"{k}-{m}", num(r[m]))
        for m in ["Precision", "Accuracy", "Sensitivity", "Specificity", "F1", "FPR", "FNR"]:
            put(f"{k}-{m}", pct(r[m], 2))
            put(f"{k}-{m}-3", pct(r[m], 3))
        put(f"{k}-AUC", dec(r.ROC_AUC, 4))
        put(f"{k}-AUC-5", dec(r.ROC_AUC, 5))
        put(f"{k}-fit", dec(r.fit_time_s, 1))
        put(f"{k}-pred-us", dec(r.predict_time_per_record_us, 1))
        put(f"{k}-ntrain", num(r.n_train))
        put(f"{k}-throughput", num(1e6 / r.predict_time_per_record_us))
        rows.append([NAME[r.algo], r.attack, num(r.TP), num(r.TN), num(r.FP), num(r.FN),
                     pct(r.Precision), pct(r.Accuracy), pct(r.Sensitivity)])
        rows2.append([NAME[r.algo], r.attack, pct(r.Specificity), pct(r.F1), pct(r.FPR, 3), dec(r.ROC_AUC, 4),
                      dec(r.fit_time_s, 1), dec(r.predict_time_per_record_us, 1), num(1e6 / r.predict_time_per_record_us)])
        c = cv[f"{r.algo}_{r.attack}"]
        put(f"{k}-cv", pct(c["cv_f1_mean"], 2))
        put(f"{k}-cv-std", pct(c["cv_f1_std"], 2))
        put(f"{k}-cv-time", dec(c["cv_time_s"], 0))
        put(f"{k}-params", esc(", ".join(f"{p.replace('clf__', '')}={v}" for p, v in c["best_params"].items())))
    rows = [[NAME[r.algo], r.attack, num(r.TP), num(r.TN), num(r.FP), num(r.FN), pct(r.Accuracy), pct(r.Precision),
             pct(r.Sensitivity), pct(r.F1), pct(r.Specificity)] for _, r in res.iterrows()]
    write("tab_main", table("@{}llrrrrrrrrr@{}", ["Модел", "Атака", "TP", "TN", "FP", "FN", "Accuracy", "Precision",
                                                 r"Recall\\(Sensitivity)", "F1", "Specificity"], rows))
    # Тълкуване „на 100 записа“ (по учебния материал „Метрика“)
    def per100(v):
        return f"{int(round(v * 100))} от 100"
    spec = [("Accuracy", "Accuracy", "правилни решения от 100 тестови записа"),
            ("Precision", "Precision", "истински атаки от 100 вдигнати аларми"),
            ("Recall", "Sensitivity", "открити атаки от 100 реални атаки"),
            ("Specificity", "Specificity", "нормални записи без аларма от 100 нормални")]
    cols = [f"{NAME[r.algo]} {r.attack}" for _, r in res.iterrows()]
    rows_p = []
    for lbl, col, meaning in spec:
        rows_p.append([lbl, meaning] + [per100(r[col]) for _, r in res.iterrows()])
        for _, r in res.iterrows():
            put(f"{r.algo}-{r.attack}-{lbl}-100", str(int(round(r[col] * 100))))
    write("tab_per100", table("@{}ll" + "r" * len(cols) + "@{}", ["Метрика", "Какво означава"] + cols, rows_p))
    write("tab_extra", table("@{}llrrrrrrr@{}", ["Модел", "Атака", "Specificity", "F1", "FPR", "ROC-AUC", "Обуч., s",
                                               r"µs/запис", r"записа/s"], rows2))

    rows = []
    for name, c in cv.items():
        g, a = name.split("_", 1)
        sc = c["cv_fold_scores"]
        rows.append([NAME[g], a, r"\scriptsize\texttt{" + esc(", ".join(f"{p.replace('clf__', '')}={v}" for p, v in c["best_params"].items())) + "}",
                     pct(c["cv_f1_mean"]), pct(c.get("cv_f1_top_mean", c["cv_f1_mean"])), pct(c["cv_f1_std"]),
                     f"{pct(min(sc))} -- {pct(max(sc))}", dec(c["cv_time_s"], 0)])
    write("tab_cv", table("@{}llp{4.4cm}rrrrr@{}", ["Модел", "Атака", "Избрани хиперпараметри", r"CV F1\\избрана", r"CV F1\\най-висока",
                                                     "Ст. откл.", "Мин.--макс. по фолд", "Време, s"], rows,
                          note="Избраната конфигурация е най-простата в рамките на една стандартна грешка от най-високата CV оценка (правило 1-SE)."))

    # E1
    rows = []
    for _, r in res.iterrows():
        f = full[(full.attack == r.attack) & (full.algo == r.algo)].iloc[0]
        k = f"{r.algo}-{r.attack}"
        dF1 = (r.F1 - f.F1) * 100
        put(f"{k}-full-F1", pct(f.F1, 3))
        put(f"{k}-dF1", dec(dF1, 2))
        put(f"{k}-full-fit", dec(f.fit_time_s, 1))
        put(f"{k}-full-pred-us", dec(f.predict_time_per_record_us, 1))
        put(f"{k}-speedup", dec(f.predict_time_per_record_us / r.predict_time_per_record_us, 2))
        rows.append([NAME[r.algo], r.attack, pct(f.F1, 3), pct(r.F1, 3), (r"$+$" if dF1 >= 0 else r"$-$") + dec(abs(dF1), 2),
                     f"{dec(f.fit_time_s, 1)} / {dec(r.fit_time_s, 1)}", f"{dec(f.predict_time_per_record_us, 1)} / {dec(r.predict_time_per_record_us, 1)}"])
    write("tab_e1", table("@{}llrrrrr@{}", ["Модел", "Атака", "F1 пълен", "F1 редуциран", "Разлика, пр. п.", "Обучение, s", "µs/запис"], rows,
                          note="Времената са във формат „пълен / редуциран“ вектор."))

    # E2
    rows = []
    for _, r in nat.iterrows():
        k = f"{r.algo}-{r.attack}"
        for m in ["Precision", "Sensitivity", "Specificity", "Accuracy", "F1"]:
            put(f"{k}-nat-{m}", pct(r[m], 1))
        for m in ["TP", "FP", "FN", "TN"]:
            put(f"{k}-nat-{m}", num(r[m]))
        put(f"{k}-nat-n", num(r.n_test))
        put(f"{k}-nat-fp-per-tp", dec(r.FP / max(r.TP, 1), 2))
        rows.append([NAME[r.algo], r.attack, pct(r.attack_share, 1), num(r.n_test), num(r.TP), num(r.FP), num(r.FN),
                     pct(r.Precision, 1), pct(r.Sensitivity, 1), pct(r.Specificity, 1)])
    write("tab_e2", table("@{}llrrrrrrrr@{}", ["Модел", "Атака", "Дял атаки", "Записи", "TP", "FP", "FN", "Precision", "Sensitivity", "Specificity"], rows))

    # E4
    rows = []
    for _, r in noiat.iterrows():
        b = res[(res.attack == r.attack) & (res.algo == r.algo)].iloc[0]
        d = (r.F1 - b.F1) * 100
        put(f"{r.algo}-{r.attack}-noiat-F1", pct(r.F1, 3))
        put(f"{r.algo}-{r.attack}-noiat-dF1", dec(d, 2))
        rows.append([NAME[r.algo], r.attack, pct(b.F1, 3), pct(r.F1, 3), (r"$+$" if d >= 0 else r"$-$") + dec(abs(d), 2),
                     pct(r.Precision), pct(r.Sensitivity)])
    oth = csv("metrics_other_reduction.csv")
    if not oth.empty:
        rows = []
        for _, r in oth.iterrows():
            b = res[(res.attack == r.attack) & (res.algo == r.algo)].iloc[0]
            d = (b.F1 - r.F1) * 100
            put(f"{r.algo}-{r.attack}-pca-F1", pct(r.F1, 3))
            put(f"{r.algo}-{r.attack}-pca-dF1", dec(d, 2))
            rows.append([NAME[r.algo], r.attack, int(r.n_reduced), pct(r.F1, 3), pct(b.F1, 3),
                         (r"$+$" if d >= 0 else r"$-$") + dec(abs(d), 2)])
        write("tab_e5", table("@{}llrrrr@{}", ["Модел", "Атака", "PCA компоненти", "F1 с PCA", "F1 с подбор", r"Подбор $-$ PCA, пр. п."], rows))
    ov = csv("overfitting.csv")
    if not ov.empty:
        rows = []
        for _, r in ov.iterrows():
            k = f"{r.algo}-{r.attack}"
            put(f"{k}-train-acc", pct(r.train_accuracy, 3))
            put(f"{k}-gap", dec(r.gap_accuracy * 100, 3))
            rows.append([NAME[r.algo], r.attack, pct(r.train_accuracy, 3), pct(r.cv_f1, 3), pct(r.test_accuracy, 3),
                         pct(r.test_f1, 3), (r"$+$" if r.gap_accuracy >= 0 else r"$-$") + dec(abs(r.gap_accuracy) * 100, 3)])
        put("gap-max", dec(ov.gap_accuracy.abs().max() * 100, 3))
        write("tab_overfit", table("@{}llrrrrr@{}", ["Модел", "Атака", r"Accuracy\\обучение", r"F1\\5-CV", r"Accuracy\\тест", r"F1\\тест",
                                                     r"Разлика обуч.$-$тест,\\пр. п."], rows))
    write("tab_e4", table("@{}llrrrrr@{}", ["Модел", "Атака", r"F1 всички\\признаци", r"F1 без времеви", "Разлика, пр. п.", "Precision", "Sensitivity"], rows))
    return res


def stats_values():
    ci = csv("bootstrap_ci.csv")
    if ci.empty:
        return
    rows = []
    for (a, g), d in ci.groupby(["attack", "algo"], sort=False):
        d = d.set_index("metric")
        for m in d.index:
            put(f"{g}-{a}-{m}-lo", pct(d.loc[m, "ci_low"], 2))
            put(f"{g}-{a}-{m}-hi", pct(d.loc[m, "ci_high"], 2))
            put(f"{g}-{a}-{m}-hw", dec((d.loc[m, "ci_high"] - d.loc[m, "ci_low"]) * 50, 2))
        cell = lambda m: (r"\makecell{" + pct(d.loc[m, 'value']) + r"\\[-1pt]{\scriptsize\color{muted}[" + pct(d.loc[m, 'ci_low'])
                          + "; " + pct(d.loc[m, 'ci_high']) + "]}}")
        rows.append([NAME[g], a, *[cell(m) for m in ["Precision", "Accuracy", "Sensitivity", "F1"]]])
    write("tab_ci", table("@{}llcccc@{}", ["Модел", "Атака", "Precision", "Accuracy", "Sensitivity", "F1"], rows,
                          note="В квадратни скоби: 95\\% перцентилен bootstrap интервал по формула (\\ref{eq:bootstrap}), 1000 повторни извадки."))
    mc = read_json("mcnemar.json")
    rows = []
    for key, m in mc.items():
        a, pair = key.split(":")
        A, B = pair.split("-")
        k = f"{a}-{A}-{B}"
        put(f"mcn-{k}-b", num(m["only_A_correct"]))
        put(f"mcn-{k}-c", num(m["only_B_correct"]))
        put(f"mcn-{k}-both-wrong", num(m["both_wrong"]))
        put(f"mcn-{k}-chi2", dec(m["statistic"], 1))
        put(f"mcn-{k}-p", pval(m["p_value"]))
        put(f"mcn-{k}-sig", "значима" if m["p_value"] < 0.05 else "незначима")
        for kk in ["accuracy_diff", "f1_diff", "precision_diff", "sensitivity_diff"]:
            if kk in m:
                v = m[kk]
                put(f"mcn-{k}-{kk}", dec(v["diff"] * 100, 3))
                put(f"mcn-{k}-{kk}-ci", f"[{dec(v['ci_low'] * 100, 3)}; {dec(v['ci_high'] * 100, 3)}]")
        dd = lambda kk: (r"\makecell{" + dec(m[kk]['diff'] * 100, 3) + r"\\[-1pt]{\scriptsize\color{muted}[" + dec(m[kk]['ci_low'] * 100, 3)
                         + "; " + dec(m[kk]['ci_high'] * 100, 3) + "]}}")
        rows.append([a, f"{A} / {B}", num(m["both_correct"]), num(m["only_A_correct"]), num(m["only_B_correct"]), num(m["both_wrong"]),
                     dec(m["statistic"], 2), pval(m["p_value"]), dd("f1_diff"), dd("sensitivity_diff")])
    put("mcn-n-sig", sum(1 for m in mc.values() if m["p_value"] < 0.05))
    put("mcn-n", len(mc))
    write("tab_mcnemar", table("@{}llrrrrrrrr@{}", ["Атака", r"Двойка\\A / B", r"И двата\\верни", r"Само A\\верен ($b$)", r"Само B\\верен ($c$)",
                         r"И двата\\грешни", r"$\chi^2$", "$p$", r"$\Delta$F1", r"$\Delta$Sensitivity"], rows,
                         note=r"$\Delta$ = A $-$ B в процентни пунктове, в скоби 95\% сдвоен bootstrap интервал."))


def extended_values():
    ks = csv("ks_separability.csv")
    if not ks.empty:
        for a in ATTS:
            d = ks[ks.attack == a].sort_values("KS_D", ascending=False)
            put(f"ks-top-{a}", esc(d.feature.iloc[0]))
            put(f"ks-topD-{a}", dec(d.KS_D.iloc[0], 2))
            put(f"ks-top5-{a}", dec(d.KS_D.head(5).mean(), 2))
            put(f"ks-top5names-{a}", esc(", ".join(d.feature.head(5))))
            put(f"ks-n05-{a}", int((d.KS_D > 0.5).sum()))
            put(f"ks-median-{a}", dec(d.KS_D.median(), 2))
    lc = csv("learning_curves.csv")
    if not lc.empty:
        for (a, g), d in lc.groupby(["attack", "algo"]):
            d = d.sort_values("n_train")
            put(f"lc-{g}-{a}-first", pct(d.F1.iloc[0], 2))
            put(f"lc-{g}-{a}-last", pct(d.F1.iloc[-1], 2))
            put(f"lc-{g}-{a}-nfirst", num(d.n_train.iloc[0]))
            put(f"lc-{g}-{a}-nlast", num(d.n_train.iloc[-1]))
            put(f"lc-{g}-{a}-tlast", dec(d.fit_time_s.iloc[-1], 1))
            put(f"lc-{g}-{a}-gain-last", dec((d.F1.iloc[-1] - d.F1.iloc[-2]) * 100, 2) if len(d) > 1 else "0")
        rows = []
        for (a, g), d in lc.groupby(["attack", "algo"], sort=False):
            for _, r in d.sort_values("n_train").iterrows():
                rows.append([a, NAME[g], num(r.n_train), pct(r.F1), pct(r.Precision), pct(r.Sensitivity), dec(r.fit_time_s, 2)])
        write("tab_lc", table("@{}llrrrrr@{}", ["Атака", "Модел", "Обучаващи записи", "F1", "Precision", "Sensitivity", "Обучение, s"], rows))
    ps = csv("pca_sweep.csv")
    if not ps.empty:
        rows = []
        mlab = {"select": "подбор", "pca": "PCA"}
        for (a, g, meth), d in ps.groupby(["attack", "algo", "method"], sort=False):
            red, fu = d[d.reduced], d[~d.reduced]
            best = red.loc[red.F1.idxmax()]
            put(f"ps-{meth}-{g}-{a}-k2", pct(red[red.n_components == red.n_components.min()].F1.iloc[0], 2))
            put(f"ps-{meth}-{g}-{a}-best", pct(best.F1, 2))
            put(f"ps-{meth}-{g}-{a}-bestk", int(best.n_components))
            k5 = red[red.n_components == 5]
            if len(k5):
                put(f"ps-{meth}-{g}-{a}-k5", pct(k5.F1.iloc[0], 2))
            if len(fu):
                put(f"ps-{g}-{a}-full", pct(fu.F1.iloc[0], 2))
            for _, r in d.iterrows():
                rows.append([a, NAME[g], mlab.get(meth, meth), "без редукция" if not r.reduced else int(r.n_components),
                             pct(r.explained, 1) if r.reduced else "--", pct(r.F1), dec(r.fit_time_s, 2), dec(r.predict_us_per_record, 1)])
        write("tab_ps", table("@{}lllrrrrr@{}", ["Атака", "Модел", "Метод", "Измерения", r"Дял важност /\\дисперсия", "F1", "Обучение, s", "µs/запис"], rows))
    ops = csv("threshold_operating_points.csv")
    if not ops.empty:
        rows = []
        for _, r in ops.iterrows():
            key = f"op-{r.algo}-{r.attack}-{int(round(r.target_precision * 100))}"
            put(f"{key}-recall", pct(r.recall, 1))
            put(f"{key}-alarms", dec(r.alarms_per_10k, 0))
            rows.append([r.attack, NAME[r.algo], pct(r.target_precision, 0), pct(r.precision, 1) if pd.notna(r.precision) else "--",
                         pct(r.recall, 1), sig(r.threshold, 3) if pd.notna(r.threshold) else "--", dec(r.alarms_per_10k, 0)])
        write("tab_ops", table("@{}llrrrrr@{}", ["Атака", "Модел", "Цел Precision", "Постигнат", "Recall", r"Праг $\theta$", "Аларми / 10\\,000"], rows))
        pr = read_json("pr_curves.json")
        for k, v in pr.items():
            a, g = k.split("_")[0], k.split("_")[1]
            put(f"ap-{g}-{a}", dec(v["average_precision"], 3))
    rops = csv("threshold_recall_points.csv")
    if not rops.empty:
        rows = []
        for _, r in rops.iterrows():
            key = f"rop-{r.algo}-{r.attack}-{int(round(r.target_recall * 100))}"
            put(f"{key}-precision", pct(r.precision, 1))
            put(f"{key}-spec", pct(r.specificity, 1))
            put(f"{key}-alarms", dec(r.alarms_per_10k, 0))
            put(f"{key}-fp-per-tp", dec(r.FP / max(r.TP, 1), 1))
            rows.append([r.attack, NAME[r.algo], pct(r.target_recall, 0), pct(r.recall, 1), pct(r.precision, 1),
                         pct(r.specificity, 1), dec(r.alarms_per_10k, 0)])
        write("tab_recall_ops", table("@{}llrrrrr@{}", ["Атака", "Модел", "Цел Recall", "Постигнат Recall", "Precision",
                                                        "Specificity", "Аларми / 10\\,000"], rows))
    # обединена таблица на работните точки за най-трудната атака (по Precision и по Recall)
    if not ops.empty and not rops.empty:
        hard = csv("metrics.csv").groupby("attack").F1.min().idxmin()
        put("hard-attack", hard)
        rows = []
        for g in ALGOS:
            for _, r in ops[(ops.attack == hard) & (ops.algo == g) & (ops.target_precision.isin([0.9, 0.95, 0.99]))].iterrows():
                rows.append([NAME[g], r"Precision $\geq$ " + pct(r.target_precision, 0), pct(r.precision, 1), pct(r.recall, 1),
                             dec(r.alarms_per_10k, 0)])
            for _, r in rops[(rops.attack == hard) & (rops.algo == g)].iterrows():
                rows.append([NAME[g], r"Recall $\geq$ " + pct(r.target_recall, 0), pct(r.precision, 1), pct(r.recall, 1),
                             dec(r.alarms_per_10k, 0)])
        write("tab_ops_ps", table("@{}llrrr@{}", ["Модел", "Критерий за праг", "Precision", "Recall", "Аларми / 10\\,000"], rows))
    sub = csv("ddos_subtypes.csv")
    if not sub.empty:
        rows = []
        piv = sub.pivot_table(index=["attack", "label"], columns="algo", values=["n", "missed", "detection_rate"])
        for (a, lbl) in piv.index:
            rows.append([a, r"\texttt{" + esc(lbl) + "}", num(piv.loc[(a, lbl), ("n", ALGOS[0])]),
                         *[f"{int(piv.loc[(a, lbl), ('missed', g)])} ({pct(piv.loc[(a, lbl), ('detection_rate', g)], 2)})" for g in ALGOS]])
            for g in ALGOS:
                put(f"sub-{g}-{lbl.replace(' ', '')}-missed", int(piv.loc[(a, lbl), ("missed", g)]))
                put(f"sub-{g}-{lbl.replace(' ', '')}-rate", pct(piv.loc[(a, lbl), ("detection_rate", g)], 2))
            put(f"sub-{lbl.replace(' ', '')}-n", num(piv.loc[(a, lbl), ("n", ALGOS[0])]))
        write("tab_subtypes", table("@{}llr" + "r" * len(ALGOS) + "@{}", ["Атака", "Разновидност", "Тестови записи"]
                                    + [f"Пропуснати {g} (открити)" for g in ALGOS], rows))
    ca = csv("cross_attack.csv")
    if not ca.empty:
        rows = []
        for _, r in ca.iterrows():
            put(f"ca-{r.algo}-{r.train_attack}-on-{r.test_attack}", pct(r.detection_rate, 1))
            put(f"ca-{r.algo}-{r.train_attack}-on-{r.test_attack}-fa", pct(r.false_alarm_rate, 2))
            rows.append([NAME[r.algo], r.train_attack, r.test_attack, pct(r.detection_rate, 2), pct(r.false_alarm_rate, 2)])
        write("tab_cross", table("@{}lllrr@{}", ["Модел", "Обучен за", "Тестван върху", "Открити атаки", "Фалшиви аларми"], rows))
    pi = csv("permutation_importance.csv")
    if not pi.empty:
        for (a, g), d in pi.groupby(["attack", "algo"]):
            d = d.sort_values("importance_mean", ascending=False)
            put(f"pi-{g}-{a}-top", esc(d.feature.iloc[0]))
            put(f"pi-{g}-{a}-topv", dec(d.importance_mean.iloc[0] * 100, 2))
            put(f"pi-{g}-{a}-top3", esc(", ".join(d.feature.head(3))))
    for a in ATTS:
        fi = RES / f"feature_importance_{a}.csv"
        if fi.exists():
            d = pd.read_csv(fi, index_col=0)["importance"]
            put(f"mdi-{a}-top3", esc(", ".join(d.index[:3])))
            put(f"mdi-{a}-top5share", pct(d.iloc[:5].sum(), 0))


def robustness_values():
    """A10-A14: проверки за приложимост (results/robustness/)."""
    rd = RES / "robustness"
    if not (rd / "split_features.csv").exists():
        return
    sf = pd.read_csv(rd / "split_features.csv")
    od = pd.read_csv(rd / "other_day_benign.csv")
    lo = pd.read_csv(rd / "held_out_day_benign.csv")
    st = pd.read_csv(rd / "subtype_transfer.csv")
    hm = pd.read_csv(rd / "hybrid_main.csv")
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    N4 = {**NAME, "XGB|SVM": "XGBoost или SVM"}
    key = lambda g: g.replace("|", "or")
    rows = []
    for a in ATTS:
        for g in ALGOS + ["XGB|SVM"]:
            cells = []
            for sp in ("random", "chronological"):
                for fs in ("all", "no_fp"):
                    r = sf[(sf.attack == a) & (sf.algo == g) & (sf.split == sp) & (sf.features == fs)].iloc[0]
                    put(f"rob-{sp}-{fs}-{key(g)}-{a}-acc", pct(r.Accuracy, 2))
                    put(f"rob-{sp}-{fs}-{key(g)}-{a}-sens", pct(r.Sensitivity, 1))
                    put(f"rob-{sp}-{fs}-{key(g)}-{a}-fpr", pct(r.FPR, 2))
                    cells.append(f"{pct(r.Accuracy, 2)} ({pct(r.Sensitivity, 1)})")
            rows.append([a, N4[g], *cells])
    write("tab_rob_split", table("@{}llrrrr@{}", ["Атака", "Модел", r"Случайно,\\всички", r"Случайно,\\без отпечатъци",
                                                  r"Хронологично,\\всички", r"Хронологично,\\без отпечатъци"], rows))
    rnd = sf[(sf.split == "random") & (sf.algo != "XGB|SVM")].pivot_table(index=["attack", "algo"], columns="features", values="Accuracy")
    put("rob-fp-maxdrop", dec((rnd["all"] - rnd["no_fp"]).max() * 100, 2))
    ch = sf[(sf.split == "chronological") & (sf.features == "all")]
    put("rob-chrono-min-DDoSBF", pct(ch[ch.attack != "Bot"].Accuracy.min(), 2))
    for a in ATTS:
        c = meta.get(f"{a}_chronological", {})
        put(f"rob-cut-{a}", c.get("attack_test_from", "--"))
    # A13
    one = od[od.features == "all"].groupby(["attack", "algo"]).FPR.max()
    two = lo.groupby(["attack", "algo"]).FPR_held_out.max()
    rows = []
    for a in ATTS:
        for g in ALGOS + ["XGB|SVM"]:
            put(f"rob-1day-{key(g)}-{a}", pct(one[(a, g)], 2))
            put(f"rob-2day-{key(g)}-{a}", pct(two[(a, g)], 2))
            rows.append([a, N4[g], pct(one[(a, g)], 3), pct(two[(a, g)], 3)])
    write("tab_rob_days", table("@{}llrr@{}", ["Атака", "Модел", r"Нормален трафик\\от 1 ден", r"Нормален трафик\\от 2 дни"], rows))
    bp = meta["benign_profile_by_day"]
    put("benign-2102-win65535", pct(bp["Wednesday-21-02-2018"]["init_win_65535_share"], 1))
    put("benign-2102-tcp", pct(bp["Wednesday-21-02-2018"]["tcp_share"], 1))
    put("benign-other-tcp", pct(bp["Friday-02-03-2018"]["tcp_share"], 1))
    put("benign-2102-rows", num(bp["Wednesday-21-02-2018"]["rows"]))
    # A12
    sn = {"DDOS attack-HOIC": "HOIC", "DDOS attack-LOIC-UDP": "LOIC-UDP", "SSH-Bruteforce": "SSH", "FTP-BruteForce": "FTP"}
    rows = []
    for (src, dst), d in st.groupby(["train_on", "test_on"], sort=False):
        cells = []
        for fs in ("all", "no_fp"):
            for g in ALGOS:
                r = d[(d.algo == g) & (d.features == fs)].iloc[0]
                put(f"sub-{sn[src]}-{sn[dst]}-{g}-{fs}", pct(r.detected_other_subtype, 1))
                cells.append(pct(r.detected_other_subtype, 1))
        rows.append([f"{sn[src]} $\\to$ {sn[dst]}", num(d.n_train_attack.iloc[0]), num(d.n_test_attack.iloc[0]), *cells])
    write("tab_rob_sub", table("@{}lrrrrrrrr@{}", ["Обучен $\\to$ тест", "Обуч. атаки", "Тест. атаки", "RF", "SVM", "XGB",
                                                    r"RF\\без отп.", r"SVM\\без отп.", r"XGB\\без отп."], rows))
    # A14 в основната постановка
    for _, r in hm.iterrows():
        k = f"hyb-{r.split}-{r.attack}"
        put(f"{k}-acc", pct(r.Accuracy, 3)); put(f"{k}-fp", int(r.FP)); put(f"{k}-fn", int(r.FN))
        put(f"{k}-prec", pct(r.Precision, 2)); put(f"{k}-sens", pct(r.Sensitivity, 2))


def _feature_doc(f: str) -> tuple[str, str]:
    """Семейство и описание на признак на CICFlowMeter-V3 (Fwd = клиент -> сървър, Bwd = обратно)."""
    d = {"Fwd": "в права посока (клиент → сървър)", "Bwd": "в обратна посока (сървър → клиент)"}
    stat = {"Max": "максимална", "Min": "минимална", "Mean": "средна", "Std": "стандартно отклонение на",
            "Tot": "обща", "Var": "дисперсия на"}
    fixed = {
        "Protocol": ("заглавка", "номер на транспортния протокол (6 = TCP, 17 = UDP, 0 = друг)"),
        "Flow Duration": ("време", "продължителност на потока, µs"),
        "Flow Byts/s": ("интензивност", "байтове в секунда в потока"),
        "Flow Pkts/s": ("интензивност", "пакети в секунда в потока"),
        "Fwd Pkts/s": ("интензивност", "пакети в секунда в права посока"),
        "Bwd Pkts/s": ("интензивност", "пакети в секунда в обратна посока"),
        "Down/Up Ratio": ("обем", "отношение на пакетите в обратна към права посока"),
        "Pkt Size Avg": ("дължини", "среден размер на пакет"),
        "Fwd Seg Size Avg": ("дължини", "среден размер на сегмент в права посока"),
        "Bwd Seg Size Avg": ("дължини", "среден размер на сегмент в обратна посока"),
        "Fwd Seg Size Min": ("заглавка", "минимален размер на сегмента в права посока (зависи от TCP опциите)"),
        "Init Fwd Win Byts": ("заглавка", "начален размер на TCP прозореца в права посока"),
        "Init Bwd Win Byts": ("заглавка", "начален размер на TCP прозореца в обратна посока"),
        "Fwd Act Data Pkts": ("обем", "пакети с полезни данни в права посока"),
        "Fwd Header Len": ("заглавка", "обща дължина на заглавките в права посока"),
        "Bwd Header Len": ("заглавка", "обща дължина на заглавките в обратна посока"),
        "Tot Fwd Pkts": ("обем", "брой пакети в права посока"),
        "Tot Bwd Pkts": ("обем", "брой пакети в обратна посока"),
        "TotLen Fwd Pkts": ("обем", "общ размер на пакетите в права посока, байтове"),
        "TotLen Bwd Pkts": ("обем", "общ размер на пакетите в обратна посока, байтове"),
        "CWE Flag Count": ("TCP флагове", "брой пакети с флаг CWR"),
    }
    if f in fixed:
        return fixed[f]
    w = f.split()
    if f.endswith("Flag Cnt"):
        return "TCP флагове", f"брой пакети с флаг {w[0]}"
    if f.endswith("Flags") and len(w) == 3:
        return "TCP флагове", f"брой пакети с флаг {w[1]} {d[w[0]]}"
    if "IAT" in w:
        where = "в потока" if w[0] == "Flow" else d[w[0]]
        return "време", f"{stat[w[-1]]} интервал между пакетите {where}"
    if w[0] in ("Active", "Idle"):
        kind = "активност" if w[0] == "Active" else "бездействие"
        return "време", f"{stat[w[-1]]} продължителност на период на {kind}"
    if "Pkt Len" in f:
        where = d.get(w[0], "в двете посоки")
        return "дължини", f"{stat[w[-1]]} дължина на пакет {where}"
    if w[0] == "Subflow":
        return "обем", ("пакети" if w[2] == "Pkts" else "байтове") + f" в подпоток {d[w[1]]}"
    if "/b" in f or "Blk Rate" in f:
        what = {"Byts/b": "байтове на пакет в насрещен блок", "Pkts/b": "пакети в насрещен блок", "Blk": "честота на насрещни блокове"}
        return "насрещни блокове", f"средна стойност: {what[w[1]]} {d[w[0]]}"
    return "", ""


def appendix_tables():
    ds = read_json("dataset_stats.json")
    ks = csv("ks_separability.csv")
    red = read_json("reduction.json")
    rows, md_rows = [], []
    for i, f in enumerate(ds["load"]["feature_names_raw"], 1):
        fam, desc = _feature_doc(f)
        cells = [str(i), r"\texttt{" + esc(f) + "}", fam, desc]
        mcells = [str(i), f"`{f}`", fam, desc]
        for a in ATTS:
            d = ks[(ks.attack == a) & (ks.feature == f)].KS_D if not ks.empty else []
            r = red[f"{ALGOS[0]}_{a}"]
            sel = f in r.get("selected_features", [])
            mark = r"$^{\ast}$" if sel else (r"$^{\times}$" if f in r["dropped_low_variance"] else "")
            cells.append((dec(d.iloc[0], 2) if len(d) else "--") + mark)
            mcells.append((f"{d.iloc[0]:.2f}" if len(d) else "--") + (" *" if sel else (" ×" if mark else "")))
        rows.append(cells)
        md_rows.append(mcells)
    md = [f"# Речник на признаците на {C.DATASET_NAME}", "",
          "Генерирано автоматично от `report/build_report.py`. D е статистиката на Колмогоров–Смирнов "
          "между атаката и нормалния трафик; * означава, че признакът е сред 20-те избрани; × означава, "
          "че е премахнат заради нулева дисперсия.", "",
          "| № | Признак | Семейство | Описание | " + " | ".join(f"D {a}" for a in ATTS) + " |",
          "|---:|---|---|---|" + "---:|" * len(ATTS)]
    md += ["| " + " | ".join(c) + " |" for c in md_rows]
    (RES / "feature_dictionary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write("tab_features", r"{\scriptsize" + "\n" + table(r"@{}rl l L" + "r" * len(ATTS) + "@{}", ["№", "Признак", "Семейство", "Описание"]
                                    + [rf"$D$ {a}" for a in ATTS], rows, width=r"\textwidth",
                                    note=r"$D$ е статистиката на Колмогоров--Смирнов (формула~\ref{eq:ks}); $^{\ast}$ означава, че признакът е сред избраните 20 за съответната задача, а $^{\times}$, че е премахнат заради нулева дисперсия.") + "}\n")
    rows = []
    for f in sorted(RES.glob("cv_*.csv")):
        if f.name == "cv_summary.json":
            continue
        g, a = f.stem[3:].split("_", 1)
        d = pd.read_csv(f).sort_values("rank_test_score")
        pcols = [c for c in d.columns if c.startswith("param_")]
        for _, r in d.iterrows():
            p = ", ".join(f"{c.replace('param_clf__', '')}={r[c]}" for c in pcols)
            rows.append([NAME[g], a, r"\scriptsize\texttt{" + esc(p.replace("nan", "None")) + "}", int(r.rank_test_score),
                         pct(r.mean_test_score, 2), pct(r.std_test_score, 2), dec(r.mean_fit_time, 1)])
    write("tab_cv_full", r"{\small" + "\n" + table("@{}llLrrrr@{}", ["Модел", "Атака", "Хиперпараметри", "Ранг", "CV F1", "Ст. откл.", "Обуч., s"], rows, width=r"\textwidth") + "}\n")


def write_values():
    write("paths", "\\graphicspath{{" + (RES / "figures").as_posix() + "/}}\n")
    lines = ["% Автоматично генерирано от report/build_report.py - не редактирайте ръчно"]
    for k, v in sorted(VALUES.items()):
        lines.append(rf"\expandafter\def\csname val@{k}\endcsname{{{v}}}")
    write("values", "\n".join(lines) + "\n")


def check_keys():
    used = set()
    for f in (TEX / "chapters").glob("*.tex"):
        used |= set(re.findall(r"\\val\{([^}]+)\}", f.read_text(encoding="utf-8")))
    used |= set(re.findall(r"\\val\{([^}]+)\}", (TEX / "main.tex").read_text(encoding="utf-8")))
    missing = sorted(used - set(VALUES))
    if missing:
        print("ВНИМАНИЕ: липсващи стойности:", ", ".join(missing))
    return missing


def compile_pdf(out: Path):
    if not shutil.which("latexmk"):
        print("latexmk/XeLaTeX не е наличен - генерирани са само .tex файловете")
        return
    cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"]
    r = subprocess.run(cmd, cwd=TEX, capture_output=True, text=True)
    if r.returncode != 0:
        log = (TEX / "main.log").read_text(encoding="utf-8", errors="replace")
        errs = [l for l in log.splitlines() if l.startswith("!") or ".tex:" in l and "error" in l.lower()]
        print("\n".join(errs[:20]) or r.stdout[-3000:])
        raise SystemExit("Компилацията на LaTeX е неуспешна")
    shutil.copy(TEX / "main.pdf", out)
    print("written", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "report" / "Доклад_ML_IDS_CSE-CIC-IDS2018.pdf"))
    args = ap.parse_args()
    data_values()
    reduction_values()
    model_values()
    stats_values()
    extended_values()
    robustness_values()
    appendix_tables()
    write_values()
    check_keys()
    if not args.no_pdf:
        compile_pdf(Path(args.out))


if __name__ == "__main__":
    main()
