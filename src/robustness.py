"""Проверки за приложимост (анализи A10-A13).

Основните резултати са почти перфектни. Тези анализи проверяват дали това се дължи на
„отпечатъка“ на атакуващия инструмент и на случайното разделяне на записите, или моделите
наистина научават поведението на атаката:

  A10  без признаци-отпечатъци: премахват се параметрите на TCP стека на изпращача
       (Init Fwd Win Byts, Init Bwd Win Byts, Fwd Seg Size Min), които зависят от
       операционната система и инструмента, а не от самата атака;
  A11  хронологично разделяне: моделът се обучава върху по-ранните 70% от потоците в деня
       на атаката и се тества върху по-късните 30% (вместо случайно разделяне);
  A12  пренос между подвидове: обучение върху единия инструмент, тест върху другия
       (HOIC <-> LOIC-UDP при DDoS, SSH <-> FTP при грубата сила);
  A13  нормален трафик от други дни: (а) модел, обучен с нормалния трафик само от деня на
       атаката, се проверява за фалшиви аларми върху нормалния трафик от другите два дни;
       (б) модел, обучен с нормален трафик от два дни, се проверява върху третия (непознат) ден;
  A14  комбиниран детектор: запис се обявява за атака, ако XGBoost ИЛИ SVM го обяви за атака.
       Дърветата са най-точни върху познати модели на атаката, а гладката граница на SVM
       улавя и нови разновидности; комбинацията се оценява във всички горни постановки.

Всеки анализ се изпълнява с всички признаци и без признаците-отпечатъци, с хиперпараметрите,
избрани в основния експеримент (results/cv_summary.json).

Бележка за времето: в CSV файловете на CSE-CIC-IDS2018 часът е записан в 12-часов формат без
AM/PM. Атаките по официалния график са между 10:00 и 16:00, затова часовете 1-7 се приемат за
следобедни (+12 ч). Така атаките попадат точно в публикуваните интервали
(напр. LOIC-UDP 10:09-10:43 и HOIC 14:11-14:33 на 21.02.2018).

    python -m src.robustness
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import compute_metrics
from .preprocess import save_json
from .registry import ATTACKS
from .train import fit_final

log = logging.getLogger(__name__)

FINGERPRINT = ("Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min")
DAY_OF = {"DDoS": "Wednesday-21-02-2018", "BruteForce": "Wednesday-14-02-2018", "Bot": "Friday-02-03-2018"}
SUBTYPES = {"DDoS": ("DDOS attack-HOIC", "DDOS attack-LOIC-UDP"), "BruteForce": ("SSH-Bruteforce", "FTP-BruteForce")}
ALGOS = ("RF", "SVM", "XGB")
FEATURE_SETS = {"all": (), "no_fp": FINGERPRINT}
OTHER_DAY_BENIGN = 100_000


# ---------------------------------------------------------------------------
# Данни с време и ден
# ---------------------------------------------------------------------------
def load_with_time() -> pd.DataFrame:
    parts = []
    for f in sorted(C.RAW_DIR.glob("*.csv")):
        day = f.name.split("_")[0]
        d = pd.read_csv(f, low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        t = pd.to_datetime(d["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        sec = (t - t.dt.normalize()).dt.total_seconds()
        sec = np.where(sec < 8 * 3600, sec + 12 * 3600, sec)        # 12-часов запис без AM/PM
        ok = t.dt.year.eq(2018).to_numpy()                           # няколко записа с дата 1970 г.
        d = d.drop(columns=["Timestamp", "Dst Port"])
        lab = d.pop("Label").astype(str).str.strip()
        d = d.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        d["label"], d["day"], d["sec"] = lab.to_numpy(), day, sec
        parts.append(d.loc[ok])
        log.info("прочетен %s: %d записа", f.name, ok.sum())
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in ("label", "day", "sec")]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats)
    # Хронологичен ред; от повтарящите се вектори остава най-ранният (строго за времевия тест).
    df = df.sort_values(["day", "sec"], kind="stable")
    df = df.drop_duplicates(subset=feats + ["label"], keep="first").reset_index(drop=True)
    return df


def features(df):
    return [c for c in df.columns if c not in ("label", "day", "sec", "y")]


def _drop_conflicts(df):
    """Премахва вектори, срещани и като атака, и като нормален трафик, и дубликати в задачата."""
    feats = features(df)
    key = pd.util.hash_pandas_object(df[feats], index=False)
    nun = df["y"].groupby(key.values).transform("nunique").to_numpy()
    dup = key.duplicated().to_numpy()
    return df.loc[(nun == 1) & ~dup]


def _sample(df, n, seed):
    return df if len(df) <= n else df.sample(n, random_state=seed)


def task_frame(df, attack, labels=None, day_only=True):
    rule = ATTACKS[attack]
    is_att = df["label"].isin(labels) if labels else df["label"].map(rule)
    is_ben = df["label"].eq(C.BENIGN_LABEL)
    if day_only:
        is_ben &= df["day"].eq(DAY_OF[attack])
    t = df.loc[is_att | is_ben].copy()
    t["y"] = is_att.loc[t.index].astype(int)
    return _drop_conflicts(t)


# ---------------------------------------------------------------------------
# Обучение и оценка
# ---------------------------------------------------------------------------
PARAMS = json.loads((C.RESULTS_DIR / "cv_summary.json").read_text(encoding="utf-8"))


def fit_eval(algo, attack, tr, te, drop):
    feats = features(tr)
    pipe, t_fit, n_fit = fit_final(algo, PARAMS[f"{algo}_{attack}"]["best_params"], tr[feats], tr["y"], drop=drop)
    yp = pipe.predict(te[feats])
    return pipe, compute_metrics(te["y"].to_numpy(), yp), t_fit


def row(analysis, attack, algo, fs, m, **extra):
    keep = ("TP", "TN", "FP", "FN", "Precision", "Accuracy", "Sensitivity", "Specificity", "F1", "FPR")
    return {"analysis": analysis, "attack": attack, "algo": algo, "features": fs,
            **{k: m[k] for k in keep if k in m}, **extra}


def balanced_split(t, seed, chronological):
    """Балансирана извадка (до MAX_PER_CLASS на клас) и разделяне 70/30 - случайно или по време."""
    pos, neg = t[t.y == 1], t[t.y == 0]
    n = min(len(pos), len(neg), C.MAX_PER_CLASS)
    if chronological:
        # равномерна извадка по време: запазва хронологията и покритието на целия ден
        pick = lambda d: d.iloc[np.linspace(0, len(d) - 1, n).astype(int)]
        pos, neg = pick(pos.sort_values("sec")), pick(neg.sort_values("sec"))
        cut = lambda d: (d.iloc[: int(len(d) * (1 - C.TEST_SIZE))], d.iloc[int(len(d) * (1 - C.TEST_SIZE)):])
        (p_tr, p_te), (n_tr, n_te) = cut(pos), cut(neg)
        info = {"train_until": float(max(p_tr.sec.max(), n_tr.sec.max())), "test_from": float(p_te.sec.min()),
                "attack_test_from": float(p_te.sec.min()), "attack_train_until": float(p_tr.sec.max())}
        return pd.concat([p_tr, n_tr]), pd.concat([p_te, n_te]), info
    pos, neg = pos.sample(n, random_state=seed), neg.sample(n, random_state=seed)
    from sklearn.model_selection import train_test_split
    s = pd.concat([pos, neg])
    tr, te = train_test_split(s, test_size=C.TEST_SIZE, stratify=s.y, random_state=seed)
    return tr, te, {}


def hhmm(s):
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}"


def hybrid_main():
    """A14 в основната постановка: записаните основни модели XGBoost и SVM върху същите тестови
    записи като в глава 4 (балансиран тест и тест при естествено съотношение)."""
    import joblib
    from .preprocess import build_task, clean, load_raw, natural_test_set
    raw, ls = load_raw()
    d, _ = clean(raw)
    out = []
    for attack in ATTACKS:
        task = build_task(d, attack, ls["original_class_counts"])
        nat = natural_test_set(d, task, attack)
        px = joblib.load(C.MODELS_DIR / f"XGB_{attack}.joblib")
        ps = joblib.load(C.MODELS_DIR / f"SVM_{attack}.joblib")
        for test, (X, y) in (("balanced", (task["X_test"], task["y_test"])), ("natural", nat)):
            yh = px.predict(X) | ps.predict(X)
            m = compute_metrics(np.asarray(y), yh)
            out.append(row("A14", attack, "XGB|SVM", "all", m, split=test, n_test=len(X)))
            log.info("основна постановка %s %s XGB|SVM: Acc=%.5f FP=%d FN=%d", test, attack, m["Accuracy"], m["FP"], m["FN"])
    return out


def run():
    t00 = time.perf_counter()
    df = load_with_time()
    log.info("общо %d уникални записа", len(df))
    rows, subtype_rows, otherday_rows, lodo_rows, meta = [], [], [], [], {"fingerprint_features": list(FINGERPRINT)}

    # отпечатъци: стойности на началния TCP прозорец по етикет (за доклада)
    fp_tab = (df.groupby("label")["Init Fwd Win Byts"]
                .agg(lambda s: {str(int(k)): round(float(v), 4) for k, v in s.value_counts(normalize=True).head(3).items()}))
    meta["init_win_top_values"] = fp_tab.to_dict()

    for attack in ATTACKS:
        t = task_frame(df, attack)
        for split in ("random", "chronological"):
            tr, te, info = balanced_split(t, C.SEED, split == "chronological")
            if info:
                meta[f"{attack}_chronological"] = {**{k: hhmm(v) for k, v in info.items()},
                                                   "train_rows": len(tr), "test_rows": len(te),
                                                   "test_attack_labels": te[te.y == 1].label.value_counts().to_dict(),
                                                   "train_attack_labels": tr[tr.y == 1].label.value_counts().to_dict()}
            for fs, drop in FEATURE_SETS.items():
                pred = {}
                for algo in ALGOS:
                    pipe, m, t_fit = fit_eval(algo, attack, tr, te, drop)
                    pred[algo] = pipe.predict(te[features(te)]), pipe
                    rows.append(row("A10" if split == "random" else "A11", attack, algo, fs, m, split=split,
                                    n_train=len(tr), n_test=len(te)))
                    log.info("%s %s %s %s: Acc=%.5f Sens=%.5f FPR=%.5f", split, attack, algo, fs,
                             m["Accuracy"], m["Sensitivity"], m["FPR"])
                    if split == "random":
                        # A13: фалшиви аларми върху нормалния трафик от другите дни
                        for day in sorted(set(DAY_OF.values()) - {DAY_OF[attack]}):
                            b = _sample(df[(df.label == C.BENIGN_LABEL) & (df.day == day)], OTHER_DAY_BENIGN, C.SEED)
                            fp = int(pipe.predict(b[features(b)]).sum())
                            otherday_rows.append({"attack": attack, "algo": algo, "features": fs, "benign_day": day,
                                                  "n": len(b), "FP": fp, "FPR": fp / len(b),
                                                  "FPR_same_day": m["FPR"]})
                yh = pred["XGB"][0] | pred["SVM"][0]
                mh = compute_metrics(te["y"].to_numpy(), yh)
                rows.append(row("A14", attack, "XGB|SVM", fs, mh, split=split, n_train=len(tr), n_test=len(te)))
                log.info("%s %s XGB|SVM %s: Acc=%.5f Sens=%.5f FPR=%.5f", split, attack, fs,
                         mh["Accuracy"], mh["Sensitivity"], mh["FPR"])
                if split == "random":
                    for day in sorted(set(DAY_OF.values()) - {DAY_OF[attack]}):
                        b = _sample(df[(df.label == C.BENIGN_LABEL) & (df.day == day)], OTHER_DAY_BENIGN, C.SEED)
                        fp = int((pred["XGB"][1].predict(b[features(b)]) | pred["SVM"][1].predict(b[features(b)])).sum())
                        otherday_rows.append({"attack": attack, "algo": "XGB|SVM", "features": fs, "benign_day": day,
                                              "n": len(b), "FP": fp, "FPR": fp / len(b), "FPR_same_day": mh["FPR"]})

        # A13 (б): нормален трафик от два дни при обучението, тест върху третия ден
        for held in sorted(set(DAY_OF.values()) - {DAY_OF[attack]}):
            t2 = task_frame(df, attack, day_only=False)
            t2 = t2[(t2.y == 1) | (t2.day != held)]
            tr, te, _ = balanced_split(t2, C.SEED, False)
            b = _sample(df[(df.label == C.BENIGN_LABEL) & (df.day == held)], OTHER_DAY_BENIGN, C.SEED)
            pb = {}
            for algo in ALGOS:
                pipe, m, _ = fit_eval(algo, attack, tr, te, ())
                pb[algo] = pipe.predict(b[features(b)])
                lodo_rows.append({"attack": attack, "algo": algo, "held_out_day": held, "Accuracy": m["Accuracy"],
                                  "FPR_test": m["FPR"], "FP_held_out": int(pb[algo].sum()), "n_held_out": len(b),
                                  "FPR_held_out": float(pb[algo].mean())})
            yh = pb["XGB"] | pb["SVM"]
            lodo_rows.append({"attack": attack, "algo": "XGB|SVM", "held_out_day": held, "FP_held_out": int(yh.sum()),
                              "n_held_out": len(b), "FPR_held_out": float(yh.mean())})
            log.info("LODO %s %s: %s", attack, held, {k: round(float(v.mean()), 5) for k, v in pb.items()})

    # A12: пренос между подвидове
    for attack, (a, b) in SUBTYPES.items():
        for src, dst in ((a, b), (b, a)):
            t_src = task_frame(df, attack, labels=[src])
            t_dst = task_frame(df, attack, labels=[dst])
            tr, te, _ = balanced_split(t_src, C.SEED, False)
            dst_att = t_dst[t_dst.y == 1]
            for fs, drop in FEATURE_SETS.items():
                for algo in ALGOS:
                    feats = features(tr)
                    pipe, t_fit, _ = fit_final(algo, PARAMS[f"{algo}_{attack}"]["best_params"], tr[feats], tr["y"], drop=drop)
                    same = float(pipe.predict(te[te.y == 1][feats]).mean())
                    det = float(pipe.predict(dst_att[feats]).mean())
                    subtype_rows.append({"attack": attack, "train_on": src, "test_on": dst, "algo": algo, "features": fs,
                                         "n_train_attack": int(tr.y.sum()), "n_test_attack": len(dst_att),
                                         "detected_same_subtype": same, "detected_other_subtype": det})
                    log.info("подвид %s->%s %s %s: %.4f (собствен %.4f)", src, dst, algo, fs, det, same)

    out = C.RESULTS_DIR / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hybrid_main()).to_csv(out / "hybrid_main.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "split_features.csv", index=False)
    pd.DataFrame(subtype_rows).to_csv(out / "subtype_transfer.csv", index=False)
    pd.DataFrame(otherday_rows).to_csv(out / "other_day_benign.csv", index=False)
    pd.DataFrame(lodo_rows).to_csv(out / "held_out_day_benign.csv", index=False)
    b = df[df.label == C.BENIGN_LABEL]
    meta["benign_profile_by_day"] = {d: {"rows": int(len(g)),
                                         "init_win_65535_share": float((g["Init Fwd Win Byts"] == 65535).mean()),
                                         "tcp_share": float((g["Protocol"] == 6).mean())} for d, g in b.groupby("day")}
    meta["runtime_min"] = (time.perf_counter() - t00) / 60
    save_json(meta, out / "meta.json")
    log.info("готово за %.1f min", meta["runtime_min"])


if __name__ == "__main__" and "--hybrid-only" in sys.argv:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    (C.RESULTS_DIR / "robustness").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hybrid_main()).to_csv(C.RESULTS_DIR / "robustness" / "hybrid_main.csv", index=False)
elif __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    run()
