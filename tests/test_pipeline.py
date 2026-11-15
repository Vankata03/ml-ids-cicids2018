"""Проверки на коректността на конвейера върху реални записи."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src import config as C
from src import stats as S
from src.evaluate import compute_metrics, operational_phase
from src.preprocess import FeatureFilter, build_task, natural_test_set
from src.registry import ALGORITHMS, ATTACKS
from src.train import fit_final, make_pipeline


def test_registry_selects_real_labels(real_df):
    df, _ = real_df
    labels = set(df[C.LABEL_COLUMN].unique())
    for spec in ATTACKS.values():
        assert spec.benign_label in labels, spec.key
        assert any(spec(l) for l in labels), spec.key
    for spec in ALGORITHMS.values():
        assert hasattr(spec.make(), "fit")
        assert all(k.startswith("clf__") for k in spec.param_grid)


def test_no_label_column_leaks_into_features(real_df):
    df, stats = real_df
    task = build_task(df, "PortScan", stats["original_class_counts"])
    assert C.LABEL_COLUMN not in task["X_train"].columns
    assert task["X_train"].shape[1] == 77


def test_benign_comes_from_the_dataset_of_the_attack(real_df):
    """Нормалният трафик на всяка задача е само от набора на атаката (без смесване на наборите)."""
    df, stats = real_df
    for attack, spec in ATTACKS.items():
        task = build_task(df, attack, stats["original_class_counts"])
        labels = pd.concat([task["labels_train"], task["labels_test"]])
        y = pd.concat([task["y_train"], task["y_test"]])
        assert set(labels[y == 0]) == {spec.benign_label}
        assert all(spec(l) for l in set(labels[y == 1]))
        assert task["stats"]["dataset"] == spec.dataset


def test_cic2017_columns_match_cse2018(real_df):
    """След преименуването записите на CIC-IDS2017 имат същите 77 признака без липсващи стойности."""
    df, stats = real_df
    assert stats["n_features_raw"] == 77
    assert not set(C.CIC2017_RENAME.values()) - set(df.columns)
    assert not df[df[C.LABEL_COLUMN] == "PortScan"].isna().any().any()


def test_train_test_disjoint_and_balanced(real_df):
    df, stats = real_df
    for attack in ATTACKS:
        task = build_task(df, attack, stats["original_class_counts"])
        tr, te = set(task["X_train"].index), set(task["X_test"].index)
        assert tr.isdisjoint(te)
        # и по съдържание: нито един вектор на признаците не е и в двете части
        rows = lambda X: set(map(tuple, X.to_numpy().tolist()))
        assert rows(task["X_train"]).isdisjoint(rows(task["X_test"]))
        y = pd.concat([task["y_train"], task["y_test"]])
        assert abs(y.mean() - 0.5) < 1e-9          # балансирано 1:1
        assert abs(len(te) / (len(tr) + len(te)) - C.TEST_SIZE) < 0.01
        X_nat, _ = natural_test_set(df, task, attack)
        if X_nat is not None:
            assert tr.isdisjoint(set(X_nat.index))  # E2 не използва обучаващи записи


def test_feature_filter_learns_only_from_train(real_df):
    df, stats = real_df
    task = build_task(df, "PortScan", stats["original_class_counts"])
    f = FeatureFilter().fit(task["X_train"])
    Z = task["X_train"][f.selected_]
    corr = Z.corr().abs().to_numpy().copy()
    np.fill_diagonal(corr, 0)
    assert np.nanmax(corr) <= C.CORR_THRESHOLD + 1e-9
    assert (Z.var() > C.VARIANCE_THRESHOLD).all()
    assert f.transform(task["X_test"]).shape[1] == len(f.selected_)


def test_selector_learns_only_from_train(real_df):
    from src.preprocess import ImportanceSelector
    df, stats = real_df
    task = build_task(df, "PortScan", stats["original_class_counts"])
    sel = ImportanceSelector(k=10).fit(task["X_train"], task["y_train"])
    assert len(sel.support_) == 10
    assert sel.transform(task["X_test"]).shape == (len(task["X_test"]), 10)


def test_pipeline_structure():
    for algo in ALGORITHMS:
        p = make_pipeline(algo)
        assert [s for s, _ in p.steps] == ["filter", "select", "scaler", "clf"]
        assert [s for s, _ in make_pipeline(algo, method="pca").steps] == ["filter", "scaler", "pca", "clf"]
        assert [s for s, _ in make_pipeline(algo, reduced=False).steps] == ["scaler", "clf"]


def test_metrics_match_confusion_matrix(real_df):
    df, stats = real_df
    task = build_task(df, "PortScan", stats["original_class_counts"])
    pipe, _, _ = fit_final("RF", {"clf__n_estimators": 50}, task["X_train"], task["y_train"])
    m, y_pred, sc = operational_phase(pipe, task["X_test"], task["y_test"])
    tn, fp, fn, tp = confusion_matrix(task["y_test"], y_pred).ravel()
    assert (m["TP"], m["TN"], m["FP"], m["FN"]) == (tp, tn, fp, fn)
    assert np.isclose(m["Precision"], tp / (tp + fp))
    assert np.isclose(m["Accuracy"], (tp + tn) / (tp + tn + fp + fn))
    assert np.isclose(m["Sensitivity"], tp / (tp + fn))
    assert np.isclose(m["F1"], 2 * tp / (2 * tp + fp + fn))

    ci = S.bootstrap_ci(task["y_test"], y_pred, sc, rounds=200)
    for k in ("Precision", "Accuracy", "Sensitivity", "F1"):
        assert ci[k]["ci_low"] <= ci[k]["value"] <= ci[k]["ci_high"]

    other, _, _ = fit_final("SVM", {"clf__C": 10}, task["X_train"], task["y_train"])
    p2 = other.predict(task["X_test"])
    r = S.mcnemar(task["y_test"], y_pred, p2)
    assert r["both_correct"] + r["only_A_correct"] + r["only_B_correct"] + r["both_wrong"] == len(p2)
    assert 0.0 <= r["p_value"] <= 1.0
    same = S.mcnemar(task["y_test"], y_pred, y_pred)
    assert same["p_value"] == 1.0


def test_recall_points_guarantee_target(real_df):
    """Прагът за целеви Recall наистина го постига (оценка: реален признак Flow Duration)."""
    from src.analysis import recall_points
    df, stats = real_df
    task = build_task(df, "PortScan", stats["original_class_counts"])
    y = task["y_test"].to_numpy()
    s = task["X_test"]["Flow Duration"].to_numpy()
    pts = recall_points(y, s, "test", targets=[0.5, 0.9])
    for _, r in pts.iterrows():
        assert r.recall >= r.target_recall - 1e-12
        pred = s >= r.threshold
        assert r.TP == int((pred & (y == 1)).sum())
        assert abs(r.precision - r.TP / max(pred.sum(), 1)) < 1e-12
