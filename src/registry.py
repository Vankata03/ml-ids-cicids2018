"""Регистри на изследваните атаки и алгоритми.

Експериментът е описан декларативно: всяка атака е AttackSpec (как се избират
нейните записи от набора), а всеки алгоритъм е AlgorithmSpec (как се създава
класификаторът, каква е решетката от хиперпараметри и колко данни се ползват).
Конвейерът, оценката, графиките и докладът обхождат регистрите, така че нова
атака или нов алгоритъм се добавят с един запис тук, без промяна на останалия код.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C


@dataclass(frozen=True)
class AttackSpec:
    key: str                          # кратко име (използва се в имената на файлове)
    title_bg: str                     # заглавие за графики и доклад
    category_bg: str                  # категория на атаката
    selector: Callable[[str], bool]   # кои оригинални етикети принадлежат на атаката
    description_bg: str = ""

    def __call__(self, label: str) -> bool:
        return self.selector(label)


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    name_bg: str
    family_bg: str
    factory: Callable[[], object]     # създава ненастроен класификатор
    param_grid: dict = field(default_factory=dict)
    cv_max: int = 50_000              # максимален брой записи за GridSearchCV
    train_max: int | None = None      # максимален брой записи за финалното обучение
    parallel_param: str | None = None  # параметър за паралелно обучение (n_jobs)
    threshold: float = 0.5            # праг на непрекъснатата оценка (вероятност 0,5; SVM: 0)
    # Сложност на конфигурацията (по-малко = по-прост, по-силно регуляризиран модел).
    # Използва се от правилото „една стандартна грешка“ при избора на хиперпараметри.
    complexity: Callable[[dict], tuple] = lambda p: ()

    def make(self):
        return self.factory()


ATTACKS: dict[str, AttackSpec] = {
    "DDoS": AttackSpec(
        key="DDoS",
        title_bg="Distributed Denial of Service (DDoS)",
        category_bg="DDoS (обемна атака)",
        selector=lambda lbl: lbl.startswith("DDOS attack-"),
        description_bg="DDoS с инструментите HOIC (HTTP наводнение) и LOIC-UDP от множество атакуващи "
                       "машини срещу уеб сървър в облачната инфраструктура (21.02.2018).",
    ),
    "BruteForce": AttackSpec(
        key="BruteForce",
        title_bg="Brute Force по речник (FTP и SSH)",
        category_bg="Brute Force (BruteForce)",
        selector=lambda lbl: lbl in ("FTP-BruteForce", "SSH-Bruteforce"),
        description_bg="Отгатване на пароли по речник с Patator срещу FTP и SSH сървъри (14.02.2018).",
    ),
    "Bot": AttackSpec(
        key="Bot",
        title_bg="Botnet (Ares)",
        category_bg="Botnet (Bot), команден сървър",
        selector=lambda lbl: lbl == "Bot",
        description_bg="Заразени машини, управлявани от команден сървър на Botnet Ares: "
                       "периодична връзка, изпълнение на команди, снимки на екрана (02.03.2018).",
    ),
}

ALGORITHMS: dict[str, AlgorithmSpec] = {
    "RF": AlgorithmSpec(
        key="RF",
        name_bg="Random Forest",
        family_bg="ансамбъл от решаващи дървета (bagging)",
        factory=lambda: RandomForestClassifier(random_state=C.SEED, n_jobs=1),
        # Регуляризирана решетка: по-големи листа и случайно подмножество от признаци
        # на всяко разклонение ограничават преобучението (train/test разлика).
        param_grid={
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [None, 25],
            "clf__min_samples_leaf": [2, 5, 10],
            "clf__max_features": ["sqrt", 0.5],
        },
        cv_max=C.RF_CV_MAX,
        train_max=None,
        parallel_param="clf__n_jobs",
        complexity=lambda p: (-p["clf__min_samples_leaf"],
                              p["clf__max_depth"] or 10**6,
                              {"sqrt": 0.0}.get(p["clf__max_features"], p["clf__max_features"]),
                              p["clf__n_estimators"]),
    ),
    "SVM": AlgorithmSpec(
        key="SVM",
        name_bg="SVM (RBF)",
        family_bg="метод на опорните вектори с RBF ядро",
        factory=lambda: SVC(kernel="rbf", random_state=C.SEED, cache_size=2000),
        param_grid={
            "clf__C": [1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1],
        },
        cv_max=C.SVM_CV_MAX,
        train_max=C.SVM_MAX_TRAIN,
        threshold=0.0,
        # gamma="scale" = 1 / (брой признаци · дисперсия) = 1 / брой признаци след мащабиране
        complexity=lambda p: (p["clf__C"],
                              1.0 / C.SELECT_K if p["clf__gamma"] == "scale" else p["clf__gamma"]),
    ),
    "XGB": AlgorithmSpec(
        key="XGB",
        name_bg="XGBoost",
        family_bg="градиентно усилване на решаващи дървета (boosting)",
        factory=lambda: XGBClassifier(tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                                      eval_metric="logloss", random_state=C.SEED, n_jobs=1),
        param_grid={
            "clf__n_estimators": [300, 800],
            "clf__max_depth": [6, 10],
            "clf__learning_rate": [0.03, 0.1],
            "clf__min_child_weight": [1, 5],
        },
        cv_max=C.RF_CV_MAX,
        train_max=None,
        parallel_param="clf__n_jobs",
        complexity=lambda p: (p["clf__max_depth"], -p["clf__min_child_weight"],
                              p["clf__n_estimators"] * p["clf__learning_rate"]),
    ),
}


def attack(key: str) -> AttackSpec:
    return ATTACKS[key]


def algorithm(key: str) -> AlgorithmSpec:
    return ALGORITHMS[key]
