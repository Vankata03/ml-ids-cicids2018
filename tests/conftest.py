"""Общи фикстури: малка извадка от РЕАЛНИТЕ файлове на CSE-CIC-IDS2018.

Тестовете не използват синтетични данни: от всеки оригинален дневен файл се копират
първите до 4 000 реда от всеки етикет, без промяна на стойностите. Ако data/raw липсва,
тестовете се пропускат.
"""
from pathlib import Path

import pandas as pd
import pytest

from src import config as C

RAW = C.RAW_DIR
PER_LABEL = 4_000


@pytest.fixture(scope="session")
def real_subset(tmp_path_factory) -> Path:
    if not RAW.exists() or not any(RAW.rglob("*.csv")):
        pytest.skip("Няма реални данни в data/raw (виж data/README.md)")
    root = tmp_path_factory.mktemp("raw")
    for f in sorted(RAW.glob("*.csv")):
        taken, parts = {}, []
        for chunk in pd.read_csv(f, chunksize=200_000, low_memory=False, dtype=str):
            chunk.columns = [c.strip() for c in chunk.columns]
            for lbl, g in chunk.groupby("Label", sort=False):
                need = PER_LABEL - taken.get(lbl, 0)
                if need > 0 and lbl != "Label":
                    parts.append(g.head(need))
                    taken[lbl] = taken.get(lbl, 0) + min(need, len(g))
        pd.concat(parts).to_csv(root / f.name, index=False)
    return root


@pytest.fixture(scope="session")
def real_df(real_subset):
    from src.preprocess import clean, load_raw
    df, stats = load_raw(real_subset)
    df, _ = clean(df)
    return df, stats
