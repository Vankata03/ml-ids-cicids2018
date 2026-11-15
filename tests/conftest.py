"""Общи фикстури: малка извадка от РЕАЛНИТЕ файлове на CSE-CIC-IDS2018 и CIC-IDS2017.

Тестовете не използват синтетични данни: от всеки оригинален дневен файл се копират
първите до 4 000 реда от всеки етикет, без промяна на стойностите. Ако data/raw или
data/raw_cicids2017 липсва, тестовете се пропускат.
"""
from pathlib import Path

import pandas as pd
import pytest

from src import config as C

RAW = C.RAW_DIR
PER_LABEL = 4_000


def _head_per_label(src: Path, dst: Path) -> None:
    """Копира първите до PER_LABEL реда от всеки етикет на src в dst."""
    taken, parts = {}, []
    for chunk in pd.read_csv(src, chunksize=200_000, low_memory=False, dtype=str):
        chunk.columns = [c.strip() for c in chunk.columns]
        for lbl, g in chunk.groupby("Label", sort=False):
            need = PER_LABEL - taken.get(lbl, 0)
            if need > 0 and lbl != "Label":
                parts.append(g.head(need))
                taken[lbl] = taken.get(lbl, 0) + min(need, len(g))
    pd.concat(parts).to_csv(dst, index=False)


@pytest.fixture(scope="session")
def real_subset(tmp_path_factory) -> tuple[Path, Path]:
    from src.preprocess import cic2017_files
    if not RAW.exists() or not any(RAW.rglob("*.csv")) or not cic2017_files():
        pytest.skip("Няма реални данни в data/raw и data/raw_cicids2017 (виж data/README.md)")
    root, root2017 = tmp_path_factory.mktemp("raw"), tmp_path_factory.mktemp("raw_cicids2017")
    for f in sorted(RAW.glob("*.csv")):
        _head_per_label(f, root / f.name)
    for f in cic2017_files():
        _head_per_label(f, root2017 / f.name)
    return root, root2017


@pytest.fixture(scope="session")
def real_df(real_subset):
    from src.preprocess import clean, load_raw
    df, stats = load_raw(*real_subset)
    df, _ = clean(df)
    return df, stats
