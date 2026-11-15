"""Изпълнява целия експеримент от край до край.

    python run_all.py                       # всички атаки и алгоритми + разширените анализи
    python run_all.py --no-extended         # само основните 4 модела и E1-E4
    python run_all.py --attacks PortScan --algos RF

Атаките и алгоритмите се избират от регистрите в src/registry.py.
Резултатите се записват в results/ (CSV, JSON, лог, фигури), моделите - в models/.
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

from src import config as C
from src.experiment import Experiment
from src.preprocess import save_json
from src.registry import ALGORITHMS, ATTACKS


def setup_logging():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        handlers=[logging.FileHandler(C.RESULTS_DIR / "run_log.txt", mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])


def environment_info() -> dict:
    import scipy
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ML-IDS експеримент върху CSE-CIC-IDS2018 и CIC-IDS2017")
    p.add_argument("--attacks", nargs="+", choices=list(ATTACKS), default=list(ATTACKS))
    p.add_argument("--algos", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS))
    p.add_argument("--no-extended", action="store_true", help="без разширените анализи A1-A14")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging()
    env = environment_info()
    save_json(env, C.RESULTS_DIR / "environment.json")
    logging.getLogger("run_all").info("Среда: %s", env)
    Experiment(args.attacks, args.algos, extended=not args.no_extended).run()
    if not args.no_extended and set(args.attacks) == set(ATTACKS) and set(args.algos) == set(ALGORITHMS):
        from src import robustness
        logging.getLogger("run_all").info("A10-A14: проверки за приложимост")
        robustness.run()


if __name__ == "__main__":
    main()
