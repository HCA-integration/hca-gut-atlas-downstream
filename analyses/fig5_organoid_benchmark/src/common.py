from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = Path(__file__).resolve().parents[1]
DATA = ANALYSIS / "data"
OUT = ANALYSIS / "out"
LOGS = ANALYSIS / "logs"
CONFIG_PATH = ANALYSIS / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    for key, value in config["inputs"].items():
        expanded = os.path.expandvars(str(value))
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        config["inputs"][key] = path
    return config


def require_files(config: dict) -> None:
    missing = [
        f"{key}: {path}"
        for key, path in config["inputs"].items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n  " + "\n  ".join(missing)
        )


def setup_logging(name: str) -> logging.Logger:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS / f"{name}.log", mode="w"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def normalize_missing(series: pd.Series) -> pd.Series:
    values = series.astype(object)
    text = values.astype(str).str.strip()
    missing = values.isna() | text.str.lower().isin(
        {"", "nan", "none", "<na>", "unknown"}
    )
    return text.mask(missing)


def single_value(series: pd.Series, sample: str, column: str):
    values = normalize_missing(series).dropna().unique()
    if len(values) > 1:
        raise ValueError(
            f"Sample {sample!r} has multiple values for {column!r}: "
            f"{sorted(map(str, values))}"
        )
    return values[0] if len(values) else np.nan


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def broad_region(series: pd.Series) -> pd.Series:
    text = normalize_missing(series).str.lower()
    result = pd.Series(index=series.index, dtype=object)
    result[text.str.contains("duoden", na=False)] = "Duodenum"
    result[text.str.contains("jejun", na=False)] = "Jejunum"
    result[text.str.contains("ile", na=False)] = "Ileum"
    result[text.str.contains("colon|rect|cecum", regex=True, na=False)] = "Colon"
    result[text.str.contains("intestine", na=False) & result.isna()] = "Intestine"
    return result


def time_class(series: pd.Series) -> pd.Series:
    text = normalize_missing(series).str.lower()
    result = pd.Series(index=series.index, dtype=object)
    result[text.eq("early")] = "Early"
    result[text.eq("late")] = "Late"
    days = pd.to_numeric(text.str.extract(r"(\d+)")[0], errors="coerce")
    result[days.notna() & (days <= 14)] = "≤14 d"
    result[days.notna() & (days > 14) & (days < 56)] = "15–55 d"
    result[days.notna() & (days >= 56)] = "≥56 d"
    return result


def ensure_session_info() -> None:
    import importlib.metadata
    import platform

    packages = [
        "anndata",
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "scipy",
        "seaborn",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    write_json(
        DATA / "session_info.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
        },
    )
