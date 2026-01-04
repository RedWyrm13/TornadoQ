# tornadoq/io.py
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def _read_table(path: str) -> pd.DataFrame:
    """Read .xlsx/.xls or .csv into a DataFrame."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suf} for {path}")


def load_train_val_test(train_path: str, val_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from Excel/CSV files."""
    df_train = _read_table(train_path)
    df_val   = _read_table(val_path)
    df_test  = _read_table(test_path)

    logger.info("Loaded train: %s rows, %s cols", df_train.shape[0], df_train.shape[1])
    logger.info("Loaded val  : %s rows, %s cols", df_val.shape[0], df_val.shape[1])
    logger.info("Loaded test : %s rows, %s cols", df_test.shape[0], df_test.shape[1])
    return df_train, df_val, df_test
