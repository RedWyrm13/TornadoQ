from __future__ import annotations
import logging
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .config import DROP_COLS, PreprocessConfig


logger = logging.getLogger(__name__)

# wrapper for pandas.load_csv
def load_csv(filename):
    df = pd.read_csv(filename)
    return df


def split_xy(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target; drop known target columns from X.
    Avoids leakage by NOT touching the test data beyond column selection here.
    """
    X_tr = df_train.drop(list(DROP_COLS), axis=1, errors="ignore")
    X_te = df_test.drop(list(DROP_COLS), axis=1, errors="ignore")
    y_tr = df_train[target].copy()
    y_te = df_test[target].copy()
    return X_tr, X_te, y_tr, y_te




def _num_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]




def build_preprocessor(df: pd.DataFrame, cfg: PreprocessConfig) -> ColumnTransformer:
    """Create a ColumnTransformer to impute + scale numeric features only.
    Fit only on training data when used inside a Pipeline.
    """
    num_cols = _num_cols(df)
    if cfg.scaler == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=(cfg.feature_min, cfg.feature_max))


    num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy=cfg.impute_strategy)),
    ("scale", scaler),
    ])
    pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ], remainder="drop")
    logger.info("Preprocessor built with %d numeric columns", len(num_cols))
    return pre