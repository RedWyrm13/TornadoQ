from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


ScalerKind = Literal["minmax", "standard"]


@dataclass(frozen=True)
class DataPaths:
    train_path: str
    test_path: str


@dataclass(frozen=True)
class PreprocessConfig:
    scaler: ScalerKind = "minmax"
    feature_min: float = 0.0
    feature_max: float = 1.0
    impute_strategy: str = "median"


@dataclass(frozen=True)
class CVConfig:
    folds: int = 5
    shuffle: bool = True
    random_state: int = 42
    repeats: int = 1 # set >1 for RepeatedStratifiedKFold


# Task-specific meta
TARGET_BINARY = "ef_binary"
TARGET_MULTI = "ef_class"
DROP_COLS = (TARGET_BINARY, TARGET_MULTI)