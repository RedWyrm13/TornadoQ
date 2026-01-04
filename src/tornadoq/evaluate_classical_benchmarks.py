from __future__ import annotations
from typing import Dict

import numpy as np
from .config import CVConfig
import pandas as pd
import logging
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

logger = logging.getLogger(__name__)




def _make_cv(cfg: CVConfig):
    if cfg.repeats > 1:
        return RepeatedStratifiedKFold(n_splits=cfg.folds, n_repeats=cfg.repeats, random_state=cfg.random_state)
    return StratifiedKFold(n_splits=cfg.folds, shuffle=cfg.shuffle, random_state=cfg.random_state)




def cv_evaluate_binary(models: Dict[str, object], X, y, cfg: CVConfig) -> pd.DataFrame:
    cv = _make_cv(cfg)
    rows = []
    for name, pipe in models.items():
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc_mean": auc.mean(), "cv_roc_auc_std": auc.std()})
    df = pd.DataFrame(rows).set_index("model").sort_values("cv_roc_auc_mean", ascending=False)
    logger.info("Binary CV done:%s", df)
    return df




def cv_evaluate_multiclass(models: Dict[str, object], X, y, cfg: CVConfig) -> pd.DataFrame:
    cv = _make_cv(cfg)
    rows = []
    for name, pipe in models.items():
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append({"model": name, "cv_acc_mean": acc.mean(), "cv_acc_std": acc.std()})
    df = pd.DataFrame(rows).set_index("model").sort_values("cv_acc_mean", ascending=False)
    logger.info("Multiclass CV done:%s", df)
    return df




def fit_and_eval_binary(pipe, X_train, y_train, X_test, y_test) -> dict:
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
    y_pred = (y_prob >= 0.5).astype(int) if y_prob.ndim == 1 else pipe.predict(X_test)
    return {
        "test_roc_auc": roc_auc_score(y_test, y_prob),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["Weak (EF0-1)", "Strong (EF2+)"])
    }




def fit_and_eval_binary_many(models: dict, X_train, y_train, X_test, y_test) -> dict:
    return {name: fit_and_eval_binary(pipe, X_train, y_train, X_test, y_test)
            for name, pipe in models.items()}
