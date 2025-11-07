from __future__ import annotations
import logging
from typing import Dict
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


try:
    from xgboost import XGBClassifier # type: ignore
except Exception: # pragma: no cover
    XGBClassifier = None # type: ignore


logger = logging.getLogger(__name__)




def get_binary_estimators(preprocessor, random_state: int = 42, use_smote: bool = True, models_to_test = {"logreg", "rf", "gb", "xgb"}) -> Dict[str, object]:
    """Binary classifiers wrapped in pipelines. Optional SMOTE **inside** the pipeline
    so that CV resampling occurs only on training folds.
    """
    # Base steps with preprocessor and optional SMOTE
    base_steps = [("prep", preprocessor)]
    if use_smote:
        base_steps.append(("smote", SMOTE(random_state=random_state)))

    # Initialize models dictionary
    models: Dict[str, object] = {}

    # Add models to the dictionary only if they are in models_to_test
    if 'logreg' in models_to_test:
        models["logreg"] = ImbPipeline(base_steps + [("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))])

    if 'rf' in models_to_test:
        models["rf"] = ImbPipeline(base_steps + [("clf", RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, class_weight="balanced_subsample", random_state=random_state))])

    if 'gb' in models_to_test:
        models["gb"] = ImbPipeline(base_steps + [("clf", GradientBoostingClassifier(random_state=random_state))])

    if 'xgb' in models_to_test:
        models["xgb"] = ImbPipeline(base_steps + [("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, random_state=random_state, eval_metric="logloss"))])

    return models




def get_multiclass_estimators(preprocessor, random_state: int = 42, use_smote: bool = True) -> Dict[str, object]:
    base_steps = [("prep", preprocessor)]
    if use_smote:
        base_steps.append(("smote", SMOTE(random_state=random_state)))


    models: Dict[str, object] = {
    "logreg": ImbPipeline(base_steps + [("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state, multi_class="multinomial"))]),
    "rf": ImbPipeline(base_steps + [("clf", RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, class_weight="balanced_subsample", random_state=random_state))]),
    "gb": ImbPipeline(base_steps + [("clf", GradientBoostingClassifier(random_state=random_state))]),
    }
    if XGBClassifier is not None:
        models["xgb"] = ImbPipeline(base_steps + [("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, random_state=random_state, eval_metric="mlogloss"))])
    return models