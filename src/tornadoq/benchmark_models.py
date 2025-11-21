from __future__ import annotations
import logging
from typing import Dict
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Helper: Builds a pipeline with (preprocessor → optional SMOTE → classifier)
# -------------------------------------------------------------------------
def _make_pipeline(preprocessor, clf, random_state, use_smote):
    steps = [("prep", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


# -------------------------------------------------------------------------
# BINARY CLASSIFIERS
# -------------------------------------------------------------------------
def get_binary_estimators(
    preprocessor,
    random_state: int = 42,
    use_smote: bool = True,
    models_to_test={"logreg", "rf", "gb", "xgb"},
) -> Dict[str, object]:
    """
    Returns a dictionary of binary classifiers wrapped inside pipelines.
    SMOTE is applied **inside** the pipeline so oversampling is done per CV fold.
    """

    # Registry of all available binary models (not filtered yet)
    registry = {
        "logreg": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gb": GradientBoostingClassifier(random_state=random_state),
    }

    # Add XGB if installed
    if XGBClassifier is not None:
        registry["xgb"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
        )

    # Build only the pipelines requested in models_to_test
    return {
        name: _make_pipeline(preprocessor, clf, random_state, use_smote)
        for name, clf in registry.items()
        if name in models_to_test
    }


# -------------------------------------------------------------------------
# MULTICLASS CLASSIFIERS
# -------------------------------------------------------------------------
def get_multiclass_estimators(
    preprocessor,
    random_state: int = 42,
    use_smote: bool = True,
    models_to_test={"logreg", "rf", "gb", "xgb"},
) -> Dict[str, object]:
    """
    Returns a dictionary of multiclass classifiers wrapped inside pipelines.
    """

    registry = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            multi_class="multinomial",
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gb": GradientBoostingClassifier(random_state=random_state),
    }

    if XGBClassifier is not None:
        registry["xgb"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="mlogloss",
        )

    return {
        name: _make_pipeline(preprocessor, clf, random_state, use_smote)
        for name, clf in registry.items()
        if name in models_to_test
    }
