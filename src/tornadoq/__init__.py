"""TornadoQ package."""
from .config import DataPaths, PreprocessConfig, CVConfig
from .io import load_excel_pair
from .preprocess import split_xy, build_preprocessor
from .benchmark_models import get_binary_estimators, get_multiclass_estimators
from .evaluate_classical_benchmarks import (
cv_evaluate_binary, cv_evaluate_multiclass, fit_and_eval_binary, fit_and_eval_multiclass,
)