#!/usr/bin/env python
# coding: utf-8

# # TornadoQ Pipeline
# We use xgboost on the given datasets to classifiy tornadoes. We show the results for both the binary (weak and strong) classification and the classification into specific EF categories, called multiclass. We also have ignored the XGBoost portion of the pipeline. Currently we comapre a classical DNN (baseline) with 
# * Hybrid DNN - Classical DNN with the dataset enhanced via quantum feature engineering
# * QNN - Strongly Entangling Ansatz with original dataset
# * QNN - RandomLayers Ansatz with original dataset
# 

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tornadoq.preprocessing import (
    ClassificationDataset,
    LUQPIDataset,
    Preprocess,
    DataMaker,
)
from tornadoq.train import train, train_luqpi
from tornadoq.models import InitializeModel, BinaryDNN_LUQPI
from tornadoq.loadSaveEval import eval_and_plot
import argparse
import json
import os

print("IMPORTS FINISHED!")
parser = argparse.ArgumentParser(description="Load dataset with partial percentage")
parser.add_argument("--percent_data", type = str, required=True, help = "Percentage of data to use in experiment")
args = parser.parse_args()
percent_data = float(args.percent_data)
classifier = 'binary' # binary or multiclass

# These can be in either .xlsx or .csv
TRAIN_FILE = "../Data/NOAA/train.csv"
VAL_FILE   = "../Data/NOAA/val.csv"
TEST_FILE  = "../Data/NOAA/test.csv"

# pytorch device 
# device = 'nps' if torch.mps.is_available() else 'cpu' # For Andrei's mac.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
# fontsize for graphs and plots
fontsize = 12

# Create pandas dataframes
# Without Shadows
debug = False
if debug:
    batch_size = 5


df_train_ns, df_test_ns, df_val_ns = DataMaker(TRAIN_FILE, 
                                               TEST_FILE, 
                                               VAL_FILE, 
                                               withShadows = False, 
                                               output_filename=None,
                                               debug=debug,
                                               percent_data=percent_data)

# With Shadows
shadow_options = {
    "pair_mode": "all",
    "ring_paulis": [],
    "T": 3000,
    "shots": 1000,
    "seed": 123,
    "all_weight2": True,
    "save": True,
}
df_train_ws, df_test_ws, df_val_ws = DataMaker(TRAIN_FILE,
                                               TEST_FILE,
                                               VAL_FILE,
                                               withShadows = True,
                                               output_filename="original_shadow_enhanced.csv",
                                               shadow_options = shadow_options,
                                               debug = debug,
                                               percent_data=percent_data)

# ── Feature counts ────────────────────────────────────────────────────────────
num_targets    = 2  # ef_class and ef_binary
num_features_ws = len(df_train_ws.columns) - num_targets
num_features_ns = len(df_train_ns.columns) - num_targets
n_shadow        = num_features_ws - num_features_ns
print(f"Original features: {num_features_ns}  Shadow features: {n_shadow}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
# No Shadows: SMOTE on the 8 original features (valid interpolation)
X_train_ns, y_train_ns, X_test_ns, y_test_ns, X_val_ns, y_val_ns = Preprocess(
    df_train_ns, df_test_ns, df_val_ns,
    balance='smote', classes=classifier,
)

# With Shadows (LUQPI): NO SMOTE — shadow features are quantum-derived and
# interpolating them creates physically meaningless synthetic values.
# Class imbalance is handled by WeightedRandomSampler below.
X_train_ws, y_train_ws, X_test_ws, y_test_ws, X_val_ws, y_val_ws = Preprocess(
    df_train_ws, df_test_ws, df_val_ws,
    balance=None, classes=classifier,
)

# ── No-Shadows DataLoaders ────────────────────────────────────────────────────
train_data_ns      = ClassificationDataset(X_train_ns, y_train_ns)
validation_data_ns = ClassificationDataset(X_val_ns,   y_val_ns)
test_data_ns       = ClassificationDataset(X_test_ns,  y_test_ns)

train_loader_ns = DataLoader(train_data_ns,      batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader_ns   = DataLoader(validation_data_ns, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_ns  = DataLoader(test_data_ns,       batch_size=batch_size, shuffle=False, drop_last=True)

# ── LUQPI DataLoaders ─────────────────────────────────────────────────────────
# Split ws preprocessed features into original and shadow parts.
# Preprocess preserves column order, so original features come first.
X_train_orig   = X_train_ws.iloc[:, :num_features_ns]
X_train_shadow = X_train_ws.iloc[:, num_features_ns:]
X_val_orig     = X_val_ws.iloc[:,   :num_features_ns]

# WeightedRandomSampler: oversample minority class to address imbalance
y_arr          = y_train_ws.values if hasattr(y_train_ws, 'values') else np.array(y_train_ws)
class_counts   = np.bincount(y_arr.astype(int))
sample_weights = (1.0 / class_counts)[y_arr.astype(int)]
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

luqpi_train_data = LUQPIDataset(X_train_orig, X_train_shadow, y_train_ws)
luqpi_val_data   = ClassificationDataset(X_val_orig, y_val_ws)

luqpi_train_loader = DataLoader(luqpi_train_data, batch_size=batch_size, sampler=sampler,  drop_last=True)
luqpi_val_loader   = DataLoader(luqpi_val_data,   batch_size=batch_size, shuffle=False,    drop_last=True)

print(f"\nLUQPI train batches : {len(luqpi_train_loader)}")
print(f"NS train batches    : {len(train_loader_ns)}")
print(f"NS test batches     : {len(test_loader_ns)}")


# ── LUQPI: DNN trained with shadow privileged information ─────────────────────
# The main encoder sees only the 8 original features at all times.
# Shadow features guide training via an auxiliary reconstruction head that
# forces the encoder to capture quantum-relevant structure; the head is
# discarded after training so deployment requires no quantum resources.
n_epochs = 45
lr = .01

luqpi_model = BinaryDNN_LUQPI(n_orig=num_features_ns, n_shadow=n_shadow)
print(type(luqpi_model))
train_luqpi(
    model        = luqpi_model,
    n_epochs     = n_epochs,
    lr           = lr,
    train_loader = luqpi_train_loader,
    val_loader   = luqpi_val_loader,
    device       = device,
    percent_data = percent_data,
)

# Evaluate on ORIGINAL features only — no quantum at deployment (true LUQPI)
metrics_ws = eval_and_plot(luqpi_model,
              test_loader_ns,
              classifier=classifier,
              title="EVAL With Shadows (LUQPI)",
              class_names=None,
              fontsize=fontsize,
              device=device,
              percent_data=percent_data)


# Choose model and hyperparameters
model = "DNN"

n_epochs = 45
lr = .01

# Initialize, train, and evaluate model
model = InitializeModel(model, load_path = None, classifier = classifier, input_size = num_features_ns)
print(type(model))
train(model = model, 
      n_epochs = n_epochs, 
      lr = lr,
      train_loader = train_loader_ns, 
      val_loader = val_loader_ns, 
      classifier = classifier, 
      device = device,
      percent_data=percent_data)

metrics_ns = eval_and_plot(model,
              test_loader_ns,
              classifier=classifier,
              title="EVAL No Shadows",
              class_names=None,
              fontsize=fontsize,
              device=device,
              percent_data=percent_data)



# Choose model and hyperparameters
# model = "DNN"
model = "StronglyEntangling"
# model = "RandomLayer"
n_epochs = 45
lr = .01
print(num_features_ns)

# Initialize, train, and evaluate model
model = InitializeModel(model, load_path = None, classifier = classifier, input_size = num_features_ns, device = device)
print(type(model))
train(model = model, 
      n_epochs = n_epochs, 
      lr = lr, 
      train_loader = train_loader_ns, 
      val_loader = val_loader_ns, 
      classifier = classifier, 
      device = device,
      percent_data=percent_data)

metrics_se = eval_and_plot(model,
              test_loader_ns,
              classifier=classifier,
              title="EVAL StronglyEntangling",
              class_names=None,
              fontsize=fontsize,
              device=device,
              percent_data=percent_data)


# Choose model and hyperparameters
# model = "DNN"
# model = "StronglyEntangling"
model = "RandomLayer"
n_epochs = 45
lr = .01

# Initialize, train, and evaluate model
model = InitializeModel(model, load_path = None, classifier = classifier, input_size = num_features_ns, device = device)
print(type(model))
train(model = model, 
      n_epochs = n_epochs, 
      lr = lr, 
      train_loader = train_loader_ns, 
      val_loader = val_loader_ns,
      classifier = classifier, 
      device = device,
      percent_data=percent_data
      )
metrics_rl = eval_and_plot(model,
              test_loader_ns,
              classifier=classifier,
              title="EVAL RandomLayer",
              class_names=None,
              fontsize=fontsize,
              device=device,
              percent_data=percent_data)


# ── AUC summary table ──────────────────────────────────────────────────────────
AUC_KEY = "auc" if classifier == "binary" else "auc_ovr"
RESULTS_JSON = "./results/auc_results.json"
RESULTS_TXT  = "./results/auc_table.txt"
os.makedirs("./results", exist_ok=True)

# Load existing results
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results[str(percent_data)] = {
    "no_shadows":         metrics_ns.get(AUC_KEY),
    "with_shadows":       metrics_ws.get(AUC_KEY),
    "random_layers":      metrics_rl.get(AUC_KEY),
    "strongly_entangling": metrics_se.get(AUC_KEY),
}

with open(RESULTS_JSON, "w") as f:
    json.dump(all_results, f, indent=2)

# Build pretty table
PERCENTS = [round(i * 0.1, 1) for i in range(1, 11)]
COLS = ["no_shadows", "with_shadows", "random_layers", "strongly_entangling"]
HEADERS = ["No Shadows", "With Shadows", "Random Layers", "Strongly Entangling"]
COL_W = 20

def _fmt(val):
    return f"{val:.4f}" if val is not None else "  N/A  "

header_row  = f"{'% Data':>8}  " + "  ".join(h.center(COL_W) for h in HEADERS)
divider     = "-" * len(header_row)
lines = [divider, header_row, divider]
for p in PERCENTS:
    row = all_results.get(str(p), {})
    vals = "  ".join(_fmt(row.get(c)).center(COL_W) for c in COLS)
    lines.append(f"{p:>8.1f}  {vals}")
lines.append(divider)

table = "\n".join(lines)
print("\n" + table)
with open(RESULTS_TXT, "w") as f:
    f.write("AUC Results by Model and Percent Data\n\n")
    f.write(table + "\n")
print(f"\nTable saved to {RESULTS_TXT}")

