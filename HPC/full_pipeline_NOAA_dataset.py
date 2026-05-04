#!/usr/bin/env python
# coding: utf-8

# # TornadoQ Pipeline
# We use xgboost on the given datasets to classifiy tornadoes. We show the results for both the binary (weak and strong) classification and the classification into specific EF categories, called multiclass. We also have ignored the XGBoost portion of the pipeline. Currently we comapre a classical DNN (baseline) with 
# * Hybrid DNN - Classical DNN with the dataset enhanced via quantum feature engineering
# * QNN - Strongly Entangling Ansatz with original dataset
# * QNN - RandomLayers Ansatz with original dataset
# 

import torch
from torch.utils.data import DataLoader
from tornadoq.preprocessing import (
    ClassificationDataset,
    Preprocess,
    DataMaker,
)
from tornadoq.train import train
from tornadoq.models import InitializeModel
from tornadoq.loadSaveEval import eval_and_plot
import argparse

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
    "all_weight2": False,
    "save": True,
}
balance = 'smote'
df_train_ws, df_test_ws, df_val_ws = DataMaker(TRAIN_FILE, 
                                               TEST_FILE, 
                                               VAL_FILE, 
                                               withShadows = True, 
                                               output_filename="original_shadow_enhanced.csv", 
                                               shadow_options = shadow_options,
                                               debug = debug,
                                               percent_data=percent_data)
print("df_train_ws:", df_train_ws)
# Preprocessing
# with shadows (ws)
X_train_ws, y_train_ws, X_test_ws, y_test_ws, X_val_ws, y_val_ws = Preprocess(df_train_ws, 
                                                                              df_test_ws, 
                                                                              df_val_ws, 
                                                                              balance = balance, 
                                                                              classes = classifier,
                                                                              )
print("X_train_ws", X_train_ws)
# Without shadows
X_train_ns, y_train_ns, X_test_ns, y_test_ns, X_val_ns, y_val_ns = Preprocess(df_train_ns, 
                                                                              df_test_ns,
                                                                              df_val_ns,
                                                                              balance = balance,
                                                                              classes = classifier,
                                                                              )

# Prepare train, val and test data with shadows
train_data_ws = ClassificationDataset(X_train_ws, y_train_ws)
validation_data_ws = ClassificationDataset(X_val_ws, y_val_ws)
test_data_ws = ClassificationDataset(X_test_ws, y_test_ws)
print("train_data_ws", train_data_ws)

# Prepare train, val and test data without shadows
train_data_ns = ClassificationDataset(X_train_ns, y_train_ns)
validation_data_ns = ClassificationDataset(X_val_ns, y_val_ns)
test_data_ns = ClassificationDataset(X_test_ns, y_test_ns)

# Dataloaders with shadows
train_loader_ws = DataLoader(train_data_ws, batch_size=batch_size, shuffle=True, drop_last=True)
print(len(train_loader_ws))
val_loader_ws = DataLoader(validation_data_ws, batch_size=batch_size, shuffle=False, drop_last=True)
print(len(val_loader_ws))
test_loader_ws = DataLoader(test_data_ws, batch_size=batch_size, shuffle=False, drop_last=True)
print(len(test_loader_ws))
# Dataloaders without shadows
train_loader_ns = DataLoader(train_data_ns, batch_size=batch_size, shuffle=True, drop_last=True)
print(len(train_loader_ns))
val_loader_ns = DataLoader(validation_data_ns, batch_size=batch_size, shuffle=False, drop_last=True)
print(len(val_loader_ns))
test_loader_ns = DataLoader(test_data_ns, batch_size=batch_size, shuffle=False, drop_last=True)
print(len(test_loader_ns))
# Used for the input size of the model in the next cell
num_targets = 2 # ef_class and ef_binary are the targets
num_features_ws = (len(df_train_ws.columns)) - num_targets
num_features_ns = (len(df_train_ns.columns)) - num_targets

print("\nNumber of training samples:", len(train_data_ws))
print("Number of validation samples:", len(validation_data_ws))
print("Number of test samples:", len(test_data_ws))

print("\nNumber of training samples:", len(train_data_ns))
print("Number of validation samples:", len(validation_data_ns))
print("Number of test samples:", len(test_data_ns))


# Choose model and hyperparameters
model = "DNN"

n_epochs = 45
lr = .01

# Initialize, train, and evaluate model
model = InitializeModel(model, load_path = None, classifier = classifier, input_size = num_features_ws)
print(type(model))
train(model = model, 
      n_epochs = n_epochs, 
      lr = lr, 
      train_loader = train_loader_ws, 
      val_loader = val_loader_ws, 
      classifier = classifier, 
      device = device,
      percent_data=percent_data)

eval_and_plot(model, 
              test_loader_ws, 
              classifier=classifier, 
              title="EVAL With Shadows", 
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

eval_and_plot(model, 
              test_loader_ns, 
              classifier=classifier, 
              title="EVAL No Shadows", 
              class_names=None, 
              fontsize=fontsize, 
              device=device,
              percent_data = percent_data)



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

eval_and_plot(model, 
              test_loader_ns, 
              classifier=classifier, 
              title="EVAL", 
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
eval_and_plot(model, 
              test_loader_ns, 
              classifier=classifier, 
              title="EVAL", 
              class_names=None, 
              fontsize=fontsize, 
              device=device,
              percent_data = percent_data)



