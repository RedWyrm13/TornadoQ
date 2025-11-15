#!/usr/bin/env python
# coding: utf-8

# Helper functions for plotting, metrics, etc
from Imports import *

def accuracy(outputs, targets):
    preds = (outputs > 0.5).float()
    correct = (preds == targets).float().sum()
    return correct / targets.numel()

def plot_metrics(g_losses, d_losses):
    epochs = range(1, len(g_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, g_losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Losses Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, labels, title, vmin=0, vmax=10000):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar_kws={"label": "Count"})
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_percent(cm, labels, title):
    cm_percent = cm / cm.sum() * 200  # Normalize to percentages
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=100, cbar_kws={"label": "Percentage (%)"})
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.show()