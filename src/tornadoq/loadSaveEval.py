import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, auc
)


def save(model, save_path=None):
    ## Save Model
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    if save_path == None:
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_path = f"models/final_model_{timestamp}.pt"
    # Save model states and tracked data in a single file
    torch.save({
        "state_dict": model.state_dict(),
    }, save_path)
    
    print(f"Model and statistics saved to {save_path}")



def load(model, load_path):
    load_path = load_path # Path to load model

    # Load the checkpoint
    checkpoint = torch.load(load_path)
    
    # Restore model weights
    model.load_state_dict(checkpoint["state_dict"])

    print(f"Loaded model from {load_path}")
    


def eval_model(model, test_loader, classifier="binary", fontsize=12):
    # NEW: global font scaling
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize + 2,
        "axes.labelsize": fontsize + 1,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device)
            target = target.to(device).long()

            outputs = model(features)

            # ============================================================
            #                      BINARY CASE
            # ============================================================
            if classifier == "binary":
                logits = outputs.squeeze(-1)

                if logits.min() < 0 or logits.max() > 1:
                    probs = torch.sigmoid(logits)
                else:
                    probs = logits

                preds = (probs > 0.5).long()

            # ============================================================
            #                      MULTICLASS CASE
            # ============================================================
            else:
                probs = torch.softmax(outputs, dim=1)     # (batch, num_classes)
                preds = torch.argmax(probs, dim=1)

            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    # ============================================================
    #               CONCATENATE ACROSS ALL BATCHES
    # ============================================================
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    print("Shapes:", all_targets.shape, all_preds.shape, all_probs.shape)
    print("Unique targets:", np.unique(all_targets))


    # ============================================================
    #                      METRICS — BINARY
    # ============================================================
    if classifier == "binary":
        cm = confusion_matrix(all_targets, all_preds)
        auc_score = roc_auc_score(all_targets, all_probs)
        f1 = f1_score(all_targets, all_preds)
        acc = accuracy_score(all_targets, all_preds)

        tp = cm[1, 1]
        fn = cm[1, 0]
        fp = cm[0, 1]
        csi = tp / (tp + fn + fp)

        print(f"AUC: {auc_score:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, CSI: {csi:.4f}")

    # ============================================================
    #                    METRICS — MULTICLASS
    # ============================================================
    else:
        num_classes = all_probs.shape[1]
        labels = np.arange(num_classes)

        cm = confusion_matrix(all_targets, all_preds, labels=labels)

        # AUC: One-vs-Rest multi-class
        targets_1hot = label_binarize(all_targets, classes=labels)
        auc_score = roc_auc_score(targets_1hot, all_probs, multi_class='ovr')


        f1 = f1_score(all_targets, all_preds, average="macro")
        acc = accuracy_score(all_targets, all_preds)

        # CSI per class
        csi = []
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            denom = tp + fn + fp
            csi.append(tp / denom if denom > 0 else 0.0)

        print(f"Multiclass AUC: {auc_score:.4f}, Macro F1: {f1:.4f}, Accuracy: {acc:.4f}")
        for i in range(num_classes):
            print(f"Class {i} CSI: {csi[i]:.4f}")


    # ============================================================
    #                   CONFUSION MATRIX PLOT
    # ============================================================
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


    # ============================================================
    #                       ROC CURVE
    # ============================================================
    plt.figure(figsize=(7,6))

    if classifier == "binary":
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        plt.plot(fpr, tpr, label=f"AUC={auc_score:.4f}")

    else:
        num_classes = all_probs.shape[1]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve((all_targets == i).astype(int), all_probs[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC={class_auc:.2f})")

    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{classifier.capitalize()} ROC Curve")
    plt.legend()
    plt.show()


    # ============================================================
    #                     METRICS BAR CHART
    # ============================================================
    if classifier == "binary":
        metrics = {"Accuracy": acc, "F1": f1, "CSI": csi}
    else:
        metrics = {"Accuracy": acc, "Macro F1": f1, "AUC (OVR)": auc_score}

    plt.figure(figsize=(6,4))
    plt.bar(metrics.keys(), metrics.values(), color=['skyblue','orange','green'])
    plt.ylim(0,1)
    plt.title("Classification Metrics")
    plt.show()

    # Multiclass CSI bar chart
    if classifier == "multiclass":
        plt.figure(figsize=(6,4))
        plt.bar([f"Class {i}" for i in range(num_classes)], csi, color='purple')
        plt.ylim(0,1)
        plt.title("CSI per Class")
        plt.show()