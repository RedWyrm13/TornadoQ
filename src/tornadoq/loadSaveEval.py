import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from tornadoq.helper import resolve_device

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


def collect_outputs(model, loader, classifier="binary", device=None):
    """
    Runs model over loader and returns:
      all_targets: (N,)
      all_preds:   (N,)
      all_probs:   (N,) for binary OR (N, C) for multiclass
    """
    model.eval()
    device = resolve_device(device)

    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device).long()

            outputs = model(features)

            if classifier == "binary":
                logits = outputs.squeeze(-1)
                probs = torch.sigmoid(logits) if (logits.min() < 0 or logits.max() > 1) else logits
                preds = (probs > 0.5).long()
            else:
                probs = torch.softmax(outputs, dim=1)  # (B, C)
                preds = torch.argmax(probs, dim=1)

            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_preds   = torch.cat(all_preds).numpy()
    all_probs   = torch.cat(all_probs).numpy()

    return all_targets, all_preds, all_probs


def plot_confusion(cm, title, class_names=None, fontsize=12):
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(values_format="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_binary(y_true, y_score, title, fontsize=12):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return roc_auc


def plot_roc_multiclass_ovr(y_true, y_score, title, class_names=None, fontsize=12):
    """
    One-vs-Rest ROC curves + macro-average AUC.
    y_score: (N, C) probabilities
    """
    num_classes = y_score.shape[1]
    labels = np.arange(num_classes)

    y_true_1hot = label_binarize(y_true, classes=labels)

    # per-class AUCs
    class_aucs = []
    plt.figure(figsize=(7, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_1hot[:, i], y_score[:, i])
        a = auc(fpr, tpr)
        class_aucs.append(a)

        name = class_names[i] if class_names is not None else f"Class {i}"
        plt.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")

    # macro avg (simple average of per-class AUCs)
    macro_auc = float(np.mean(class_aucs))

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} — macro AUC={macro_auc:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return macro_auc, class_aucs


def eval_and_plot(model, loader, classifier="binary", title="EVAL", class_names=None, fontsize=12, device=None):
    safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(title))
    device = resolve_device(device)
    
    # global font sizing (optional)
    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize + 2,
        "axes.labelsize": fontsize + 1,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
    })

    y_true, y_pred, y_prob = collect_outputs(model, loader, classifier=classifier, device=device)

    # metrics + plots
    if classifier == "binary":
        cm = confusion_matrix(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        tp = cm[1, 1]; fn = cm[1, 0]; fp = cm[0, 1]
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0

        print(f"[{title}] AUC={auc_score:.4f}  F1={f1:.4f}  Acc={acc:.4f}  CSI={csi:.4f}")
        plt.figure()
        plot_confusion(cm, title=f"{title} — Confusion Matrix", class_names=["0", "1"], fontsize=fontsize)
        plt.savefig(f"confusion_matrix_{safe_title}.png")
        plt.close()
        
        plot_roc_binary(y_true, y_prob, title=f"{title} — ROC", fontsize=fontsize)
        plt.figure()
        plt.savefig(f"roc_binary{safe_title}.png")
        plt.close()
        return {"auc": auc_score, "f1": f1, "acc": acc, "csi": csi}

    else:
        num_classes = y_prob.shape[1]
        labels = np.arange(num_classes)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # AUC (OVR)
        y_true_1hot = label_binarize(y_true, classes=labels)
        auc_ovr = roc_auc_score(y_true_1hot, y_prob, multi_class="ovr")

        f1 = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)

        print(f"[{title}] AUC(OVR)={auc_ovr:.4f}  MacroF1={f1:.4f}  Acc={acc:.4f}")
        plt.figure()
        plot_confusion(cm, title=f"{title} — Confusion Matrix", class_names=class_names, fontsize=fontsize)
        plt.savefig(f"confusion_matrix_{safe_title}.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        macro_auc, per_class_aucs = plot_roc_multiclass_ovr(...)
        plt.savefig(f"roc_ovr_{safe_title}.png", dpi=300, bbox_inches="tight")
        plt.close()

        return {"auc_ovr": auc_ovr, "macro_f1": f1, "acc": acc, "macro_auc_simple": macro_auc, "auc_per_class": per_class_aucs}
