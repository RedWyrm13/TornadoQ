import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Helper function to check if device is proper type
def resolve_device(device: str | None) -> torch.device:
    if device is None:
        raise ValueError("Device must be specified.")

    try:
        dev = torch.device(device)
    except Exception as e:
        raise ValueError(f"Invalid device string: {device}") from e

    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")

    return dev

# Helper functions for plotting, metrics, etc

def accuracy(outputs, targets):

    if outputs.ndim == 1 or outputs.shape[1] == 1:
        preds = (outputs.view(-1) > 0.5).float()
        targets = targets.view(-1)
        return (preds == targets).float().mean()

    else:
        preds = torch.argmax(outputs, dim=1)
        return (preds == targets).float().mean()



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