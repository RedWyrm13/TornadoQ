#!/usr/bin/env python
# coding: utf-8

def save(model, save_path=None)
    ## Save Model
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    if save_path == None:
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_path = f"models/final_model_{n_epochs}epochs_{timestamp}.pt"
    
    # Save model states and tracked data in a single file
    torch.save({
        "state_dict": model.state_dict(),
        "losses": losses,
    }, save_path)
    
    print(f"Model and statistics saved to {save_path}")


def load(model, load_path):
    load_path = load_path # Path to load model

    # Load the checkpoint
    checkpoint = torch.load(load_path)
    
    # Restore model weights
    DNN.load_state_dict(checkpoint["DNN_state_dict"])

    print(f"Loaded model from {load_path}")


def eval_model(model, test_loader)
    # --- Set model to evaluation mode ---
    model.eval()
    
    all_targets = []
    all_preds = []
    all_outputs = []
    
    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device)
            target = target.to(device).float().unsqueeze(-1)
    
            outputs = model(features)
            preds = (outputs > 0.5).float()
    
            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())
            all_outputs.append(outputs.cpu())

    # --- Concatenate all batches ---
    all_targets = torch.cat(all_targets).squeeze().long().numpy()  # integers 0/1
    all_preds = torch.cat(all_preds).squeeze().long().numpy()
    all_outputs = torch.cat(all_outputs).squeeze().numpy()             # floats in [0,1]

    # --- Sanity check ---
    print(all_targets.shape, all_preds.shape, all_outputs.shape)
    print(np.unique(all_targets))  # should be [0,1]

    # --- Compute Metrics ---
    cm = confusion_matrix(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_outputs)  # should work now
    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)

    # Critical Success Index (CSI)
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    csi = tp / (tp + fn + fp)

    print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, CSI: {csi:.4f}")

    # --- Plot Confusion Matrix ---
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # --- Plot ROC Curve ---
    fpr, tpr, thresholds = roc_curve(all_targets, all_outputs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='blue')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # --- Plot Metrics Bar Chart ---
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'CSI': csi}
    plt.figure(figsize=(6,4))
    plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green'])
    plt.ylim(0,1)
    plt.title('Classification Metrics')
    plt.show()