import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from tornadoq.helper import accuracy, resolve_device
import os
def train(model, n_epochs, lr, train_loader, val_loader, classifier, device = None, percent_data = 1.0):

    device = resolve_device(device)
    model = model.to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler_G = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    
    # Empty array to track losses
    losses = []
    
    # Initialize metrics
    if classifier == "binary":
        loss_fn = nn.BCELoss()
    if classifier == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
        
    metric = accuracy
    best_val_metric = 0
    best_val_loss = 999
    
    # Initialize a dictionary to store epoch-wise results
    history = {
            'epoch': [],
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }

    # Train for n_epochs
    for epoch in range(n_epochs):
        model.train()
        train_loss, train_metric = 0.0, 0.0
    
        for features, target in train_loader:
            if classifier == "binary":
                features, target = features.to(device), target.unsqueeze(-1).to(device)
            if classifier == "multiclass":
                features, target = features.to(device), target.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            train_metric = train_metric + metric(outputs, target)

        train_loss = train_loss / len(train_loader)
        train_metric = train_metric / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_metric = 0.0, 0.0
        with torch.no_grad():
            for features, target in val_loader:
                if classifier == "binary":
                    features, target = features.to(device), target.unsqueeze(-1).to(device)
                if classifier == "multiclass":
                    features, target = features.to(device), target.long().to(device)
            
                outputs = model(features)
                val_loss = val_loss + loss_fn(outputs, target).item()
                val_metric = val_metric + metric(outputs, target)

        val_loss = val_loss / len(val_loader)
        val_metric = val_metric / len(val_loader)

        # Logging
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)

        # Report every 5th epoch
        if epoch % 5 == 4:
            print(f'Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | '
            f'Train Acc: {train_metric:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metric:.4f}')

        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            os.makedirs("models", exist_ok=True)
            save_path = "models/best_accuracy_epoch{epoch}_percent_data{percent_data}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "history": history,
            }, save_path)
            print(f'Epoch [{epoch+1}] had best val_metric')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            save_path = f"models/best_valLoss_epoch{epoch}percent_data{percent_data}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "history": history,
            }, save_path)
            print(f'Epoch [{epoch+1}] had best val_Loss')


def train_luqpi(model, n_epochs, lr, train_loader, val_loader,
                device=None, shadow_weight=0.3, percent_data=1.0):
    """
    LUQPI training loop.

    train_loader yields (x_orig, x_shadow, y) — only real samples, no SMOTE.
    Class imbalance is handled upstream by WeightedRandomSampler.
    val_loader yields (x_orig, y) — original features only, no shadows needed.

    The auxiliary shadow reconstruction loss forces the encoder to capture
    quantum-relevant structure from the original features.  The shadow head
    is never used after training.
    """
    device = resolve_device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)

    cls_loss_fn = nn.BCELoss()
    aux_loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for x_orig, x_shadow, target in train_loader:
            x_orig   = x_orig.to(device)
            x_shadow = x_shadow.to(device)
            target   = target.unsqueeze(-1).to(device)

            optimizer.zero_grad()
            pred, shadow_pred = model(x_orig, x_shadow)
            loss = cls_loss_fn(pred, target) + shadow_weight * aux_loss_fn(shadow_pred, x_shadow)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validate on original features only (LUQPI: no quantum at deployment)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, target in val_loader:
                features = features.to(device)
                target   = target.unsqueeze(-1).to(device)
                pred     = model(features)          # x_shadow=None → inference mode
                val_loss += cls_loss_fn(pred, target).item()
        val_loss /= len(val_loader)

        if epoch % 5 == 4:
            print(f'Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save({"state_dict": model.state_dict()},
                       f"models/luqpi_best_percent_data{percent_data}.pt")
            print(f'Epoch [{epoch+1}] had best val_loss')