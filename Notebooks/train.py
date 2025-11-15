import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

def train(model, n_epochs, lr, train_loader, val_loader)

    model = model.to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler_G = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    
    # Empty array to track losses
    losses = []
    
    # Initialize metrics
    bce = nn.BCELoss()
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
            features, target = features.to(device), target.unsqueeze(-1).to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = bce(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_metric += metric(outputs, target)

        train_loss /= len(train_loader)
        train_metric /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_metric = 0.0, 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.unsqueeze(-1).to(device)
                outputs = model(X_val)
                val_loss += bce(outputs, y_val).item()
                val_metric += metric(outputs, y_val)

        val_loss /= len(val_loader)
        val_metric /= len(val_loader)

        # Logging
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)

        # Report every 10th epoch
        if epoch % 10 == 9:
            print(f'Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | '
            f'Train Acc: {train_metric:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metric:.4f}')

        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            os.makedirs("models", exist_ok=True)
            save_path = "models/best_accuracy_epoch{epoch}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "history": history,
            }, save_path)
            print(f'Epoch [{epoch+1}] had best val_metric')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            save_path = f"models/best_valLoss_epoch{epoch}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "history": history,
            }, save_path)
            print(f'Epoch [{epoch+1}] had best val_Loss')