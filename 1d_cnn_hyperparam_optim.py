"""
1d_cnn.py
1D CNN deep learning model for drought classification.

Usage:
    # 1. Optimize on a sample and save the best parameters
    python 1d_cnn.py --optimize --trials 10 --sample 2000

    # 2. Train on the WHOLE dataset using previously saved parameters
    python 1d_cnn.py --config_path "processed/best_params.json"

    # 3. Load final trained model weights and evaluate
    python 1d_cnn.py --state_path "processed/best_1d_cnn_model.pt" --config_path "processed/best_params.json"
"""

import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from dataset import DroughtDataset, FEATURE_COLS
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, classification_report
import optuna

OUT_DIR = Path("processed")
NUM_WORKERS = 0 
MAX_EPOCHS = 25


class CNNDroughtModel(nn.Module):
    def __init__(self, in_features=49, out_channels=64, fc1_units=64, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(out_channels, fc1_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc1_units, 6) 

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.adaptive_pooling(x)
        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path=OUT_DIR / '1d_cnn_checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


def objective(trial, device, train_ds, val_ds):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    out_channels = trial.suggest_int("out_channels", 32, 128, step=16)
    fc1_units = trial.suggest_int("fc1_units", 32, 128, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    sample_x, _ = train_ds[0]
    in_features = sample_x.shape[1]
    
    model = CNNDroughtModel(
        in_features=in_features, 
        out_channels=out_channels, 
        fc1_units=fc1_units, 
        dropout_rate=dropout_rate
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(y_batch.tolist())
                
        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_f1


def train_model(device: torch.device, sample: int | None = None, best_params: dict | None = None, save_path = None):
    train_ds = DroughtDataset("processed/train_flat.parquet")
    val_ds = DroughtDataset("processed/validation_flat.parquet")

    if sample is not None:
        train_ds = Subset(train_ds, torch.randperm(len(train_ds))[:sample])
        val_ds = Subset(val_ds, torch.randperm(len(val_ds))[:sample])

    sample_x, _ = train_ds[0] if isinstance(train_ds, DroughtDataset) else train_ds.dataset[0]
    in_features = sample_x.shape[1]

    batch_size = best_params.get("batch_size", 256) if best_params else 256
    lr = best_params.get("lr", 0.001) if best_params else 0.001
    out_channels = best_params.get("out_channels", 48) if best_params else 48
    fc1_units = best_params.get("fc1_units", 32) if best_params else 32
    dropout_rate = best_params.get("dropout_rate", 0.2) if best_params else 0.2

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    
    model = CNNDroughtModel(
        in_features=in_features, 
        out_channels=out_channels, 
        fc1_units=fc1_units, 
        dropout_rate=dropout_rate
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    save_path = save_path if save_path else OUT_DIR / '1d_cnn_checkpoint.pt'
    early_stopping = EarlyStopping(path=save_path)

    print(f'Training on {len(train_ds)} timeseries...')
    for epoch in tqdm(range(MAX_EPOCHS)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * y_batch.size(0)
        val_loss = running_loss / len(val_loader.dataset)

        early_stopping(val_loss, model)
        if early_stopping.stop_training:
            print(f'Early stopped at epoch {epoch + 1}.')
            break


def main(state_path: str | None = None, sample: int | None = None, 
         optimize: bool = False, trials: int = 10, config_path: str | None = None):
    
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_params = None

    if config_path is not None:
        print(f"▶ Loading model hyperparameters from: {config_path}")
        with open(config_path, 'r') as f:
            best_params = json.load(f)

    elif optimize:
        print("▶ Phase 1: Tuning hyperparameters on dataset sample...")
        train_ds = DroughtDataset("processed/train_flat.parquet")
        val_ds = DroughtDataset("processed/validation_flat.parquet")
        if sample is not None:
            train_ds = Subset(train_ds, torch.randperm(len(train_ds))[:sample])
            val_ds = Subset(val_ds, torch.randperm(len(val_ds))[:sample])
            
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, device, train_ds, val_ds), n_trials=trials)
        best_params = study.best_params
        
        save_config_path = OUT_DIR / "best_params.json"
        with open(save_config_path, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"★ Saved best hyperparameters to {save_config_path}")

    if state_path is None:
        print("\n▶ Phase 2: Training model on the WHOLE dataset...")
        final_model_path = OUT_DIR / 'best_1d_cnn_model.pt'
        train_model(device, sample=None, best_params=best_params, save_path=final_model_path)
        state_path = final_model_path
    else:
        print(f'▶ Loading existing model state weights from {state_path}')
    
    test_ds = DroughtDataset("processed/test_flat.parquet")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
    
    sample_x, _ = test_ds[0]
    in_features = sample_x.shape[1]
    out_channels = best_params.get("out_channels", 48) if best_params else 48
    fc1_units = best_params.get("fc1_units", 32) if best_params else 32
    
    model = CNNDroughtModel(in_features=in_features, out_channels=out_channels, fc1_units=fc1_units)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_true = [], []
    print(f'Predicting on test data...')
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y_batch.tolist())
        
    overall_acc = accuracy_score(all_true, all_preds)
    overall_mae = mean_absolute_error(all_true, all_preds)
    overall_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("1D CNN — TEST RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy : {overall_acc:.4f}")
    print(f"  MAE              : {overall_mae:.4f}")
    print(f"  Macro F1         : {overall_f1:.4f}\n")
    print(classification_report(
        all_true, all_preds,
        labels=[0, 1, 2, 3, 4, 5],
        target_names=["D-None","D0","D1","D2","D3","D4"],
        zero_division=0,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of timeseries to sample for optimization/testing"
    )
    parser.add_argument(
        "--state_path", type=str, default=None,
        help="Location of state weights (.pt file)"
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Path to a JSON file containing saved hyperparameters"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Trigger Optuna hyperparameter tuning before final evaluation"
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of optimization trials to run"
    )
    args = parser.parse_args()
    main(state_path=args.state_path, sample=args.sample, 
         optimize=args.optimize, trials=args.trials, config_path=args.config_path)