"""
1d_cnn.py
1D CNN deep learning model for drought classification.

Trains on drought data or loads a 1D CNN, predicts integer classes (0-5) for
the test data, then evaluates with accuracy and macro-F1.

Run with --sample N to test on N timeseries before the full run.
Run with --state_path "path" to load model state at `path`.

Usage:
    # train + eval on full dataset
    python 1d_cnn.py                   

    # quick test on 1000 timeseries
    python 1d_cnn.py --sample 1000     

    # load from `./model.pt` and eval
    python 1d_cnn.py --state_path "model.pt"
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from dataset import DroughtDataset, FEATURE_COLS
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, f1_score, classification_report

# Config
# TODO: adjust hyperparameters here + model for best results
MAX_EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_WORKERS = 0 # Seems to break > 0
OUT_DIR = Path("processed")

# Model
class CNNDroughtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(49, 32, 2) # 49 is len(FEATURE_COLS) but that crashes
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 6) 


    def forward(self, x):
        x = x.permute(0, 2, 1) # Lookback and n_features swapped
        x = self.pool(F.relu(self.conv1(x)))
        x = self.adaptive_pooling(x)
        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path=OUT_DIR / '1d_cnn_checkpoint.pt'):
        """
        Early stops training if validation loss doesn't improve after enough epochs
        to prevent overfitting and saves the best weights.
        
        Args:
            patience  : maximum epochs where validation loss fails to improve before
                        stopping training
            min_delta : minimum improvement of validation loss per epoch
            path      : location to save best weights dictionary to
                        
        """
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


# Model Training 
def train_model(device: torch.device, sample: int | None = None):
    train_ds = DroughtDataset("processed/train_flat.parquet")
    val_ds = DroughtDataset("processed/validation_flat.parquet")

    # Optional random subsets of datasets
    if sample is not None:
        train_ds = Subset(train_ds, torch.randperm(len(train_ds))[:sample])
        val_ds = Subset(val_ds, torch.randperm(len(val_ds))[:sample])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    

    model = CNNDroughtModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f'Training model for {MAX_EPOCHS} epochs. This may take a while.')
    early_stopping = EarlyStopping()
    for epoch in tqdm(range(MAX_EPOCHS)):
        # Training Portion
        # running_loss = 0.0
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item() * y_batch.size(0)
        # train_loss = running_loss / len(val_loader.dataset)

        # Validation for early stopping
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


# Main

def main(state_path : str | None = None, sample: int | None = None):
    OUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_ds = DroughtDataset("processed/test_flat.parquet")
    # Optional random subsets of datasets
    if sample is not None:
        print(f'  ▶ Running on sample of {sample} timeseries')
        test_ds = Subset(test_ds, torch.randperm(len(test_ds))[:sample])
    else:
        print(f'  ▶ Running on all timeseries')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    # Train model if no specified weights
    if state_path is None:
        print(f'  ▶ Training model from dataset')
        train_model(device, sample)
        state_path = OUT_DIR / '1d_cnn_checkpoint.pt'
    else:
        print(f'  ▶ Loading model from {state_path}')
    
    # Model and aggregate predictions
    all_preds = []
    all_true = []

    model = CNNDroughtModel()
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()

    print(f'Predicting {len(test_loader.dataset)} labels. This may take a while.')
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)

            # Classify from output vector
            logits = model(X_batch)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y_batch.tolist())
        
    # Overall Metrics
    overall_mae = mean_absolute_error(all_true, all_preds)
    overall_f1  = f1_score(all_true, all_preds, average="macro",
                            zero_division=0)

    print("\n" + "=" * 60)
    print("1D CNN — TEST RESULTS")
    print("=" * 60)
    print(f"  MAE             : {overall_mae:.4f}")
    print(f"  Macro F1        : {overall_f1:.4f}")
    print()
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
        help="Number of timeseries to sample for a quick test run"
    )
    parser.add_argument(
        "--state_path", type=str, default=None,
        help="Location of state dictionary to avoid retraining model"
    )
    args = parser.parse_args()
    main(state_path=args.state_path, sample=args.sample)
