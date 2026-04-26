import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, mean_absolute_error, classification_report

from dataset import DroughtDataset, WEATHER_COLS, LOOKBACK


class DroughtLSTM(nn.Module):
    def __init__(self, n_temporal, n_static, hidden_size=128, num_classes=6):
        super(DroughtLSTM, self).__init__()
        self.n_temporal = n_temporal
        self.lstm = nn.LSTM(input_size=n_temporal, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + n_static, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        temporal = x[:, :, :self.n_temporal]
        static = x[:, -1, self.n_temporal:]
        _, (h_n, _) = self.lstm(temporal)
        combined = torch.cat([h_n[-1], static], dim=1)
        return self.fc(combined)


def evaluate_metrics(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(y.numpy())
    return f1_score(targets, preds, average='macro'), mean_absolute_error(targets, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    print("Loading datasets...")
    train_ds = DroughtDataset("Processed/train_flat.parquet", lookback=LOOKBACK, stride=1)
    val_ds = DroughtDataset("Processed/validation_flat.parquet", lookback=LOOKBACK, stride=1)

    n_features = train_ds._features.shape[1]
    n_static = n_features - len(WEATHER_COLS)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = DroughtLSTM(n_temporal=len(WEATHER_COLS), n_static=n_static).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("Results", exist_ok=True)
    results_path = "Results/lstm_results.txt"

    # Training
    print("\n--- Training LSTM ---")
    best_f1 = 0.0
    epoch_lines = []
    for epoch in range(args.epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            criterion(model(bx.to(device)), by.to(device)).backward()
            optimizer.step()

        f1, mae = evaluate_metrics(model, val_loader, device)
        line = f"Epoch {epoch+1}/{args.epochs} | Macro-F1: {f1:.4f} | MAE: {mae:.4f}"
        print(line)
        epoch_lines.append(line)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "Processed/lstm_best.pt")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load("Processed/lstm_best.pt", map_location=device, weights_only=True))
    f1, mae = evaluate_metrics(model, val_loader, device)
    summary = f"Best Macro-F1: {f1:.4f} | MAE: {mae:.4f}"
    print(summary)

    with open(results_path, "w") as rf:
        rf.write("--- Training LSTM ---\n")
        rf.write("\n".join(epoch_lines))
        rf.write("\n\n--- Final Evaluation ---\n")
        rf.write(summary + "\n")
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
