import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, mean_absolute_error

class DroughtCNN(nn.Module):
    def __init__(self, k=3, num_classes=6):
        super(DroughtCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=49, out_channels=49, kernel_size=k, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(49, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1) 
        return self.fc(x)
    
def prepare_tensors(df, window_size=30):
    metadata_cols = ['fips', 'date', 'label', 'year']
    all_features = [c for c in df.columns if c not in metadata_cols]
    
    df = df.sort_values(['fips', 'date'])
    data_list, label_list, year_list = [], [], []
    
    for _, group in df.groupby('fips'):
        if len(group) < window_size:
            continue
        
        feat_vals = group[all_features].values[-window_size:].T 
        
        label = group['label'].iloc[-1]
        
        # Convert timestamp to year for expanding window evaluation
        year = pd.to_datetime(group['date'].iloc[-1], unit='ms').year
        
        data_list.append(feat_vals)
        label_list.append(label)
        year_list.append(year)

    return (torch.tensor(np.array(data_list), dtype=torch.float32), 
            torch.tensor(np.array(label_list), dtype=torch.long),
            torch.tensor(np.array(year_list)))

def evaluate_metrics(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            out = model(x.to(device))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(y.numpy())
    
    return f1_score(targets, preds, average='macro'), mean_absolute_error(targets, preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load Parquets
    print("Loading datasets...")
    train_df = pd.read_parquet("processed/train_flat.parquet")
    val_df = pd.read_parquet("processed/validation_flat.parquet")
    test_df = pd.read_parquet("processed/test_flat.parquet")

    # Add year column for windowing logic
    for df in [train_df, val_df, test_df]:
        df['year'] = pd.to_datetime(df['date']).dt.year

    # Combine train and val for the expansion pool
    full_val_pool = pd.concat([train_df, val_df])
    val_years = [2008, 2009, 2010, 2011]
    
    print("\n--- Starting Expanding Window Model Selection ---")
    for v_year in val_years:
        t_data = full_val_pool[full_val_pool['year'] < v_year]
        v_data = full_val_pool[full_val_pool['year'] == v_year]
        
        x_train, y_train, _ = prepare_tensors(t_data)
        x_val, y_val, _ = prepare_tensors(v_data)
        
        train_loader = DataLoader(TensorDataset(x_train, y_train, torch.zeros(len(y_train))), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val, torch.zeros(len(y_val))), batch_size=64)

        model = DroughtCNN(k=args.k).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Fold Training
        for epoch in range(args.epochs):
            model.train()
            for bx, by, _ in train_loader:
                optimizer.zero_grad()
                criterion(model(bx.to(device)), by.to(device)).backward()
                optimizer.step()
        
        f1, mae = evaluate_metrics(model, val_loader, device)
        print(f"Validation Year {v_year} | Macro-F1: {f1:.4f} | MAE: {mae:.4f}")

    # Final Test on 2012-2020
    print("\n--- Final Evaluation on Test Set (2012-2020) ---")
    x_test, y_test, y_years = prepare_tensors(test_df)
    test_loader = DataLoader(TensorDataset(x_test, y_test, y_years), batch_size=64)
    
    # Evaluate windows
    f1_total, mae_total = evaluate_metrics(model, test_loader, device)
    print(f"OVERALL TEST RESULTS | Macro-F1: {f1_total:.4f} | MAE: {mae_total:.4f}")

if __name__ == "__main__":
    main()