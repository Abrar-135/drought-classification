"""
dataset.py
PyTorch Dataset for drought classification.

Loads the preprocessed flat parquet files (from preprocessing.py) and
generates sliding-window sequences on-the-fly, avoiding the need to
materialise hundreds of GB of sequence arrays upfront.

Usage:
    from dataset import DroughtDataset, FEATURE_COLS
    from torch.utils.data import DataLoader

    train_ds = DroughtDataset("processed/train_flat.parquet", lookback=180)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)

    for X_batch, y_batch in train_loader:
        # X_batch : (batch, lookback, n_features)  float32
        # y_batch : (batch,)                        int64
        ...
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOOKBACK = 180  # days of history per window

WEATHER_COLS = [
    "PRECTOT", "PS", "QV2M",
    "T2M", "T2MDEW", "T2MWET", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "TS",
    "WS10M", "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE",
    "WS50M", "WS50M_MAX", "WS50M_MIN", "WS50M_RANGE",
]

# Columns to use as model input features (all except fips, date, label)
# Populated once by DroughtDataset.__init__ from the parquet schema.
FEATURE_COLS: list[str] = []


class DroughtDataset(Dataset):
    """
    Lazily indexes all valid (county, start_day) windows on construction,
    then fetches individual windows on __getitem__ without ever holding all
    sequences in memory simultaneously.

    Args:
        parquet_path  : path to one of the flat parquet files produced by
                        preprocessing.py
        lookback      : number of consecutive days fed as input to the model
        stride        : step between consecutive windows (default 1 = every day;
                        increase to reduce dataset size, e.g. stride=7 for weekly)
    """

    def __init__(self, parquet_path: str, lookback: int = LOOKBACK, stride: int = 1):
        super().__init__()
        self.lookback = lookback

        # Load entire flat file into memory (parquet is column-oriented so
        # this is much more efficient than per-row random access)
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(["fips", "date"]).reset_index(drop=True)

        # Determine feature columns from schema
        non_feat = {"fips", "date", "label"}
        global FEATURE_COLS
        FEATURE_COLS = [c for c in df.columns if c not in non_feat]

        labels = df["label"].values.astype("int64")
        features = df[FEATURE_COLS].values.astype("float32")
        fips_vals = df["fips"].values

        # Build index of (global_start_row, length) per county window
        # so __getitem__ can slice directly into the numpy arrays.
        self._features = features   # (total_rows, n_features)
        self._labels   = labels     # (total_rows,)
        self._index: list[tuple[int, int]] = []  # (start_idx, target_idx)

        # Iterate county groups using numpy for speed
        unique_fips, first_idx = np.unique(fips_vals, return_index=True)
        county_lengths = np.diff(np.append(first_idx, len(fips_vals)))

        for start, length in zip(first_idx, county_lengths):
            if length <= lookback:
                continue
            # Every valid window start within this county
            for offset in range(0, length - lookback, stride):
                self._index.append((start + offset, start + offset + lookback))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        start, end = self._index[idx]
        X = torch.from_numpy(self._features[start:end])   # (lookback, n_features)
        y = torch.tensor(self._labels[end - 1], dtype=torch.long)  # scalar label
        return X, y
