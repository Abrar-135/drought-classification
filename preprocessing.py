"""
preprocessing.py
Prepares the drought classification dataset for baseline, ARIMA, LSTM, and 1D-CNN models.

Pipeline:
  1. Load raw timeseries + static soil data
  2. Merge soil features onto timeseries by FIPS county code
  3. Forward-fill weekly drought scores to daily within each county
  4. Bin continuous 0-5 score into 6 integer classes (0-5)
  5. Fit StandardScaler on train features only; transform all splits
  6. Save flat (non-sequential) .parquet files  → baseline & ARIMA
  7. Build sliding-window sequences             → LSTM & 1D-CNN
  8. Save sequences as compressed .npz arrays

Inputs (read from ./Data/):
  train_timeseries.csv / validation_timeseries.csv / test_timeseries.csv
  soil_data.csv

Outputs (written to ./Processed/):
  scaler.pkl
  train_flat.parquet / validation_flat.parquet / test_flat.parquet
"""

import gc
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Data"
OUT_DIR  = ROOT_DIR / "Processed"

# Days of history fed into LSTM / 1D-CNN per prediction step
LOOKBACK = 180

WEATHER_COLS = [
    "PRECTOT", "PS", "QV2M",
    "T2M", "T2MDEW", "T2MWET", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "TS",
    "WS10M", "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE",
    "WS50M", "WS50M_MAX", "WS50M_MIN", "WS50M_RANGE",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_split(name: str) -> pd.DataFrame:
    """Load one timeseries CSV split and parse dates."""
    path = DATA_DIR / f"{name}_timeseries.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["fips", "date"]).reset_index(drop=True)
    return df


def merge_soil(df: pd.DataFrame, soil: pd.DataFrame) -> pd.DataFrame:
    """Left-join static soil features onto the timeseries on 'fips'."""
    return df.merge(soil, on="fips", how="left")


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Forward-fill weekly drought scores to daily within each county.
    2. Drop rows still NaN after ffill (start-of-series with no prior score).
    3. Bin the continuous 0-5 score into integer class labels 0-5.
    Operates in-place to avoid doubling memory usage.
    """
    # Forward-fill within each county's time series (in-place on score column)
    df["score"] = (
        df.groupby("fips")["score"]
          .transform(lambda s: s.ffill())
    )

    # Drop rows with no score at all (beginning of series before first report)
    df.dropna(subset=["score"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Bin to integer drought level: 0 (None) through 5 (D4 extreme)
    df["label"] = df["score"].clip(0, 5).astype(int).astype("int8")
    df.drop(columns=["score"], inplace=True)

    return df


def build_sequences(df: pd.DataFrame, feature_cols: list, lookback: int):
    """
    For each county build overlapping windows of shape (lookback, n_features).
    The label for each window is the drought class on the final day of the window.

    Returns:
        X : np.ndarray  shape (N, lookback, n_features)
        y : np.ndarray  shape (N,)  int8 labels
    """
    all_X, all_y = [], []

    for _, county in df.groupby("fips"):
        feats  = county[feature_cols].values.astype("float32")  # (T, F)
        labels = county["label"].values                          # (T,)
        T = len(feats)

        if T <= lookback:
            continue

        # Build all windows at once using stride tricks
        starts = np.arange(T - lookback)
        windows = np.stack([feats[s : s + lookback] for s in starts])  # (N, L, F)
        targets = labels[lookback:]                                      # (N,)

        all_X.append(windows)
        all_y.append(targets)

    if not all_X:
        return np.empty((0, lookback, len(feature_cols)), dtype="float32"), \
               np.empty(0, dtype="int8")

    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ── Soil data ──────────────────────────────────────────────────────────────
    print("Loading soil data …")
    soil = pd.read_csv(DATA_DIR / "soil_data.csv")
    soil_feature_cols = [c for c in soil.columns if c != "fips"]
    feature_cols = WEATHER_COLS + soil_feature_cols
    print(f"  {len(feature_cols)} total features "
          f"({len(WEATHER_COLS)} weather + {len(soil_feature_cols)} soil)")

    # ── Load, merge, and label all three splits ────────────────────────────────
    print("\nLoading and preparing splits …")
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        print(f"  {name} …", end=" ", flush=True)
        df = load_split(name)
        df = merge_soil(df, soil)
        # Cast weather + soil features to float32 to halve memory footprint
        df[feature_cols] = df[feature_cols].astype("float32")
        df = prepare_labels(df)
        splits[name] = df
        label_dist = df["label"].value_counts().sort_index().to_dict()
        print(f"{len(df):,} rows | {df['fips'].nunique()} counties | "
              f"class dist: {label_dist}")
        gc.collect()

    # ── Fit scaler on training set only ───────────────────────────────────────
    print("\nFitting StandardScaler on training features …")
    scaler = StandardScaler()
    scaler.fit(splits["train"][feature_cols])

    for name in splits:
        splits[name][feature_cols] = scaler.transform(
            splits[name][feature_cols]
        ).astype("float32")
        gc.collect()

    scaler_path = OUT_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {scaler_path}")

    # ── Save flat (non-sequential) files for baseline models & ARIMA ──────────
    print("\nSaving flat feature files …")
    for name, df in splits.items():
        out = OUT_DIR / f"{name}_flat.parquet"
        cols_to_save = ["fips", "date", "label"] + feature_cols
        df[cols_to_save].to_parquet(out, index=False)
        print(f"  {out.name}  shape: {df[cols_to_save].shape}")

    # ── Sequence building is handled lazily at training time ──────────────────
    # Pre-materialising all windows (19M rows × 3108 counties × 180 days)
    # would require hundreds of GB.  DroughtDataset (see dataset.py) builds
    # windows on-the-fly from the flat parquet files instead.
    print("\nSkipping in-memory sequence build — use DroughtDataset (dataset.py)")
    print("for lazy window generation during model training.")

    print("\nPreprocessing complete.")
    print(f"Flat files + scaler saved under {OUT_DIR}/")


if __name__ == "__main__":
    main()
