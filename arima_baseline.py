"""
arima_baseline.py
ARIMA baseline for drought classification.

Fits one ARIMA(p, d, q) model per county on the training label series,
forecasts forward over the validation period, rounds to the nearest integer
class (0-5), then evaluates with accuracy and macro-F1.

ARIMA order (7, 1, 1):
  p=7  — captures a week of auto-regressive drought persistence
  d=1  — one-differencing removes the long-run trend
  q=1  — one moving-average term for short-term shock correction

Reads flat parquet splits from ./Processed/ and writes per-county metrics to
./Results/arima_results.csv.
Run with --sample N to test on N counties before the full run.

Usage:
    python arima_baseline.py               # all 3108 counties
    python arima_baseline.py --sample 50   # quick test on 50 counties
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

warnings.filterwarnings("ignore")   # suppress ARIMA convergence noise

# ── Config ─────────────────────────────────────────────────────────────────────
ARIMA_ORDER   = (7, 1, 1)
ROOT_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "Processed"
RESULTS_DIR   = ROOT_DIR / "Results"
TRAIN_PARQUET = PROCESSED_DIR / "train_flat.parquet"
VAL_PARQUET   = PROCESSED_DIR / "validation_flat.parquet"
RESULTS_FILE  = RESULTS_DIR / "arima_results.csv"
N_JOBS        = -1          # use all CPU cores


# ── Per-county worker ─────────────────────────────────────────────────────────

def fit_and_forecast(fips: int,
                     train_series: np.ndarray,
                     n_forecast: int) -> tuple[int, np.ndarray] | None:
    """
    Fit ARIMA on train_series and return (fips, forecast_array).
    Returns None on failure (model falls back to caller's default).
    """
    try:
        model  = ARIMA(train_series, order=ARIMA_ORDER)
        result = model.fit(method_kwargs={"warn_convergence": False})
        raw = result.forecast(steps=n_forecast)
        # Clip and round to integer class 0-5
        preds  = np.clip(np.round(raw), 0, 5).astype(int)
        return fips, preds
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main(sample: int | None = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load only the columns we need (fips, date, label) ─────────────────────
    print("Loading label data …")
    train_df = pd.read_parquet(TRAIN_PARQUET,
                               columns=["fips", "date", "label"])
    val_df   = pd.read_parquet(VAL_PARQUET,
                               columns=["fips", "date", "label"])

    train_df = train_df.sort_values(["fips", "date"]).reset_index(drop=True)
    val_df   = val_df.sort_values(["fips", "date"]).reset_index(drop=True)

    # Optional subset for quick testing
    all_fips = train_df["fips"].unique()
    if sample is not None:
        rng      = np.random.default_rng(42)
        all_fips = rng.choice(all_fips, size=min(sample, len(all_fips)),
                              replace=False)
        print(f"  ▶ Running on sample of {len(all_fips)} counties")
    else:
        print(f"  ▶ Running on all {len(all_fips)} counties")

    # Pre-extract per-county arrays into dicts for fast lookup
    train_by_fips = {
        f: grp["label"].values.astype("float32")
        for f, grp in train_df.groupby("fips")
        if f in set(all_fips)
    }
    val_by_fips = {
        f: grp["label"].values.astype(int)
        for f, grp in val_df.groupby("fips")
        if f in set(all_fips)
    }

    # ── Determine forecast horizon (can differ slightly per county) ────────────
    # Use the most common validation series length as default fallback
    val_lengths = [len(v) for v in val_by_fips.values()]
    default_n   = int(np.median(val_lengths))

    print(f"\nFitting ARIMA{ARIMA_ORDER} on {len(all_fips)} counties "
          f"(val horizon ≈ {default_n} days) …")
    print("This may take several minutes — running in parallel …\n")

    # ── Parallel fit + forecast ────────────────────────────────────────────────
    tasks = [
        delayed(fit_and_forecast)(
            fips,
            train_by_fips[fips],
            len(val_by_fips.get(fips, np.array([]))) or default_n
        )
        for fips in all_fips
        if fips in train_by_fips and fips in val_by_fips
    ]

    raw_results = Parallel(n_jobs=N_JOBS, verbose=5)(tasks)

    # ── Aggregate predictions ──────────────────────────────────────────────────
    all_preds, all_true, result_rows = [], [], []
    n_failed = 0

    for res in raw_results:
        if res is None:
            n_failed += 1
            continue
        fips, preds = res
        truth = val_by_fips[fips]
        # Trim to shorter length in case of mismatch
        n = min(len(preds), len(truth))
        all_preds.extend(preds[:n].tolist())
        all_true.extend(truth[:n].tolist())

        # Per-county row for CSV
        county_acc = accuracy_score(truth[:n], preds[:n])
        county_f1  = f1_score(truth[:n], preds[:n], average="macro",
                              zero_division=0)
        result_rows.append({
            "fips":     fips,
            "accuracy": round(county_acc, 4),
            "macro_f1": round(county_f1, 4),
            "n_days":   n,
        })

    # ── Overall metrics ────────────────────────────────────────────────────────
    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    overall_acc = accuracy_score(all_true, all_preds)
    overall_f1  = f1_score(all_true, all_preds, average="macro",
                           zero_division=0)

    print("\n" + "=" * 60)
    print("ARIMA BASELINE — VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Counties fitted : {len(result_rows)}")
    print(f"  Counties failed : {n_failed}")
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Macro F1        : {overall_f1:.4f}")
    print()
    print(classification_report(
        all_true, all_preds,
        labels=[0, 1, 2, 3, 4, 5],
        target_names=["D-None","D0","D1","D2","D3","D4"],
        zero_division=0,
    ))

    # ── Save per-county results ────────────────────────────────────────────────
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Per-county results saved → {RESULTS_FILE}")
    print(f"\nOverall accuracy: {overall_acc:.4f}  |  Macro F1: {overall_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of counties to sample for a quick test run"
    )
    args = parser.parse_args()
    main(sample=args.sample)
