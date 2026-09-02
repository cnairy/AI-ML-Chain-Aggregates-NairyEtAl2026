#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage ranking across ALL models & epochs:

Stage 1:
  - Collect every epoch row from every model's summary_metrics.csv
  - Rank ALL rows by F1 (descending)
  - Keep top N (default: 10)

Stage 2:
  - Re-rank those top N rows by precision (descending)
  - Tie-breakers: FP (asc), FN (asc), then calibration metrics (asc if present)

Reads from:
  /path/to/nas_workspace/CNN/figures/30_epochs/**/{MODEL}_validation_constant_thresh/summary_metrics.csv

Outputs (NEW folder, no overwriting):
  /path/to/nas_workspace/CNN/figures/30_epochs/global_epoch_ranking_YYYYMMDD/
      all_epochs_all_models.csv
      all_epochs_ranked_by_f1.csv
      topN_by_f1_then_ranked_by_precision.csv
      report.txt
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# USER SETTINGS
# =========================
BASE_DIR = "/path/to/nas_workspace/CNN/figures/30_epochs/"
VAL_SUFFIX = "_validation_constant_thresh"

TOPN = 20


# =========================
# HELPERS
# =========================
def find_validation_folders(base_dir: str):
    val_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            if d.endswith(VAL_SUFFIX):
                val_dirs.append(os.path.join(root, d))
    return sorted(val_dirs)

def parse_model_name_from_valdir(val_dir: str) -> str:
    b = os.path.basename(val_dir.rstrip("/"))
    if b.endswith(VAL_SUFFIX):
        return b[: -len(VAL_SUFFIX)]
    return b

def ensure_out_dir(base_dir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(base_dir, f"global_epoch_ranking_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c == "epoch":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_cols(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]

def stage1_rank_by_f1(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df = df.sort_values(by=["f1", "precision", "FP", "FN"],
                        ascending=[False, False, True, True],
                        na_position="last")
    df["rank_f1_global"] = np.arange(1, len(df) + 1)
    return df

def stage2_rank_by_precision(df_top: pd.DataFrame) -> pd.DataFrame:
    df = df_top.copy()

    # Optional calibration columns
    calib_cols = []
    for c in ["ece", "brier", "logloss"]:
        if c in df.columns:
            calib_cols.append(c)

    sort_cols = ["precision", "f1", "FP", "FN"] + calib_cols + ["model", "epoch"]
    ascending = [False, False, True, True] + [True]*len(calib_cols) + [True, True]

    df = df.sort_values(by=sort_cols, ascending=ascending, na_position="last")
    df["rank_precision_within_topN"] = np.arange(1, len(df) + 1)
    return df


# =========================
# MAIN
# =========================
def main():
    val_dirs = find_validation_folders(BASE_DIR)
    if not val_dirs:
        raise RuntimeError(f"No folders ending with '{VAL_SUFFIX}' found under: {BASE_DIR}")

    out_dir = ensure_out_dir(BASE_DIR)

    rows = []
    required_min = ["epoch", "TN", "FP", "FN", "TP", "precision", "f1"]

    for vdir in tqdm(val_dirs, desc="Scanning models"):
        model = parse_model_name_from_valdir(vdir)
        summary_csv = os.path.join(vdir, "summary_metrics.csv")
        if not os.path.exists(summary_csv):
            continue

        df = pd.read_csv(summary_csv)

        # Ensure required columns exist
        missing = [c for c in required_min if c not in df.columns]
        if missing:
            # Skip, but keep going
            continue

        df = coerce_numeric(df)

        df["model"] = model
        df["val_dir"] = vdir

        # Keep a consistent column set but don’t discard extra useful columns if present
        # (e.g., accuracy, roc_auc, logloss, brier, ece)
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid summary_metrics.csv files found with required columns.")

    df_all = pd.concat(rows, ignore_index=True)

    # Drop rows with missing epoch/f1/precision
    df_all = df_all.dropna(subset=["epoch", "f1", "precision"]).copy()
    df_all["epoch"] = df_all["epoch"].astype(int)

    # Save raw combined table
    all_epochs_csv = os.path.join(out_dir, "all_epochs_all_models.csv")
    df_all.to_csv(all_epochs_csv, index=False)

    # Stage 1: global rank by F1
    df_rank_f1 = stage1_rank_by_f1(df_all)
    ranked_by_f1_csv = os.path.join(out_dir, "all_epochs_ranked_by_f1.csv")
    df_rank_f1.to_csv(ranked_by_f1_csv, index=False)

    # Top N by F1
    df_topN = df_rank_f1.head(TOPN).copy()

    # Stage 2: re-rank top N by precision
    df_topN_ranked = stage2_rank_by_precision(df_topN)
    topN_csv = os.path.join(out_dir, "topN_by_f1_then_ranked_by_precision.csv")
    df_topN_ranked.to_csv(topN_csv, index=False)

    # Report
    report_path = os.path.join(out_dir, "report.txt")
    lines = []
    lines.append(f"BASE_DIR: {BASE_DIR}")
    lines.append(f"Run timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append("Two-stage global ranking:")
    lines.append(f"  Stage 1: rank ALL epochs across ALL models by F1 (desc), tie: precision (desc), FP (asc), FN (asc)")
    lines.append(f"  Stage 2: take top {TOPN} by F1, then rank by precision (desc), tie: F1 (desc), FP (asc), FN (asc), then calibration (asc if present)")
    lines.append("")
    lines.append(f"TOTAL CANDIDATES (all models/epochs): {len(df_rank_f1)}")
    lines.append("")
    lines.append(f"TOP {TOPN} by F1 (then sorted by precision):")
    lines.append("")

    show_cols = ["model", "epoch", "precision", "f1", "FP", "FN", "TP", "TN"]
    # add optional calib columns if present
    for c in ["ece", "brier", "logloss"]:
        if c in df_topN_ranked.columns:
            show_cols.append(c)

    df_show = df_topN_ranked[show_cols].copy()

    # Pretty print into report
    for i, r in df_show.iterrows():
        parts = [
            f"{int(r.name)+1:02d}) {r['model']} | epoch {int(r['epoch']):02d}",
            f"precision={float(r['precision']):.4f}",
            f"f1={float(r['f1']):.4f}",
            f"FP={int(r['FP'])}",
            f"FN={int(r['FN'])}",
            f"TP={int(r['TP'])}",
            f"TN={int(r['TN'])}",
        ]
        for c in ["ece", "brier", "logloss"]:
            if c in df_show.columns and pd.notna(r.get(c, np.nan)):
                parts.append(f"{c}={float(r[c]):.4f}")
        lines.append("  " + "  ".join(parts))

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("")
    print("DONE")
    print(f"Output folder:\n  {out_dir}")
    print("Key files:")
    print(f"  {all_epochs_csv}")
    print(f"  {ranked_by_f1_csv}")
    print(f"  {topN_csv}")
    print(f"  {report_path}")
    print("")
    print("This script does NOT overwrite any model outputs; it only reads and writes a new summary folder.")


if __name__ == "__main__":
    main()

