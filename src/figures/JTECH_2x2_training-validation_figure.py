#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot multi-model training/validation history for CNN chain aggregate models.

This script:
1. Searches the base directory for each JTech-* training directory.
2. Recursively finds:
      - REFIT*/.../plot_data/epoch_metrics.csv
      - <ModelName>_validation_constant_thresh/summary_metrics.csv
        inside each corresponding training directory
3. Extracts:
      - train_acc@0.5         from epoch_metrics.csv
      - train_loss            from epoch_metrics.csv
      - val_loss              from epoch_metrics.csv
      - validation accuracy   from summary_metrics.csv (constant threshold run)
4. Creates a single 2x2 summary figure across all CNN models:
      (a) Training accuracy vs epoch
      (b) Validation accuracy vs epoch
      (c) Training loss vs epoch
      (d) Validation loss vs epoch

Notes:
- Validation accuracy comes from the constant-threshold evaluation output
  (e.g., threshold = 0.667 in your workflow).
- Validation loss comes from epoch_metrics.csv and is threshold-independent.
- Training accuracy comes from train_acc@0.5 in epoch_metrics.csv.
"""

import os
import re
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# USER CONFIG
# ============================================================
BASE_DIR = "/path/to/nas_workspace/CNN/figures/30_epochs"
OUT_DIR = "/path/to/nas_workspace/CNN/figures/30_epochs/JTECH_Training-Validation_2x2"
OUT_NAME = "cnn_allmodels_training_validation_2x2.png"
DPI = 300

MODEL_ORDER = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG16", "VGG19"]

MODEL_LABELS = {
    "ResNet18": "ResNet-18",
    "ResNet34": "ResNet-34",
    "ResNet50": "ResNet-50",
    "ResNet101": "ResNet-101",
    "VGG16": "VGG-16",
    "VGG19": "VGG-19",
}

MODEL_MARKERS = {
    "ResNet18": "o",
    "ResNet34": "s",
    "ResNet50": "^",
    "ResNet101": "D",
    "VGG16": "P",
    "VGG19": "X",
}

# Fixed y-axis limits for consistency
ACC_YMIN, ACC_YMAX = 94.0, 100.0
LOSS_YMIN, LOSS_YMAX = 0.00, 0.30

# Plot appearance for overlapping curves
LINEWIDTH = 1.5
MARKERSIZE = 4
CURVE_ALPHA = 0.75

# Font / legend sizing
TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13
LEGEND_TITLE_FONTSIZE = 13
SUPTITLE_FONTSIZE = 17

# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "dejavuserif"

# ============================================================
# HELPERS
# ============================================================
def find_training_dirs(base_dir):
    dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("JTech-"):
            dirs.append(full)
    return sorted(dirs)


def infer_model_name(training_dir_name):
    m = re.match(r"JTech-([A-Za-z0-9]+)_training_", training_dir_name)
    if m:
        return m.group(1)
    return None


def find_epoch_metrics_csv(training_dir):
    pattern = os.path.join(training_dir, "REFIT*", "**", "plot_data", "epoch_metrics.csv")
    matches = sorted(set(glob.glob(pattern, recursive=True)))

    if not matches:
        return None

    if len(matches) > 1:
        print(f"Warning: multiple epoch_metrics.csv files found under {training_dir}")
        for m in matches:
            print(f"  {m}")
        print("Using the first match.")

    return matches[0]


def find_summary_metrics_csv(training_dir, model_name):
    """
    Prefer the top-level validation directory:
      <training_dir>/<ModelName>_validation_constant_thresh/summary_metrics.csv
    """
    preferred = os.path.join(
        training_dir,
        f"{model_name}_validation_constant_thresh",
        "summary_metrics.csv"
    )
    if os.path.isfile(preferred):
        return preferred

    patterns = [
        os.path.join(training_dir, "**", f"{model_name}_validation_constant_thresh", "summary_metrics.csv"),
        os.path.join(training_dir, "**", "summary_metrics.csv"),
    ]

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))

    filtered = []
    model_lower = model_name.lower()
    for m in matches:
        m_lower = m.lower()
        if model_lower in m_lower and "validation_constant_thresh" in m_lower:
            filtered.append(m)

    filtered = sorted(set(filtered))

    if not filtered:
        return None

    if len(filtered) > 1:
        print(f"Warning: multiple summary_metrics.csv files found for {model_name}")
        for m in filtered:
            print(f"  {m}")
        print("Using the first match.")

    return filtered[0]


def validate_columns(df, required_cols, path):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in file:\n  {path}\n"
            f"Missing: {missing}\n"
            f"Found: {list(df.columns)}"
        )

# ============================================================
# LOAD DATA
# ============================================================
def load_all_model_data(base_dir):
    model_data = {}

    training_dirs = find_training_dirs(base_dir)
    if not training_dirs:
        raise FileNotFoundError(f"No JTech-* training directories found in {base_dir}")

    for tdir in training_dirs:
        tname = os.path.basename(tdir)
        model_name = infer_model_name(tname)
        if model_name is None:
            print(f"Skipping unrecognized directory name: {tname}")
            continue

        epoch_csv = find_epoch_metrics_csv(tdir)
        summary_csv = find_summary_metrics_csv(tdir, model_name)

        if epoch_csv is None:
            print(f"Skipping {model_name}: no epoch_metrics.csv found.")
            continue

        if summary_csv is None:
            print(f"Skipping {model_name}: no summary_metrics.csv found inside:")
            print(f"  {tdir}")
            continue

        df_epoch = pd.read_csv(epoch_csv)
        df_sum = pd.read_csv(summary_csv)

        validate_columns(df_epoch, ["epoch", "train_acc@0.5", "train_loss", "val_loss"], epoch_csv)
        validate_columns(df_sum, ["epoch", "accuracy"], summary_csv)

        df = pd.merge(
            df_epoch[["epoch", "train_acc@0.5", "train_loss", "val_loss"]],
            df_sum[["epoch", "accuracy"]],
            on="epoch",
            how="inner"
        ).sort_values("epoch").reset_index(drop=True)

        if df.empty:
            print(f"Skipping {model_name}: merged dataframe is empty.")
            continue

        model_data[model_name] = {
            "training_dir": tdir,
            "epoch_csv": epoch_csv,
            "summary_csv": summary_csv,
            "df": df
        }

    return model_data

# ============================================================
# PLOTTING
# ============================================================
def make_plot(model_data, out_path):
    models_present = [m for m in MODEL_ORDER if m in model_data]
    extras = [m for m in model_data.keys() if m not in MODEL_ORDER]
    plot_models = models_present + sorted(extras)

    if not plot_models:
        raise RuntimeError("No usable model data found to plot.")

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, facecolor="white")

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    for model_name in plot_models:
        df = model_data[model_name]["df"]
        label = MODEL_LABELS.get(model_name, model_name)
        marker = MODEL_MARKERS.get(model_name, "o")

        epochs = df["epoch"].values
        train_acc = df["train_acc@0.5"].values * 100.0
        train_loss = df["train_loss"].values
        val_acc = df["accuracy"].values * 100.0
        val_loss = df["val_loss"].values

        ax1.plot(
            epochs, train_acc,
            marker=marker, linewidth=LINEWIDTH, markersize=MARKERSIZE,
            alpha=CURVE_ALPHA, label=label
        )
        ax2.plot(
            epochs, val_acc,
            marker=marker, linewidth=LINEWIDTH, markersize=MARKERSIZE,
            alpha=CURVE_ALPHA, label=label
        )
        ax3.plot(
            epochs, train_loss,
            marker=marker, linewidth=LINEWIDTH, markersize=MARKERSIZE,
            alpha=CURVE_ALPHA, label=label
        )
        ax4.plot(
            epochs, val_loss,
            marker=marker, linewidth=LINEWIDTH, markersize=MARKERSIZE,
            alpha=CURVE_ALPHA, label=label
        )

    ax1.set_title("Training Accuracy", fontsize=TITLE_FONTSIZE)
    ax2.set_title("Validation Accuracy", fontsize=TITLE_FONTSIZE)
    ax3.set_title("Training Loss", fontsize=TITLE_FONTSIZE)
    ax4.set_title("Validation Loss", fontsize=TITLE_FONTSIZE)

    ax1.set_ylabel("Percent [%]", fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel("Loss", fontsize=LABEL_FONTSIZE)
    ax3.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax4.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)

    ax1.set_ylim(ACC_YMIN, ACC_YMAX)
    ax2.set_ylim(ACC_YMIN, ACC_YMAX)
    ax3.set_ylim(LOSS_YMIN, LOSS_YMAX)
    ax4.set_ylim(LOSS_YMIN, LOSS_YMAX)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("white")
    
        # Extend axis slightly beyond last epoch
        ax.set_xlim(0, 31)
    
        # Major ticks
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    
        # Minor ticks only up to epoch 30
        minor_xticks = np.arange(2, 30, 1)
        minor_xticks = [x for x in minor_xticks if x not in [5, 10, 15, 20, 25]]
        ax.set_xticks(minor_xticks, minor=True)
    
        # Y minor ticks can still be automatic
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, length=6)
        ax.tick_params(axis="both", which="minor", length=3)
        ax.grid(True, which="major", alpha=0.35)

    ax1.legend(
        title="Model type:",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        loc="best",
        frameon=True
    )

    # fig.suptitle("CNN Training and Validation History Across Models",
    #              fontsize=SUPTITLE_FONTSIZE, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Scanning base directory:\n  {BASE_DIR}\n")
    model_data = load_all_model_data(BASE_DIR)

    if not model_data:
        raise RuntimeError("No model data found.")

    print("Models found:")
    for model_name in sorted(model_data.keys()):
        print(f"  {model_name}")
        print(f"    epoch_metrics:   {model_data[model_name]['epoch_csv']}")
        print(f"    summary_metrics: {model_data[model_name]['summary_csv']}")

    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    make_plot(model_data, out_path)

    print("\nDone.")
    print(f"Figure written to:\n  {out_path}")

if __name__ == "__main__":
    main()