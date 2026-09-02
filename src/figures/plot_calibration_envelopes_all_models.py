#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-model calibration envelope plotting (no pandas).

What it does
------------
For each model under BASE_DIR (ResNet18/34/50/101, VGG16/19), this script:
  1) Finds:  {MODEL}_validation_constant_thresh/calibration/
  2) Reads:  calibration_epoch_XX.csv (skipping warmup epochs, default: < 7)
  3) Plots:  A calibration "envelope" (min/max across epochs excluding reference)
             with:
               - epoch_ref curve in black
               - envelope outer boundary as thin dark-blue lines
               - envelope fill semi-transparent
               - dashed perfect-calibration line
               - legend below plot (no vertical squish)
  4) Writes: one PNG + one CSV per model into a new output folder

You can change the reference calibration epoch PER MODEL by editing the
REFERENCE_EPOCH_BY_MODEL dict below.

This script does NOT overwrite any model outputs; it only reads and writes a new summary folder.
"""

import os
import re
import csv
import glob
import math
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# USER CONFIG (edit these)
# ----------------------------

# Set the reference epoch for each model here
REFERENCE_EPOCH_BY_MODEL = {
    "ResNet18": 26,
    "ResNet34": 18,
    "ResNet50": 23,
    "ResNet101": 26,
    "VGG16": 22,
    "VGG19": 23,
}

MIN_EPOCH = 7
MAX_EPOCH = 30

ENVELOPE_EDGE_COLOR = "#1f4e79"
ENVELOPE_EDGE_LW = 1.0
ENVELOPE_FILL_ALPHA = 0.18

GRID_N = 201


# ----------------------------
# Small helpers
# ----------------------------
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x):
    try:
        if x is None:
            return float("nan")
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def infer_model_name_from_dir(name):
    models = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG16", "VGG19"]
    low = name.lower()
    for m in models:
        if m.lower() in low:
            return m
    return None


def list_training_roots(base_dir):
    roots = []
    for nm in os.listdir(base_dir):
        p = os.path.join(base_dir, nm)
        if not os.path.isdir(p):
            continue
        m = infer_model_name_from_dir(nm)
        if m is None:
            continue
        roots.append((m, p))
    roots.sort(key=lambda x: x[0])
    return roots


def find_validation_folder(model_root, model_name):
    direct = os.path.join(model_root, f"{model_name}_validation_constant_thresh")
    if os.path.isdir(direct):
        return direct

    matches = []
    for p in glob.glob(os.path.join(model_root, "**", f"{model_name}_validation_constant_thresh"), recursive=True):
        if os.path.isdir(p):
            matches.append(p)
    matches = sorted(set(matches))
    return matches[0] if matches else None


def set_common_style():
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


# ----------------------------
# Calibration reading
# ----------------------------
def read_calibration_csv(path):
    """
    Supports header variants:
      - prob_pred,prob_true
      - mean_predicted_prob,observed_fraction_positive
    Returns x,y arrays sorted by x, clipped to [0,1].
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return None, None

    header = [h.strip().lower() for h in rows[0]]
    start_i = 0

    x_idx = None
    y_idx = None

    header_like = False
    if len(header) >= 2:
        h0 = header[0]
        if ("prob" in h0) or ("mean" in h0):
            header_like = True

    if header_like:
        start_i = 1
        for i, h in enumerate(header):
            if h in ("prob_pred", "mean_predicted_prob", "mean_predicted_probability", "mean_pred"):
                x_idx = i
            if h in ("prob_true", "observed_fraction_positive", "fraction_positive", "obs_frac_pos"):
                y_idx = i
        if x_idx is None:
            x_idx = 0
        if y_idx is None:
            y_idx = 1
    else:
        x_idx, y_idx = 0, 1

    xs, ys = [], []
    for r in rows[start_i:]:
        if len(r) <= max(x_idx, y_idx):
            continue
        x = safe_float(r[x_idx])
        y = safe_float(r[y_idx])
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        return None, None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # de-duplicate x
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    x_unique = np.clip(x_unique, 0.0, 1.0)
    y_unique = np.clip(y_unique, 0.0, 1.0)

    return x_unique, y_unique


def interp_curve_to_grid(x, y, grid):
    if x is None or y is None or len(x) < 2:
        return None
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def parse_epoch_from_filename(fp):
    m = re.search(r"calibration_epoch_(\d+)\.csv$", os.path.basename(fp))
    if not m:
        return None
    return int(m.group(1))


# ----------------------------
# Plotting: envelope per model
# ----------------------------
def plot_calibration_envelope_for_model(
    model_name,
    cal_dir,
    out_png,
    out_csv_grid_curves,
    epoch_ref,
    min_epoch=7,
    max_epoch=None,
    grid_n=201,
):
    if not os.path.isdir(cal_dir):
        return False, f"[{model_name}] Calibration directory not found: {cal_dir}"

    files = sorted(glob.glob(os.path.join(cal_dir, "calibration_epoch_*.csv")))
    if not files:
        return False, f"[{model_name}] No calibration_epoch_*.csv files in: {cal_dir}"

    grid = np.linspace(0.0, 1.0, grid_n)

    epoch_to_curve = {}
    for fp in files:
        ep = parse_epoch_from_filename(fp)
        if ep is None:
            continue
        if ep < min_epoch:
            continue
        if max_epoch is not None and ep > max_epoch:
            continue

        x, y = read_calibration_csv(fp)
        ygrid = interp_curve_to_grid(x, y, grid)
        if ygrid is None:
            continue
        epoch_to_curve[ep] = ygrid

    if epoch_ref < min_epoch:
        return False, f"[{model_name}] epoch_ref={epoch_ref} < min_epoch={min_epoch}"
    if max_epoch is not None and epoch_ref > max_epoch:
        return False, f"[{model_name}] epoch_ref={epoch_ref} > max_epoch={max_epoch}"
    if epoch_ref not in epoch_to_curve:
        return False, f"[{model_name}] Missing calibration for epoch {epoch_ref:02d} in {cal_dir}"

    other_epochs = [ep for ep in sorted(epoch_to_curve.keys()) if ep != epoch_ref]
    if len(other_epochs) < 1:
        return False, f"[{model_name}] Not enough non-reference epochs to build an envelope."

    y_ref = epoch_to_curve[epoch_ref]

    # Envelope from OTHER epochs
    stack_other = np.vstack([epoch_to_curve[ep] for ep in other_epochs])
    y_min_excl = np.nanmin(stack_other, axis=0)
    y_max_excl = np.nanmax(stack_other, axis=0)

    # Expand envelope to include reference wherever it is the outer bound
    y_min = np.minimum(y_min_excl, y_ref)
    y_max = np.maximum(y_max_excl, y_ref)

    # Save gridded curves + BOTH envelopes (excl-ref and inclusive)
    epochs_sorted = sorted(epoch_to_curve.keys())
    header = (
        ["prob"]
        + [f"epoch_{ep:02d}" for ep in epochs_sorted]
        + ["env_min_excl_ref", "env_max_excl_ref", "env_min_including_ref", "env_max_including_ref"]
    )
    rows = []
    for i in range(grid.size):
        row = [f"{grid[i]:.6f}"]
        for ep in epochs_sorted:
            row.append(f"{epoch_to_curve[ep][i]:.6f}")
        row.append(f"{y_min_excl[i]:.6f}")
        row.append(f"{y_max_excl[i]:.6f}")
        row.append(f"{y_min[i]:.6f}")
        row.append(f"{y_max[i]:.6f}")
        rows.append(row)
    save_csv(out_csv_grid_curves, header, rows)

    # Plot
    set_common_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=300)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="0.5", label="Perfect calibration")

    # Fill envelope (now guaranteed to contain the reference)
    ax.fill_between(
        grid, y_min, y_max,
        alpha=ENVELOPE_FILL_ALPHA,
        label=f"Envelope (epochs {min_epoch}–{max(epoch_to_curve.keys())}; includes ref where needed)"
    )

    # Boundary lines
    ax.plot(grid, y_min, color=ENVELOPE_EDGE_COLOR, linewidth=ENVELOPE_EDGE_LW, label=None)
    ax.plot(grid, y_max, color=ENVELOPE_EDGE_COLOR, linewidth=ENVELOPE_EDGE_LW, label=None)

    # Reference curve
    ax.plot(grid, y_ref, color="black", linewidth=2.2, label=f"Epoch {epoch_ref} (reference)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"{model_name} calibration envelope (epochs >= {min_epoch}; warmup skipped)")
    ax.grid(True, alpha=0.25)

    # Legend under plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=True
    )
    fig.subplots_adjust(bottom=0.25)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return True, f"[{model_name}] Saved: {out_png}"


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir", type=str, required=True,
        help="Base directory containing JTech-* training folders (e.g., /nas/.../CNN/figures/30_epochs/)."
    )
    ap.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory. Default: {base_dir}/calibration_envelopes_{timestamp}"
    )
    args = ap.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(base_dir, f"calibration_envelopes_{now_stamp()}")
    os.makedirs(out_dir, exist_ok=True)

    run_log = os.path.join(out_dir, "run_log.txt")
    with open(run_log, "w") as f:
        f.write(f"Run timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"BASE_DIR: {base_dir}\n")
        f.write(f"OUT_DIR:  {out_dir}\n")
        f.write(f"MIN_EPOCH: {MIN_EPOCH}\n")
        f.write(f"MAX_EPOCH: {MAX_EPOCH}\n")
        f.write(f"GRID_N:    {GRID_N}\n")
        f.write("REFERENCE_EPOCH_BY_MODEL:\n")
        for k in sorted(REFERENCE_EPOCH_BY_MODEL.keys()):
            f.write(f"  {k}: {REFERENCE_EPOCH_BY_MODEL[k]}\n")

    training_roots = list_training_roots(base_dir)
    if not training_roots:
        raise RuntimeError(f"No model training folders found under: {base_dir}")

    preferred_order = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG16", "VGG19"]
    model_to_root = {}
    for m, root in training_roots:
        if m not in model_to_root:
            model_to_root[m] = root

    made_any = False

    for model_name in preferred_order:
        if model_name not in model_to_root:
            with open(run_log, "a") as f:
                f.write(f"[WARN] Missing training folder for {model_name} under base_dir.\n")
            continue

        if model_name not in REFERENCE_EPOCH_BY_MODEL:
            with open(run_log, "a") as f:
                f.write(f"[WARN] No reference epoch configured for {model_name}. Skipping.\n")
            continue

        model_root = model_to_root[model_name]
        val_dir = find_validation_folder(model_root, model_name)
        if val_dir is None:
            with open(run_log, "a") as f:
                f.write(f"[WARN] Missing validation folder for {model_name} under {model_root}\n")
            continue

        cal_dir = os.path.join(val_dir, "calibration")
        epoch_ref = int(REFERENCE_EPOCH_BY_MODEL[model_name])

        out_png = os.path.join(out_dir, f"{model_name}_calibration_envelope_epoch{epoch_ref:02d}.png")
        out_csv = os.path.join(out_dir, f"{model_name}_calibration_envelope_epoch{epoch_ref:02d}.csv")

        ok, msg = plot_calibration_envelope_for_model(
            model_name=model_name,
            cal_dir=cal_dir,
            out_png=out_png,
            out_csv_grid_curves=out_csv,
            epoch_ref=epoch_ref,
            min_epoch=MIN_EPOCH,
            max_epoch=MAX_EPOCH,
            grid_n=GRID_N,
        )
        with open(run_log, "a") as f:
            f.write(msg + "\n")

        if ok:
            made_any = True

    with open(run_log, "a") as f:
        f.write("\nDONE\n")

    if not made_any:
        raise RuntimeError("No calibration envelopes were generated. Check calibration folder paths and CSV availability.")

    print("DONE")
    print("Output folder:")
    print(" ", out_dir)
    print("Log:")
    print(" ", run_log)


if __name__ == "__main__":
    main()