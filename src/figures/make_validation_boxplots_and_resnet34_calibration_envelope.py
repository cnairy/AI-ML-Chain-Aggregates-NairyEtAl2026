#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-model validation plotting (no pandas):

1) Figure A (2x2 boxplots across folds, per model):
   ROC-AUC, PR-AUC, LogLoss (lower better), Brier (lower better)

2) Figure B (Precision/Recall/F1 boxplots across folds, per model)
   using the CHAIN class metrics at the operating threshold.

3) ResNet34 calibration ENVELOPE plot (epochs >= 7 only; warmup skipped):
   - epoch 18 curve in black
   - shaded min/max envelope across ALL OTHER epochs (>=7), no per-epoch lines
   - legend UNDER the plot (so plot area isn't vertically distorted)

It reads:
  - best_epoch.txt
  - summary_metrics.csv
  - calibration_epoch_XX.csv (for ResNet34 envelope)

It writes a new output folder and does not overwrite any model outputs.
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


def safe_int(x, default=None):
    try:
        return int(str(x).strip())
    except Exception:
        return default


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


def read_best_epoch(best_epoch_path):
    """
    Very permissive:
      - prefer lines containing 'epoch'
      - otherwise first integer anywhere
    """
    try:
        with open(best_epoch_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        return None

    for ln in lines:
        if "epoch" in ln.lower():
            m = re.search(r"(\d+)", ln)
            if m:
                return int(m.group(1))

    for ln in lines:
        m = re.search(r"(\d+)", ln)
        if m:
            return int(m.group(1))

    return None


def find_nearest_summary_csv(start_dir):
    d = start_dir
    while True:
        cand = os.path.join(d, "summary_metrics.csv")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def read_summary_row_for_epoch(summary_csv, epoch):
    """
    Reads summary_metrics.csv and returns metrics dict for one epoch.
    Supports a few common column-name variants.
    """
    with open(summary_csv, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return None

    header = [h.strip() for h in rows[0]]
    col = {h: i for i, h in enumerate(header)}

    aliases = {
        "epoch": ["epoch", "Epoch", "ep", "EP"],
        "tn": ["TN", "tn"],
        "fp": ["FP", "fp"],
        "fn": ["FN", "fn"],
        "tp": ["TP", "tp"],
        "precision": ["precision", "prec", "Precision", "prec_chain", "precision_chain"],
        "recall": ["recall", "rec", "Recall", "rec_chain", "recall_chain"],
        "f1": ["f1", "F1", "f1_score", "f1-score", "f1_chain", "f1score_chain"],
        "roc_auc": ["roc_auc", "roc-auc", "roc", "ROC_AUC", "auroc", "val_roc_auc"],
        "pr_auc": ["pr_auc", "pr-auc", "average_precision", "ap", "AP", "PR_AUC", "val_pr_auc"],
        "logloss": ["logloss", "log_loss", "nll", "LogLoss", "val_logloss"],
        "brier": ["brier", "brier_score", "Brier", "val_brier"],
        "ece": ["ece", "ECE", "val_ece"],
    }

    def find_col(canon):
        for k in aliases.get(canon, []):
            if k in col:
                return k
        return None

    epoch_col = find_col("epoch")
    if epoch_col is None:
        return None

    target_row = None
    for r in rows[1:]:
        if not r or len(r) <= col[epoch_col]:
            continue
        ep = safe_int(r[col[epoch_col]])
        if ep == epoch:
            target_row = r
            break

    if target_row is None:
        return None

    out = {"epoch": epoch}
    for canon in ["tn", "fp", "fn", "tp", "precision", "recall", "f1",
                  "roc_auc", "pr_auc", "logloss", "brier", "ece"]:
        key = find_col(canon)
        out[canon] = safe_float(target_row[col[key]]) if key is not None else float("nan")

    return out


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
    Supports either header:
      - prob_pred,prob_true
      - mean_predicted_prob,observed_fraction_positive
    Returns x,y arrays.
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

    if len(header) >= 2 and any(s in header[0] for s in ["prob_pred", "mean_predicted_prob", "mean_pred"]):
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

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    x_unique = np.clip(x_unique, 0.0, 1.0)
    y_unique = np.clip(y_unique, 0.0, 1.0)

    return x_unique, y_unique


def interp_curve_to_grid(x, y, grid):
    if x is None or y is None or len(x) < 2:
        return None
    return np.interp(grid, x, y, left=y[0], right=y[-1])


# ----------------------------
# Plotting: Figure A (2x2)
# ----------------------------
def plot_boxplots_scores(out_png, model_order, data_by_model):
    set_common_style()

    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("logloss", "LogLoss (lower is better)"),
        ("brier", "Brier (lower is better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=300)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, metrics):
        vals = []
        for m in model_order:
            arr = [d.get(key, float("nan")) for d in data_by_model.get(m, [])]
            arr = [v for v in arr if math.isfinite(v)]
            vals.append(arr)

        bp = ax.boxplot(vals, patch_artist=True, showfliers=True)
        for box in bp["boxes"]:
            box.set_facecolor("#e6e6e6")
            box.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)

        ax.set_title(title)
        ax.set_xticks(np.arange(1, len(model_order) + 1))
        ax.set_xticklabels(model_order, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ----------------------------
# Plotting: Figure B (P/R/F1)
# ----------------------------
def plot_boxplots_prf(out_png, model_order, data_by_model):
    set_common_style()

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=300)

    base_positions = np.arange(len(model_order)) * 4.0
    offsets = np.array([-0.8, 0.0, 0.8])

    colors = {
        "f1": "#4c78a8",
        "precision": "#f58518",
        "recall": "#54a24b",
    }
    labels = {"f1": "F1-score", "precision": "Precision", "recall": "Recall"}

    def get_vals(key, model):
        arr = [d.get(key, float("nan")) for d in data_by_model.get(model, [])]
        return [v for v in arr if math.isfinite(v)]

    for j, key in enumerate(["f1", "precision", "recall"]):
        vals = [get_vals(key, m) for m in model_order]
        pos = base_positions + offsets[j]
        bp = ax.boxplot(vals, positions=pos, widths=0.6, patch_artist=True, showfliers=True)
        for box in bp["boxes"]:
            box.set_facecolor(colors[key])
            box.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.4)

    ax.set_xlim(base_positions[0] - 2.0, base_positions[-1] + 2.0)
    ax.set_xticks(base_positions)
    ax.set_xticklabels(model_order)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Value")
    ax.set_title("Precision / Recall / F1 at operating threshold (CHAIN class, per fold)")
    ax.grid(True, axis="y", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=colors["f1"], lw=10),
        plt.Line2D([0], [0], color=colors["precision"], lw=10),
        plt.Line2D([0], [0], color=colors["recall"], lw=10),
    ]
    ax.legend(handles, [labels["f1"], labels["precision"], labels["recall"]], loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ----------------------------
# Calibration envelope plot (ENVELOPE ONLY, legend BELOW)
# ----------------------------
def plot_resnet34_calibration_envelope(
    cal_dir,
    out_png,
    out_csv_grid_curves,
    epoch_ref=18,
    grid_n=201,
    min_epoch=7,
    max_epoch=30
):
    """
    Reads calibration_epoch_XX.csv files in cal_dir, but only uses epochs in [min_epoch, max_epoch].

    Output plot:
      - Perfect calibration dashed line
      - Shaded min/max envelope across ALL other epochs (excluding epoch_ref)
      - Envelope boundaries drawn as thin dark-blue lines
      - Epoch epoch_ref curve in black
      - Legend UNDER the axes (so axes height is not compressed)
      - No individual per-epoch lines (envelope-only)
    """
    if not os.path.isdir(cal_dir):
        return False, f"Calibration directory not found: {cal_dir}"

    files = sorted(glob.glob(os.path.join(cal_dir, "calibration_epoch_*.csv")))
    if not files:
        return False, f"No calibration_epoch_*.csv files in: {cal_dir}"

    grid = np.linspace(0.0, 1.0, grid_n)

    epoch_to_curve = {}
    for fp in files:
        m = re.search(r"calibration_epoch_(\d+)\.csv$", os.path.basename(fp))
        if not m:
            continue
        ep = int(m.group(1))
        if ep < min_epoch or ep > max_epoch:
            continue
        x, y = read_calibration_csv(fp)
        ygrid = interp_curve_to_grid(x, y, grid)
        if ygrid is None:
            continue
        epoch_to_curve[ep] = ygrid

    if epoch_ref < min_epoch or epoch_ref > max_epoch:
        return False, f"epoch_ref={epoch_ref} is outside [{min_epoch}, {max_epoch}]"
    if epoch_ref not in epoch_to_curve:
        return False, f"Missing calibration for epoch {epoch_ref:02d} in {cal_dir}"

    other_epochs = [ep for ep in sorted(epoch_to_curve.keys()) if ep != epoch_ref]
    if len(other_epochs) < 1:
        return False, "Not enough non-reference epochs to build an envelope."

    stack = np.vstack([epoch_to_curve[ep] for ep in other_epochs])
    y_min = np.nanmin(stack, axis=0)
    y_max = np.nanmax(stack, axis=0)

    # Save gridded curves to CSV (for reproducibility / plotting elsewhere)
    epochs_sorted = sorted(epoch_to_curve.keys())
    header = ["prob"] + [f"epoch_{ep:02d}" for ep in epochs_sorted] + ["env_min_excl_ref", "env_max_excl_ref"]
    rows = []
    for i in range(grid.size):
        row = [f"{grid[i]:.6f}"]
        for ep in epochs_sorted:
            row.append(f"{epoch_to_curve[ep][i]:.6f}")
        row.append(f"{y_min[i]:.6f}")
        row.append(f"{y_max[i]:.6f}")
        rows.append(row)
    save_csv(out_csv_grid_curves, header, rows)

    set_common_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=300)

    # Styling for envelope
    env_edge = "#1f4e79"   # dark-ish blue
    env_fill_alpha = 0.18  # more transparent than before
    env_lw = 1.0

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="0.5", label="Perfect calibration")

    # Fill
    ax.fill_between(
        grid, y_min, y_max,
        alpha=env_fill_alpha,
        label=f"Envelope (epochs {min_epoch}–{max_epoch}, excluding {epoch_ref})"
    )
    # Outer boundary lines
    ax.plot(grid, y_min, color=env_edge, linewidth=env_lw, label=None)
    ax.plot(grid, y_max, color=env_edge, linewidth=env_lw, label=None)

    # Reference curve
    ax.plot(
        grid, epoch_to_curve[epoch_ref],
        color="black", linewidth=2.2,
        label=f"Epoch {epoch_ref} (reference)"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"ResNet34 calibration envelope (epochs >= {min_epoch}; warmup skipped)")
    ax.grid(True, alpha=0.25)

    # Legend below the axes (keeps axes size unchanged)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=True
    )
    fig.subplots_adjust(bottom=0.25)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    return True, f"Saved calibration envelope to {out_png}"


# ----------------------------
# Main: scan + aggregate + plot
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True,
                    help="Base directory containing JTech-* training folders.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory. Default: {base_dir}/validation_summary_plots_{timestamp}")
    ap.add_argument("--cal_epoch_ref", type=int, default=18,
                    help="Reference epoch for ResNet34 calibration plot.")
    ap.add_argument("--cal_min_epoch", type=int, default=7,
                    help="Minimum epoch included in ResNet34 calibration plot (skip warmup: default 7).")
    ap.add_argument("--cal_max_epoch", type=int, default=30,
                    help="Maximum epoch included in ResNet34 calibration plot (default 30).")
    args = ap.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(base_dir, f"validation_summary_plots_{now_stamp()}")
    os.makedirs(out_dir, exist_ok=True)

    run_log = os.path.join(out_dir, "run_log.txt")
    with open(run_log, "w") as f:
        f.write(f"Run timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"BASE_DIR: {base_dir}\n")
        f.write(f"OUT_DIR:  {out_dir}\n")
        f.write(f"ResNet34 calibration: ref={args.cal_epoch_ref}, min_epoch={args.cal_min_epoch}, max_epoch={args.cal_max_epoch}\n")

    training_roots = list_training_roots(base_dir)
    if not training_roots:
        raise RuntimeError(f"No model training folders found under: {base_dir}")

    data_by_model = {}
    per_fold_rows = []

    for model_name, model_root in training_roots:
        val_dir = find_validation_folder(model_root, model_name)
        if val_dir is None:
            with open(run_log, "a") as f:
                f.write(f"[WARN] Missing validation folder for {model_name} under {model_root}\n")
            continue

        best_files = sorted(glob.glob(os.path.join(val_dir, "**", "best_epoch.txt"), recursive=True))
        if not best_files:
            with open(run_log, "a") as f:
                f.write(f"[WARN] No best_epoch.txt found under {val_dir}\n")
            continue

        data_by_model.setdefault(model_name, [])

        for bf in best_files:
            fold_dir = os.path.dirname(bf)
            best_ep = read_best_epoch(bf)
            if best_ep is None:
                continue

            summary_csv = find_nearest_summary_csv(fold_dir)
            if summary_csv is None:
                continue

            metrics = read_summary_row_for_epoch(summary_csv, best_ep)
            if metrics is None:
                continue

            rel = os.path.relpath(fold_dir, val_dir)

            rec = {
                "model": model_name,
                "fold_run": rel,
                "best_epoch": best_ep,
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "f1": metrics.get("f1", float("nan")),
                "fp": metrics.get("fp", float("nan")),
                "fn": metrics.get("fn", float("nan")),
                "tp": metrics.get("tp", float("nan")),
                "tn": metrics.get("tn", float("nan")),
                "roc_auc": metrics.get("roc_auc", float("nan")),
                "pr_auc": metrics.get("pr_auc", float("nan")),
                "logloss": metrics.get("logloss", float("nan")),
                "brier": metrics.get("brier", float("nan")),
                "ece": metrics.get("ece", float("nan")),
                "best_epoch_txt": bf,
                "summary_metrics_csv": summary_csv,
            }

            data_by_model[model_name].append(rec)

            per_fold_rows.append([
                rec["model"], rec["fold_run"], rec["best_epoch"],
                f"{rec['precision']:.6f}" if math.isfinite(rec["precision"]) else "",
                f"{rec['recall']:.6f}" if math.isfinite(rec["recall"]) else "",
                f"{rec['f1']:.6f}" if math.isfinite(rec["f1"]) else "",
                int(rec["fp"]) if math.isfinite(rec["fp"]) else "",
                int(rec["fn"]) if math.isfinite(rec["fn"]) else "",
                int(rec["tp"]) if math.isfinite(rec["tp"]) else "",
                int(rec["tn"]) if math.isfinite(rec["tn"]) else "",
                f"{rec['roc_auc']:.6f}" if math.isfinite(rec["roc_auc"]) else "",
                f"{rec['pr_auc']:.6f}" if math.isfinite(rec["pr_auc"]) else "",
                f"{rec['logloss']:.6f}" if math.isfinite(rec["logloss"]) else "",
                f"{rec['brier']:.6f}" if math.isfinite(rec["brier"]) else "",
                f"{rec['ece']:.6f}" if math.isfinite(rec["ece"]) else "",
                rec["best_epoch_txt"],
                rec["summary_metrics_csv"],
            ])

    out_table = os.path.join(out_dir, "per_fold_metrics_all_models.csv")
    save_csv(
        out_table,
        header=[
            "model", "fold_run", "best_epoch",
            "precision", "recall", "f1",
            "fp", "fn", "tp", "tn",
            "roc_auc", "pr_auc", "logloss", "brier", "ece",
            "best_epoch_txt", "summary_metrics_csv",
        ],
        rows=per_fold_rows,
    )

    preferred_order = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG16", "VGG19"]
    model_order = [m for m in preferred_order if m in data_by_model and len(data_by_model[m]) > 0]
    if not model_order:
        raise RuntimeError("No usable per-fold metrics found. Check your best_epoch.txt and summary_metrics.csv locations.")

    figA = os.path.join(out_dir, "fig_boxplots_scores.png")
    plot_boxplots_scores(figA, model_order, data_by_model)

    figB = os.path.join(out_dir, "fig_boxplots_precision_recall_f1.png")
    plot_boxplots_prf(figB, model_order, data_by_model)

    resnet34_root = None
    for m, root in training_roots:
        if m == "ResNet34":
            resnet34_root = root
            break

    if resnet34_root is not None:
        resnet34_val = find_validation_folder(resnet34_root, "ResNet34")
        if resnet34_val is not None:
            cal_dir = os.path.join(resnet34_val, "calibration")
            out_cal_png = os.path.join(out_dir, "resnet34_calibration_epoch18_envelope.png")
            out_cal_csv = os.path.join(out_dir, "resnet34_calibration_gridded_curves_and_envelope.csv")

            ok, msg = plot_resnet34_calibration_envelope(
                cal_dir=cal_dir,
                out_png=out_cal_png,
                out_csv_grid_curves=out_cal_csv,
                epoch_ref=args.cal_epoch_ref,
                min_epoch=args.cal_min_epoch,
                max_epoch=args.cal_max_epoch,
            )
            with open(run_log, "a") as f:
                f.write(msg + "\n")
        else:
            with open(run_log, "a") as f:
                f.write("[WARN] ResNet34_validation_constant_thresh folder not found for calibration plot.\n")
    else:
        with open(run_log, "a") as f:
            f.write("[WARN] ResNet34 training folder not found under base_dir.\n")

    with open(run_log, "a") as f:
        f.write("\nDONE\n")
        f.write(f"Saved: {out_table}\n")
        f.write(f"Saved: {figA}\n")
        f.write(f"Saved: {figB}\n")
        f.write(f"Saved: {run_log}\n")

    print("DONE")
    print("Output folder:")
    print(" ", out_dir)
    print("Key files:")
    print(" ", out_table)
    print(" ", figA)
    print(" ", figB)
    print(" ", run_log)


if __name__ == "__main__":
    main()
