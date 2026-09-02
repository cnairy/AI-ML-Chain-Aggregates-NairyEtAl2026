#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: ResNet34_threshold_sensitivity_analysis.py

Purpose
-------
Run a focused threshold-sensitivity analysis for the selected ResNet34
checkpoint using the CLEAN, unaugmented validation subset.

The script compares four thresholding approaches:
  1) Standard binary cutoff: 0.500
  2) Unconstrained validation F1-optimal threshold
  3) Threshold floor itself: 0.667
  4) Validation F1-optimal threshold subject to threshold >= 0.667

IMPORTANT
---------
- Thresholds are selected ONLY from the validation data.
- The independent test set is not used anywhere in threshold selection.
- Metrics reported here are CHAIN-CLASS precision, recall, and F1.

Outputs
-------
A new folder is created as a sibling of MODEL_DIR:

  ResNet34_threshold_sensitivity_epoch18/

containing:
  validation_predictions_epoch18.csv
      Per-image validation truth + predicted chain probability.

  threshold_sensitivity_summary.csv
      Direct comparison of the four threshold approaches.

  threshold_sweep.csv
      Metrics at every empirical threshold returned by the validation
      precision-recall curve, including whether threshold >= 0.667.

  threshold_sensitivity_curves.png
  threshold_sensitivity_curves.pdf
      Precision, recall, and chain-class F1 versus threshold, with vertical
      markers for 0.500, 0.667, the unconstrained optimum, and the constrained
      optimum.

  threshold_sensitivity_report.txt
      Human-readable summary of the selected thresholds and metrics.

This script DOES NOT modify or overwrite any model-training outputs.
"""

import os
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import ResNet34_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from tqdm import tqdm


# ============================================================
# USER CONFIGURATION
# ============================================================

CNN_NAME = "ResNet34"
SELECTED_EPOCH = 18
THRESHOLD_FLOOR = 0.667
STANDARD_THRESHOLD = 0.500

DATA_DIR = (
    "/path/to/nas_workspace/CNN/Addtional_CPI_Images/"
    "CPI_images_aug_split_noleakage/"
)

# Folder containing model_epoch_01_*.pt ... model_epoch_30_*.pt
MODEL_DIR = (
    "/path/to/nas_workspace/CNN/figures/30_epochs/"
    "JTech-ResNet34_training_evalutation_20251021d-hyp-cv-30ep/"
    "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
    "wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/"
)

OUT_DIR_NAME = f"ResNet34_threshold_sensitivity_epoch{SELECTED_EPOCH:02d}"

BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DPI = 300


# ============================================================
# VALIDATION TRANSFORM — MATCHES TRAINING VALIDATION PREPROCESSING
# ============================================================

val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,)),
])


# ============================================================
# MODEL — MATCHES RESNET34 TRAINING ARCHITECTURE
# ============================================================

def build_model():
    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    # Convert first convolution to one channel using luma initialization.
    old_w = model.conv1.weight.data.clone()
    new_conv = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    with torch.no_grad():
        luma = (
            0.2989 * old_w[:, 0]
            + 0.5870 * old_w[:, 1]
            + 0.1140 * old_w[:, 2]
        )
        new_conv.weight[:, 0] = luma

    model.conv1 = new_conv

    # Match the training classification head.
    model.fc = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(model.fc.in_features, 1),
    )

    return model.to(DEVICE)


# ============================================================
# HELPERS
# ============================================================

def find_selected_checkpoint():
    pattern = os.path.join(
        MODEL_DIR, f"model_epoch_{SELECTED_EPOCH:02d}_*.pt"
    )
    matches = sorted(glob.glob(pattern))

    if not matches:
        # Fallback for filenames that do not zero-pad the epoch.
        pattern = os.path.join(
            MODEL_DIR, f"model_epoch_{SELECTED_EPOCH}_*.pt"
        )
        matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"Could not find epoch {SELECTED_EPOCH} checkpoint in:\n"
            f"  {MODEL_DIR}\n"
            f"Expected a file similar to model_epoch_{SELECTED_EPOCH:02d}_*.pt"
        )

    if len(matches) > 1:
        print("WARNING: Multiple matching checkpoints found. Using:")
        print(f"  {matches[0]}")

    return matches[0]


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "chain_precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "chain_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "chain_f1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
    }


def empirical_threshold_metrics(y_true, y_prob):
    """
    Return metrics for every empirical threshold from sklearn's
    precision-recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    rows = []
    for i, threshold in enumerate(thresholds):
        p = float(precision[i])
        r = float(recall[i])
        f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)

        # Get confusion counts and accuracy using exactly this threshold.
        m = metrics_at_threshold(y_true, y_prob, threshold)

        rows.append({
            "threshold": float(threshold),
            "chain_precision": p,
            "chain_recall": r,
            "chain_f1": float(f1),
            "accuracy": m["accuracy"],
            "TN": m["TN"],
            "FP": m["FP"],
            "FN": m["FN"],
            "TP": m["TP"],
            "eligible_ge_0.667": bool(threshold >= THRESHOLD_FLOOR),
        })

    if not rows:
        raise RuntimeError("No candidate thresholds were returned.")

    return rows


def choose_best_f1(rows, floor=None):
    eligible = rows
    if floor is not None:
        eligible = [r for r in rows if r["threshold"] >= floor]

    if not eligible:
        raise RuntimeError(
            f"No empirical thresholds were available at or above {floor:.3f}."
        )

    # Primary criterion: maximum chain-class F1.
    # Tie-break 1: higher precision (more conservative positive predictions).
    # Tie-break 2: higher threshold.
    return max(
        eligible,
        key=lambda r: (
            r["chain_f1"],
            r["chain_precision"],
            r["threshold"],
        ),
    )


def write_dict_rows_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    parent = os.path.dirname(MODEL_DIR.rstrip("/"))
    out_dir = os.path.join(parent, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Validation data: {os.path.join(DATA_DIR, 'val')}")
    print(f"Output directory: {out_dir}")

    checkpoint = find_selected_checkpoint()
    print(f"Checkpoint: {checkpoint}")

    # --------------------------------------------------------
    # Load clean validation dataset
    # --------------------------------------------------------
    val_root = os.path.join(DATA_DIR, "val")
    val_dataset = datasets.ImageFolder(val_root, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Validation images: {len(val_dataset):,}")
    print(f"Class mapping: {val_dataset.class_to_idx}")

    # --------------------------------------------------------
    # Load selected model checkpoint
    # --------------------------------------------------------
    model = build_model()
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    # --------------------------------------------------------
    # Validation inference
    # --------------------------------------------------------
    probs_list = []
    labels_list = []

    with torch.no_grad():
        for imgs, labels in tqdm(
            val_loader,
            desc=f"Evaluating {CNN_NAME} epoch {SELECTED_EPOCH:02d}",
            unit="batch",
        ):
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

            probs_list.append(probs)
            labels_list.append(labels.numpy().ravel())

    y_prob = np.concatenate(probs_list)
    y_true = np.concatenate(labels_list).astype(int)

    # --------------------------------------------------------
    # Save per-image predictions
    # --------------------------------------------------------
    paths = [sample[0] for sample in val_dataset.samples]
    pred_path = os.path.join(
        out_dir, f"validation_predictions_epoch{SELECTED_EPOCH:02d}.csv"
    )

    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filepath",
            "true_label",
            "true_class",
            "prob_chain",
        ])
        for path, y, prob in zip(paths, y_true, y_prob):
            writer.writerow([
                path,
                int(y),
                "chain" if int(y) == 1 else "non-chain",
                f"{float(prob):.10f}",
            ])

    # --------------------------------------------------------
    # Full empirical threshold sweep
    # --------------------------------------------------------
    sweep_rows = empirical_threshold_metrics(y_true, y_prob)

    sweep_path = os.path.join(out_dir, "threshold_sweep.csv")
    write_dict_rows_csv(
        sweep_path,
        sweep_rows,
        [
            "threshold",
            "chain_precision",
            "chain_recall",
            "chain_f1",
            "accuracy",
            "TN", "FP", "FN", "TP",
            "eligible_ge_0.667",
        ],
    )

    # --------------------------------------------------------
    # Select unconstrained and constrained F1-optimal thresholds
    # --------------------------------------------------------
    best_unconstrained = choose_best_f1(sweep_rows, floor=None)
    best_constrained = choose_best_f1(
        sweep_rows, floor=THRESHOLD_FLOOR
    )

    thr_unconstrained = float(best_unconstrained["threshold"])
    thr_constrained = float(best_constrained["threshold"])

    # --------------------------------------------------------
    # Compare four threshold approaches
    # --------------------------------------------------------
    comparisons = []

    methods = [
        ("Standard 0.500 cutoff", STANDARD_THRESHOLD),
        ("Unconstrained validation F1 optimum", thr_unconstrained),
        ("Fixed 0.667 threshold", THRESHOLD_FLOOR),
        ("Validation F1 optimum with >=0.667 floor", thr_constrained),
    ]

    for method, threshold in methods:
        m = metrics_at_threshold(y_true, y_prob, threshold)
        m = {"method": method, **m}
        comparisons.append(m)

    summary_path = os.path.join(
        out_dir, "threshold_sensitivity_summary.csv"
    )
    write_dict_rows_csv(
        summary_path,
        comparisons,
        [
            "method",
            "threshold",
            "chain_precision",
            "chain_recall",
            "chain_f1",
            "accuracy",
            "TN", "FP", "FN", "TP",
        ],
    )

    # --------------------------------------------------------
    # Threshold-sensitivity plot
    # --------------------------------------------------------
    thresholds = np.array([r["threshold"] for r in sweep_rows], dtype=float)
    precisions = np.array([r["chain_precision"] for r in sweep_rows], dtype=float)
    recalls = np.array([r["chain_recall"] for r in sweep_rows], dtype=float)
    f1s = np.array([r["chain_f1"] for r in sweep_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.plot(thresholds, precisions, linewidth=1.8, label="Chain precision")
    ax.plot(thresholds, recalls, linewidth=1.8, label="Chain recall")
    ax.plot(thresholds, f1s, linewidth=2.0, label="Chain F1")

    ax.axvline(
        STANDARD_THRESHOLD,
        linestyle=":",
        linewidth=1.0,
        alpha=0.65,
        label="0.500 cutoff",
    )
    ax.axvline(
        THRESHOLD_FLOOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="0.667 floor",
    )
    ax.axvline(
        thr_unconstrained,
        linestyle="-.",
        linewidth=1.0,
        alpha=0.80,
        label=f"Unconstrained optimum ({thr_unconstrained:.3f})",
    )
    ax.axvline(
        thr_constrained,
        linestyle="-.",
        linewidth=1.4,
        alpha=0.95,
        label=f">=0.667 optimum ({thr_constrained:.3f})",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Chain probability threshold")
    ax.set_ylabel("Validation metric")
    ax.set_title(
        f"{CNN_NAME} epoch {SELECTED_EPOCH:02d}: threshold sensitivity"
    )
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    fig.savefig(
        os.path.join(out_dir, "threshold_sensitivity_curves.png"),
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(out_dir, "threshold_sensitivity_curves.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    report_path = os.path.join(
        out_dir, "threshold_sensitivity_report.txt"
    )

    with open(report_path, "w") as f:
        f.write("ResNet34 threshold sensitivity analysis\n")
        f.write("=======================================\n\n")
        f.write(f"Selected epoch: {SELECTED_EPOCH}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Validation dataset: {val_root}\n")
        f.write(f"Validation N: {len(y_true)}\n")
        f.write(f"Validation chains: {int(np.sum(y_true == 1))}\n")
        f.write(f"Validation non-chains: {int(np.sum(y_true == 0))}\n")
        f.write(f"PR-AUC: {pr_auc:.6f}\n")
        f.write(f"ROC-AUC: {roc_auc:.6f}\n\n")

        f.write(
            f"Unconstrained F1-optimal threshold: "
            f"{thr_unconstrained:.6f}\n"
        )
        f.write(
            f"F1-optimal threshold with >=0.667 floor: "
            f"{thr_constrained:.6f}\n\n"
        )

        for row in comparisons:
            f.write(f"{row['method']}\n")
            f.write(f"  threshold = {row['threshold']:.6f}\n")
            f.write(f"  chain precision = {row['chain_precision']:.6f}\n")
            f.write(f"  chain recall = {row['chain_recall']:.6f}\n")
            f.write(f"  chain F1 = {row['chain_f1']:.6f}\n")
            f.write(f"  accuracy = {row['accuracy']:.6f}\n")
            f.write(
                f"  TN={row['TN']} FP={row['FP']} "
                f"FN={row['FN']} TP={row['TP']}\n\n"
            )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\nDONE")
    print(f"Outputs written to:\n  {out_dir}\n")
    print("Key comparison:")
    for row in comparisons:
        print(
            f"  {row['method']}: "
            f"thr={row['threshold']:.6f}, "
            f"P={row['chain_precision']:.4f}, "
            f"R={row['chain_recall']:.4f}, "
            f"F1={row['chain_f1']:.4f}, "
            f"FP={row['FP']}, FN={row['FN']}"
        )

    print("\nFiles:")
    print(f"  {summary_path}")
    print(f"  {sweep_path}")
    print(f"  {pred_path}")
    print(f"  {report_path}")
    print(f"  {os.path.join(out_dir, 'threshold_sensitivity_curves.png')}")
    print(f"  {os.path.join(out_dir, 'threshold_sensitivity_curves.pdf')}")


if __name__ == "__main__":
    main()
