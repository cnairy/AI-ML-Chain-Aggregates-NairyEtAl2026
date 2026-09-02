#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ALL VGG16 epoch checkpoints on the VALIDATION set using a CONSTANT
threshold (>= 0.667 => chain), and write per-epoch artifacts plus summary plots
to a NEW output directory.

This is the same style as your ResNet* validation-eval scripts:
- Constant threshold
- FP/FN lists per epoch
- ROC + Calibration curves saved (CSV + PNG)
- Confusion matrices (white->yellow, bold black text, colorbar, dpi=300)
- Summary plots (including combined FP+FN plot)
- Best epoch selection rule: minimize FP, then maximize F1, then minimize FN
- Progress bars via tqdm
- Does NOT overwrite your training outputs (writes to a new sibling folder)

You MUST set:
  MODEL_DIR = ".../REFIT_FULLTRAIN_best_.../"  (folder containing model_epoch_*.pt)

Run:
  python3 VGG16_validation_constant_thresh.py
"""

import os
import glob
import csv
import re
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


# ============================================================
# USER CONFIG
# ============================================================

CNN_NAME = "VGG16"
THRESHOLD = 0.667

DATA_DIR = "/path/to/nas_workspace/CNN/Addtional_CPI_Images/CPI_images_aug_split_noleakage/"

# Set this to the folder that contains:
#   model_epoch_01_*.pt, model_epoch_02_*.pt, ... (your saved per-epoch checkpoints)
MODEL_DIR = "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-VGG16_training_evalutation_20251014-hyp-cv-30ep/REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_wu6_block4_5_adamw_plateau_bs64_ls0.0_luma_aug1/"

# Output directory will be created as a SIBLING of MODEL_DIR
OUT_DIR_NAME = "VGG16_validation_constant_thresh"

BATCH_SIZE = 64
NUM_WORKERS = 8
N_CAL_BINS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your VGG16 training normalization
MEAN, STD = 0.278238, 0.180433


# ============================================================
# TRANSFORMS (MATCH TRAINING VAL)
# ============================================================
val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
])


# ============================================================
# MODEL (MATCH YOUR VGG16 TRAINING SCRIPT)
# - features[0] changed to 1-channel conv with LUMA init
# - classifier = Dropout(0.30) + Linear(512*7*7 -> 1)
# ============================================================
def build_model(dropout=0.30, init_mode="luma"):
    base = models.vgg16(weights=VGG16_Weights.DEFAULT)

    # Convert first conv to 1-channel with luma init (matches your training default)
    old_w = base.features[0].weight.data.clone()  # [64,3,3,3]
    new_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        if init_mode == "luma":
            w = old_w
            luma = 0.2989 * w[:, 0, :, :] + 0.5870 * w[:, 1, :, :] + 0.1140 * w[:, 2, :, :]
            new_conv1.weight[:, 0, :, :] = luma
        else:
            new_conv1.weight[:] = old_w.mean(dim=1, keepdim=True)
    base.features[0] = new_conv1

    # Binary head
    base.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(512 * 7 * 7, 1)
    )

    return base.to(DEVICE)


# ============================================================
# METRICS: ECE
# ============================================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        m = (y_prob >= left) & ((y_prob < right) if i < n_bins - 1 else (y_prob <= right))
        if not np.any(m):
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += np.abs(acc - conf) * m.mean()
    return float(ece)


# ============================================================
# SAVE HELPERS
# ============================================================
def ensure_dirs(base):
    for sub in ["confusion_matrices", "roc", "calibration", "summary_plots", "error_analysis"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def save_line_plot(x, y, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def parse_epoch_from_filename(fname):
    # Prefer model_epoch_XX...
    parts = os.path.basename(fname).split("_")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "epoch":
        try:
            return int(parts[2])
        except Exception:
            pass
    m = re.search(r"epoch[_-]?(\d+)", os.path.basename(fname))
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse epoch from filename: {fname}")

def white_to_yellow_cmap():
    # True white -> yellow
    return LinearSegmentedColormap.from_list("white_yellow", ["#FFFFFF", "#FFFF00"])


# ============================================================
# MAIN
# ============================================================
def main():
    parent = os.path.dirname(MODEL_DIR.rstrip("/"))
    out_dir = os.path.join(parent, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    ensure_dirs(out_dir)

    # Dataset + loader
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Checkpoints
    ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "model_epoch_*.pt")))
    if len(ckpts) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in MODEL_DIR: {MODEL_DIR}\n"
            "Expected files like model_epoch_01_*.pt"
        )

    # Summary CSV
    summary_csv = os.path.join(out_dir, "summary_metrics.csv")
    summary_header = [
        "epoch",
        "TN", "FP", "FN", "TP",
        "accuracy", "precision", "recall", "f1",
        "roc_auc",
        "logloss",
        "brier",
        "ece"
    ]
    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerow(summary_header)

    rows = []

    cm_cmap = white_to_yellow_cmap()

    for ckpt in tqdm(ckpts, desc=f"Evaluating {CNN_NAME} epochs", unit="epoch"):
        epoch = parse_epoch_from_filename(ckpt)

        model = build_model(dropout=0.30, init_mode="luma")
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        probs_list, labels_list = [], []

        with torch.no_grad():
            for imgs, y in tqdm(val_loader, desc=f"Val batches (epoch {epoch:02d})", leave=False):
                imgs = imgs.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                p = torch.sigmoid(logits).cpu().numpy().ravel()
                probs_list.append(p)
                labels_list.append(y.numpy())

        y_prob = np.concatenate(probs_list)
        y_true = np.concatenate(labels_list).astype(int)
        y_pred = (y_prob >= THRESHOLD).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        acc  = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, zero_division=0))

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))

        ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
        bs = float(brier_score_loss(y_true, y_prob))
        ece = float(expected_calibration_error(y_true.astype(float), y_prob, n_bins=N_CAL_BINS))

        row = [epoch, tn, fp, fn, tp, acc, prec, rec, f1, roc_auc, ll, bs, ece]
        rows.append(row)

        with open(summary_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # Error analysis (FP/FN lists)
        paths = [s[0] for s in val_dataset.samples]
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        fp_rows = [(paths[i], int(y_true[i]), int(y_pred[i]), float(y_prob[i])) for i in np.where(fp_mask)[0]]
        fn_rows = [(paths[i], int(y_true[i]), int(y_pred[i]), float(y_prob[i])) for i in np.where(fn_mask)[0]]

        write_csv(
            os.path.join(out_dir, "error_analysis", f"false_positives_epoch_{epoch:02d}.csv"),
            ["filepath", "y_true", "y_pred", "prob_chain"],
            fp_rows
        )
        write_csv(
            os.path.join(out_dir, "error_analysis", f"false_negatives_epoch_{epoch:02d}.csv"),
            ["filepath", "y_true", "y_pred", "prob_chain"],
            fn_rows
        )

        # Confusion matrix (PNG + CSV)
        cm = np.array([[tn, fp], [fn, tp]])
        np.savetxt(
            os.path.join(out_dir, "confusion_matrices", f"cm_epoch_{epoch:02d}.csv"),
            cm,
            delimiter=",",
            fmt="%d"
        )

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(cm, cmap=cm_cmap, vmin=0)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Count")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black"
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-chain", "Chain"])
        ax.set_yticklabels(["Non-chain", "Chain"])
        ax.set_xlabel("Predicted label", labelpad=10)
        ax.set_ylabel("True label", labelpad=10)
        ax.set_title(f"{CNN_NAME} | Epoch {epoch:02d}", pad=12)

        # Keep axis labels in-frame
        fig.subplots_adjust(left=0.18, right=0.88, bottom=0.15, top=0.88)

        plt.savefig(
            os.path.join(out_dir, "confusion_matrices", f"cm_epoch_{epoch:02d}.png"),
            dpi=300
        )
        plt.close(fig)

        # ROC curve (CSV + PNG)
        roc_csv = os.path.join(out_dir, "roc", f"roc_epoch_{epoch:02d}.csv")
        np.savetxt(
            roc_csv,
            np.column_stack([fpr, tpr]),
            delimiter=",",
            header="fpr,tpr",
            comments=""
        )

        plt.figure(figsize=(6.2, 5.2))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{CNN_NAME} ROC | Epoch {epoch:02d}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc", f"roc_epoch_{epoch:02d}.png"), dpi=300)
        plt.close()

        # Calibration curve (CSV + PNG)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=N_CAL_BINS, strategy="uniform")

        cal_csv = os.path.join(out_dir, "calibration", f"calibration_epoch_{epoch:02d}.csv")
        np.savetxt(
            cal_csv,
            np.column_stack([prob_pred, prob_true]),
            delimiter=",",
            header="mean_predicted_prob,observed_fraction_positive",
            comments=""
        )

        plt.figure(figsize=(6.2, 5.2))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraction positive")
        plt.title(f"{CNN_NAME} Calibration | Epoch {epoch:02d}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "calibration", f"calibration_epoch_{epoch:02d}.png"), dpi=300)
        plt.close()

    # ============================================================
    # SUMMARY PLOTS + BEST EPOCH REPORT
    # ============================================================
    rows_sorted = sorted(rows, key=lambda r: r[0])

    epochs = np.array([r[0] for r in rows_sorted])
    fps    = np.array([r[2] for r in rows_sorted])
    fns    = np.array([r[3] for r in rows_sorted])
    precs  = np.array([r[6] for r in rows_sorted], dtype=float)
    recs   = np.array([r[7] for r in rows_sorted], dtype=float)
    f1s    = np.array([r[8] for r in rows_sorted], dtype=float)
    aucs   = np.array([r[9] for r in rows_sorted], dtype=float)
    lls    = np.array([r[10] for r in rows_sorted], dtype=float)
    briers = np.array([r[11] for r in rows_sorted], dtype=float)
    eces   = np.array([r[12] for r in rows_sorted], dtype=float)

    summary_plot_dir = os.path.join(out_dir, "summary_plots")

    save_line_plot(epochs, fns,    f"{CNN_NAME} | False Negatives", "Epoch", "FN",
                   os.path.join(summary_plot_dir, "fn_vs_epoch.png"))
    save_line_plot(epochs, fps,    f"{CNN_NAME} | False Positives", "Epoch", "FP",
                   os.path.join(summary_plot_dir, "fp_vs_epoch.png"))

    # Combined FP & FN plot
    plt.figure(figsize=(7.5, 5.2))
    plt.plot(epochs, fns, marker="o", label="FN")
    plt.plot(epochs, fps, marker="o", label="FP")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.title(f"{CNN_NAME} | FP and FN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_plot_dir, "fp_fn_vs_epoch.png"), dpi=300)
    plt.close()

    save_line_plot(epochs, recs,   f"{CNN_NAME} | Recall", "Epoch", "Recall",
                   os.path.join(summary_plot_dir, "recall_vs_epoch.png"))
    save_line_plot(epochs, precs,  f"{CNN_NAME} | Precision", "Epoch", "Precision",
                   os.path.join(summary_plot_dir, "precision_vs_epoch.png"))
    save_line_plot(epochs, f1s,    f"{CNN_NAME} | F1", "Epoch", "F1",
                   os.path.join(summary_plot_dir, "f1_vs_epoch.png"))
    save_line_plot(epochs, aucs,   f"{CNN_NAME} | ROC-AUC", "Epoch", "ROC-AUC",
                   os.path.join(summary_plot_dir, "rocauc_vs_epoch.png"))
    save_line_plot(epochs, lls,    f"{CNN_NAME} | Log Loss", "Epoch", "Log loss",
                   os.path.join(summary_plot_dir, "logloss_vs_epoch.png"))
    save_line_plot(epochs, briers, f"{CNN_NAME} | Brier Score", "Epoch", "Brier",
                   os.path.join(summary_plot_dir, "brier_vs_epoch.png"))
    save_line_plot(epochs, eces,   f"{CNN_NAME} | ECE", "Epoch", "ECE",
                   os.path.join(summary_plot_dir, "ece_vs_epoch.png"))

    # ============================================================
    # BEST EPOCH: minimize FP, then maximize F1, then minimize FN
    # np.lexsort: last key is primary -> (FN, -F1, FP)
    # ============================================================
    best_idx = np.lexsort((fns, -f1s, fps))[0]
    best_epoch = int(epochs[best_idx])

    best_path = os.path.join(out_dir, "best_epoch.txt")
    with open(best_path, "w") as f:
        f.write(f"CNN={CNN_NAME}\n")
        f.write(f"threshold={THRESHOLD}\n")
        f.write("selection_rule: minimize FP, then maximize F1, then minimize FN\n\n")
        f.write(f"BEST_EPOCH={best_epoch:02d}\n\n")
        f.write("Top-10 (epoch, FP, F1, FN, Recall, Precision, ROC-AUC, LogLoss, Brier, ECE)\n")
        order = np.lexsort((fns, -f1s, fps))
        for k in order[:10]:
            f.write(
                f"{int(epochs[k]):02d}, {int(fps[k])}, {f1s[k]:.6f}, {int(fns[k])}, "
                f"{recs[k]:.6f}, {precs[k]:.6f}, {aucs[k]:.6f}, "
                f"{lls[k]:.6f}, {briers[k]:.6f}, {eces[k]:.6f}\n"
            )

    print("\nDONE")
    print(f"Outputs written to:\n  {out_dir}")
    print(f"Summary CSV:\n  {summary_csv}")
    print(f"Summary plots:\n  {summary_plot_dir}")
    print(f"Best epoch report:\n  {best_path}")
    print(f"Best epoch (FP -> F1 -> FN): {best_epoch:02d}")


if __name__ == "__main__":
    main()
