#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JTECH supplemental learning-curve figure generator.

Reads epoch_metrics.csv from the six final refit directories and creates
three alternative supplemental figure designs.

Outputs:
  option1_loss_small_multiples_2x3.png/.pdf
  option2_validation_metrics_by_family_3x2.png/.pdf
  option3_selected_model_diagnostics_resnet34.png/.pdf
  learning_curve_summary.csv

All figures are 13.33 inches wide and exported at 300 dpi.

Author: generated for JTECH chain-aggregate CNN manuscript
Date: 2026-08-18
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER CONFIGURATION
# ============================================================

MODEL_DIRS = {
    "ResNet18": Path(
        "/path/to/nas_workspace/CNN/figures/30_epochs/"
        "JTech-ResNet18_training_evalutation_20251021-hyp-cv-30ep/"
        "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
        "wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/plot_data"
    ),
    "ResNet34": Path(
        "/path/to/nas_workspace/CNN/figures/30_epochs/"
        "JTech-ResNet34_training_evalutation_20251021d-hyp-cv-30ep/"
        "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
        "wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/plot_data"
    ),
    "ResNet50": Path(
        "/path/to/nas_workspace/CNN/figures/30_epochs/"
        "JTech-ResNet50_training_evalutation_20251104d-hyp-cv-30ep/"
        "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
        "wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/plot_data"
    ),
    "ResNet101": Path(
        "/path/to/nas_workspace/CNN/figures/30_epochs/"
        "JTech-ResNet101_training_evalutation_20251105-hyp-cv-30ep/"
        "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
        "wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/plot_data"
    ),
    "VGG16": Path(
        "/path/to/nas_workspace/CNN/figures/30_epochs/"
        "JTech-VGG16_training_evalutation_20251014-hyp-cv-30ep/"
        "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
        "wu6_block4_5_adamw_plateau_bs64_ls0.0_luma_aug1/plot_data"
    ),
    "VGG19": Path(
        "/path/to/nas_workspace/CNN/figures/30_epochs/"
        "JTech-VGG19_training_evalutation_20251106-hyp-cv-30ep/"
        "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_"
        "wu6_block4_5_adamw_plateau_bs64_ls0.0_luma_aug1/plot_data"
    ),
}

OUTPUT_DIR = Path(
    "/path/to/nas_workspace/CNN/figures/30_epochs/"
    "JTECH_Supplemental_Learning_Curves_20260818"
)

# Warm-up epochs = 6, so backbone/block fine-tuning starts at epoch 7.
UNFREEZE_EPOCH = 7

# If you want a vertical marker for the final selected ResNet34 epoch,
# enter it here (e.g., 18). Use None to omit.
RESNET34_SELECTED_EPOCH = 18

DPI = 300
FIG_WIDTH = 13.33


# ============================================================
# HELPERS
# ============================================================

REQUIRED_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_pr_auc",
    "F1_best",
]

def load_all_metrics():
    data = {}
    missing = []

    for model, folder in MODEL_DIRS.items():
        csv_path = folder / "epoch_metrics.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue

        df = pd.read_csv(csv_path)

        absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if absent:
            raise ValueError(
                f"{csv_path} is missing required columns: {absent}"
            )

        df = df.sort_values("epoch").reset_index(drop=True)
        data[model] = df

    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            "The following epoch_metrics.csv files were not found:\n" + msg
        )

    return data


def add_unfreeze_marker(ax, label=False):
    ax.axvline(
        UNFREEZE_EPOCH,
        linestyle=":",
        linewidth=1.25,
        alpha=0.75,
    )
    if label:
        ymax = ax.get_ylim()[1]
        ax.text(
            UNFREEZE_EPOCH + 0.25,
            ymax * 0.96,
            "Backbone/block\nfine-tuning begins",
            va="top",
            ha="left",
            fontsize=8.5,
        )


def finish_axis(ax, xlabel=True):
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    if xlabel:
        ax.set_xlabel("Epoch")
    ax.grid(alpha=0.18, linewidth=0.6)


def save_figure(fig, stem):
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def make_summary(data):
    rows = []

    for model, df in data.items():
        i_loss = df["val_loss"].idxmin()
        i_f1 = df["F1_best"].idxmax()
        i_pr = df["val_pr_auc"].idxmax()

        rows.append({
            "model": model,
            "epoch_min_val_loss": int(df.loc[i_loss, "epoch"]),
            "min_val_loss": float(df.loc[i_loss, "val_loss"]),
            "epoch_max_val_chain_F1": int(df.loc[i_f1, "epoch"]),
            "max_val_chain_F1": float(df.loc[i_f1, "F1_best"]),
            "epoch_max_val_PR_AUC": int(df.loc[i_pr, "epoch"]),
            "max_val_PR_AUC": float(df.loc[i_pr, "val_pr_auc"]),
            "final_epoch_train_loss": float(df.iloc[-1]["train_loss"]),
            "final_epoch_val_loss": float(df.iloc[-1]["val_loss"]),
            "final_epoch_val_chain_F1": float(df.iloc[-1]["F1_best"]),
            "final_epoch_val_PR_AUC": float(df.iloc[-1]["val_pr_auc"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "learning_curve_summary.csv", index=False)
    return out


# ============================================================
# OPTION 1
# 2 x 3 small multiples: training and validation loss
# ============================================================

def make_option1(data):
    models_order = [
        "ResNet18", "ResNet34", "ResNet50",
        "ResNet101", "VGG16", "VGG19"
    ]

    # Use one common y-axis range across all six architectures so that
    # the magnitude of training/validation loss is directly comparable.
    all_loss_values = []
    for model in models_order:
        df = data[model]
        all_loss_values.extend(df["train_loss"].to_numpy())
        all_loss_values.extend(df["val_loss"].to_numpy())

    max_loss = float(np.nanmax(all_loss_values))
    # Round upward to a clean 0.05 increment and add a little headroom.
    loss_ymax = max(0.05, np.ceil((max_loss * 1.05) / 0.05) * 0.05)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIG_WIDTH, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for panel_idx, (ax, model) in enumerate(zip(axes.flat, models_order)):
        df = data[model]
        row = panel_idx // 3
        col = panel_idx % 3

        ax.plot(
            df["epoch"], df["train_loss"],
            linewidth=1.8, label="Training"
        )
        ax.plot(
            df["epoch"], df["val_loss"],
            linewidth=1.8, label="Validation"
        )

        ax.set_ylim(0, loss_ymax)

        add_unfreeze_marker(ax, label=(panel_idx == 0))
        finish_axis(ax, xlabel=(row == 1))

        ax.set_title(model, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Binary cross-entropy loss")
        else:
            ax.set_ylabel("")

        panel_letter = chr(ord("a") + panel_idx)
        ax.text(
            0.01, 0.98, f"({panel_letter})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold"
        )

        if panel_idx == 0:
            ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "Training and validation loss across candidate CNN architectures",
        fontsize=14,
        fontweight="bold",
    )

    return save_figure(fig, "option1_loss_small_multiples_2x3")


# ============================================================
# OPTION 2
# Comparative architecture figure:
# Rows = validation loss / chain F1 / PR-AUC
# Columns = ResNet family / VGG family
# ============================================================

def make_option2(data):
    resnets = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"]
    vggs = ["VGG16", "VGG19"]

    fig, axes = plt.subplots(
        3, 2,
        figsize=(FIG_WIDTH, 9.2),
        sharex=True,
        constrained_layout=True,
    )

    metrics = [
        ("val_loss", "Validation loss"),
        ("F1_best", "Validation chain F1 score"),
        ("val_pr_auc", "Validation PR-AUC"),
    ]

    families = [
        ("ResNet architectures", resnets),
        ("VGG architectures", vggs),
    ]

    panel = 0
    for row, (metric, ylabel) in enumerate(metrics):
        for col, (family_title, model_list) in enumerate(families):
            ax = axes[row, col]

            for model in model_list:
                df = data[model]
                ax.plot(
                    df["epoch"], df[metric],
                    linewidth=1.7,
                    label=model
                )

            add_unfreeze_marker(ax, label=(row == 0 and col == 0))
            finish_axis(ax, xlabel=(row == 2))

            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(family_title, fontsize=12, fontweight="bold")

            if metric in ("F1_best", "val_pr_auc"):
                ax.set_ylim(0, 1.02)

            panel_letter = chr(ord("a") + panel)
            ax.text(
                0.01, 0.98, f"({panel_letter})",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=11, fontweight="bold"
            )
            panel += 1

            ax.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(
        "Held-out validation behavior across candidate CNN architectures",
        fontsize=14,
        fontweight="bold",
    )

    return save_figure(fig, "option2_validation_metrics_by_family_3x2")


# ============================================================
# OPTION 3
# Focused final-model diagnostics for ResNet34
# ============================================================

def make_option3(data):
    df = data["ResNet34"]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIG_WIDTH, 4.15),
        constrained_layout=True,
    )

    # (a) Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], linewidth=1.9, label="Training")
    ax.plot(df["epoch"], df["val_loss"], linewidth=1.9, label="Validation")
    add_unfreeze_marker(ax, label=True)
    if RESNET34_SELECTED_EPOCH is not None:
        ax.axvline(
            RESNET34_SELECTED_EPOCH,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
        )
    finish_axis(ax)
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("(a) Training and validation loss", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    # (b) Chain F1
    ax = axes[1]
    ax.plot(df["epoch"], df["F1_best"], linewidth=1.9)
    add_unfreeze_marker(ax)
    if RESNET34_SELECTED_EPOCH is not None:
        ax.axvline(
            RESNET34_SELECTED_EPOCH,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
        )
    finish_axis(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Chain-class F1 score")
    ax.set_title("(b) Validation chain F1", fontsize=11.5, fontweight="bold")

    # (c) PR-AUC
    ax = axes[2]
    ax.plot(df["epoch"], df["val_pr_auc"], linewidth=1.9)
    add_unfreeze_marker(ax)
    if RESNET34_SELECTED_EPOCH is not None:
        ax.axvline(
            RESNET34_SELECTED_EPOCH,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label=f"Selected epoch ({RESNET34_SELECTED_EPOCH})",
        )
    finish_axis(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("PR-AUC")
    ax.set_title("(c) Validation PR-AUC", fontsize=11.5, fontweight="bold")
    if RESNET34_SELECTED_EPOCH is not None:
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(
        "ResNet34 convergence and held-out validation performance",
        fontsize=14,
        fontweight="bold",
    )

    return save_figure(fig, "option3_selected_model_diagnostics_resnet34")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_all_metrics()
    summary = make_summary(data)

    outputs = []
    outputs.extend(make_option1(data))
    outputs.extend(make_option2(data))
    outputs.extend(make_option3(data))

    print("\nFinished.")
    print(f"Output directory:\n  {OUTPUT_DIR}\n")
    print("Summary:")
    print(summary.to_string(index=False))
    print("\nFigures:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
