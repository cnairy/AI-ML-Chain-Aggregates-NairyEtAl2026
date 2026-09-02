#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JTECH Appendix A confusion-matrix plotting script

Purpose
-------
Create a clean 6-panel confusion-matrix figure for the descriptor-based
Random Forest and XGBoost classifiers under:
    1) ADASYN
    2) SMOTE
    3) No over- or under-sampling

This script reads the saved audit metrics CSV and plots ONLY the
independent-test confusion matrices.

Expected input
--------------
validation_and_independent_test_metrics.csv

Default location
----------------
/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/
xgb_trial_20250630/JTECH_model_comparison/Appendix_A_diagnostics/

Outputs
-------
Figure_A1_independent_test_confusion_matrices_clean.png
Figure_A1_independent_test_confusion_matrices_clean.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = (
    "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/"
    "xgb_trial_20250630/JTECH_model_comparison/Appendix_A_diagnostics/"
    "validation_and_independent_test_metrics.csv"
)

OUTPUT_DIR = os.path.dirname(INPUT_CSV)
PNG_OUT = os.path.join(OUTPUT_DIR, "Figure_A1_independent_test_confusion_matrices_clean.png")
PDF_OUT = os.path.join(OUTPUT_DIR, "Figure_A1_independent_test_confusion_matrices_clean.pdf")

DPI = 300

TREATMENT_ORDER = ["ADASYN", "SMOTE", "None"]
TREATMENT_LABELS = {
    "ADASYN": "ADASYN",
    "SMOTE": "SMOTE",
    "None": "No over- or under-sampling",
}
MODEL_ORDER = ["Random Forest", "XGBoost"]
MODEL_LABELS = {
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found:\n{INPUT_CSV}")

df = pd.read_csv(INPUT_CSV, keep_default_na=False)
df["Treatment"] = df["Treatment"].replace("", "None").fillna("None")
df = df[df["Dataset"] == "Independent test"].copy()

if df.empty:
    raise RuntimeError("No 'Independent test' rows found in the input CSV.")

fig, axes = plt.subplots(
    nrows=3, ncols=2,
    figsize=(8.3, 10.0),
    dpi=DPI,
    constrained_layout=True
)

vmax = int(df[["TN", "FP", "FN", "TP"]].to_numpy().max())
if vmax <= 0:
    vmax = 1

im = None

for i, treatment in enumerate(TREATMENT_ORDER):
    for j, model in enumerate(MODEL_ORDER):
        ax = axes[i, j]

        row = df[(df["Treatment"] == treatment) & (df["Model"] == model)]
        if row.empty:
            ax.axis("off")
            continue

        row = row.iloc[0]
        cm = np.array([
            [row["TN"], row["FP"]],
            [row["FN"], row["TP"]]
        ], dtype=float)

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-chain", "Chain"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-chain", "Chain"], fontsize=10)

        if i == 2:
            ax.set_xlabel("Predicted label", fontsize=11)
        else:
            ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("True label", fontsize=11)
        else:
            ax.set_ylabel("")

        title = f"{MODEL_LABELS[model]} — {TREATMENT_LABELS[treatment]}"
        ax.set_title(title, fontsize=11)

        for r in range(2):
            for c in range(2):
                val = int(cm[r, c])
                color = "white" if cm[r, c] > 0.45 * vmax else "black"
                ax.text(
                    c, r, f"{val}",
                    ha="center", va="center",
                    fontsize=11, color=color
                )

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.84, pad=0.02)
cbar.set_label("Count", fontsize=11)
cbar.ax.tick_params(labelsize=10)


fig.savefig(PNG_OUT, dpi=DPI, bbox_inches="tight")
fig.savefig(PDF_OUT, bbox_inches="tight")
plt.close(fig)

print("Done.")
print(f"PNG: {PNG_OUT}")
print(f"PDF: {PDF_OUT}")
