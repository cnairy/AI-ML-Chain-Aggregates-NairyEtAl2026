#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RF + XGBoost baselines for CPI chain aggregates classification (JTECH-ready)

This version (per your requests):
- NO holdout test split. The ONLY test is the UNSEEN dataset.
- Uses a single 80/20 TRAIN/VAL split from the labeled TRAIN pool.
- Hyperparameter tuning (optional) is performed on TRAIN only via CV.
- Thresholds are chosen by maximizing F1 on VAL.
- Evaluates final performance on UNSEEN using VAL-derived thresholds.
- Excludes ANY particles cut off from frame border (pct_touch == 0 only).
- Exports particle-level UNSEEN predictions/descriptors for cross-machine CNN comparison.
- Plots on VAL and UNSEEN:
    * Confusion matrices (RF vs XGB side-by-side, white -> light blue colormap)
    * Prediction probability histograms (normalized, density=True)
    * ROC curves
    * Precision-Recall curves
    * Calibration curves

Author: Christian Nairy (adapted/updated)

Script name: RF_XGBoost_Combo_JTECH_w_hyperparams-no-holdout.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, roc_auc_score, average_precision_score, accuracy_score
from scipy.stats import randint, uniform

import xgboost as xgb


# =========================
#        CONFIG
# =========================

# TRAIN files (labeled by your chain file)
TRAIN_TOTAL_FILE = "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/Filtered_Final_20250626/cpi.total-particle.properties.filtered.20250626.nasa"
TRAIN_CHAIN_FILE = "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/F-T2chain_merged_20250627/cpi.FT2chain.properties.filtered.20250626.nasa"

# UNSEEN Jan 17 2022 directory and files
UNSEEN_DIR = "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/xgb_trial_20250630/"
UNSEEN_TOTAL_BASENAME = "22_01_17_10_01_49.cpi.total-particle.properties.filtered.raw"
UNSEEN_CHAIN_BASENAME = "22_01_17_10_03_37.cpi.T2chain.properties.filtered.raw"

UNSEEN_TOTAL_FILE = os.path.join(UNSEEN_DIR, UNSEEN_TOTAL_BASENAME)
UNSEEN_CHAIN_FILE = os.path.join(UNSEEN_DIR, UNSEEN_CHAIN_BASENAME)

# Dedicated exports for the JTECH RF/XGBoost/CNN head-to-head comparison.
# Copy RF_XGB_unseen_Jan17_2022_particles.csv to littlestorm after this script finishes.
UNSEEN_DATE = "2022-01-17"
EXPORT_DIR = os.path.join(UNSEEN_DIR, "JTECH_model_comparison")
EXPORT_PARTICLE_CSV = os.path.join(EXPORT_DIR, "RF_XGB_unseen_Jan17_2022_particles.csv")
EXPORT_METRICS_CSV = os.path.join(EXPORT_DIR, "RF_XGB_unseen_Jan17_2022_metrics.csv")

# Features you want to use (canonical names)
CANON_FEATURES = [
    "dmax", "circularity", "area_ratio", "complexity",
    "curl", "solidity", "compactness", "fine_detail"
]

# QA/QC filters (applied to both total and chain tables before labeling)
# Exclude ANY particle touching the frame border (cut off)
MAX_PCT_TOUCH = 0.0
MAX_CURL = 9999.0
MAX_COMPACTNESS = 9999.0

# Split: 80/20 train/val (from TRAIN pool)
VAL_SIZE = 0.20
RANDOM_STATE = 1

# -------------------------
# Hyperparameter tuning switches
# -------------------------
DO_TUNE = True

# How much to search (increase if you want, but start small)
N_ITER_RF = 30
N_ITER_XGB = 40
CV_FOLDS = 3

# Target metric for tuning (Chain class F1)
F1_CHAIN_SCORER = make_scorer(f1_score, pos_label=1)

# -------------------------
# Default RF hyperparameters (used if DO_TUNE=False)
# -------------------------
RF_PARAMS = dict(
    n_estimators=400,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    criterion="gini",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# -------------------------
# Default XGB hyperparameters (used if DO_TUNE=False)
#   No early_stopping_rounds here to avoid version mismatch.
# -------------------------
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_weight=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

# Plot styling
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


# =========================
#   ADPAA RAW FILE READER
# =========================

def _find_table_header_idx(lines):
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("Time") and ("center_x" in s or "dmax" in s or "img_num" in s):
            return i
    return None


def read_adpaa_table(filepath, verbose=True):
    if verbose:
        print(f"\nReading: {filepath}")

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    header_idx = _find_table_header_idx(lines)
    if header_idx is None:
        raise ValueError(
            f"Could not find table header line in {filepath}. "
            "Expected a line starting with 'Time' containing 'center_x' or 'dmax' or 'img_num'."
        )

    df = pd.read_csv(
        filepath,
        skiprows=header_idx,
        sep=r"\s+",
        engine="python",
        on_bad_lines="skip"
    )

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).copy()

    rename_map = {
        "pct_touching": "pct_touch",
        "pct_touch": "pct_touch",
        "aspe_ratio": "aspect_ratio",
        "frac_dim": "fractal_dimension"
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    for c in df.columns:
        if c == "Time":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(axis=1, how="all")

    if verbose:
        cols_show = list(df.columns[:60])
        print(f"Detected columns ({len(df.columns)}):")
        print(cols_show)

    return df


def apply_qc_filters(df, verbose=False, label=""):
    """
    Applies QC filters to remove suspect data, including removing any particles
    touching the image frame border (cut off).

    Border-touch logic:
      - With MAX_PCT_TOUCH = 0.0, keep only values extremely close to 0.
      - Uses <= so that MAX_PCT_TOUCH=0.0 keeps pct_touch==0.
    """
    df = df.copy()
    n0 = len(df)

    eps = 1e-12
    border_cols = []
    for c in ["pct_touch", "pct_touching", "pct_border_touch", "border_touch", "touching_border"]:
        if c in df.columns:
            border_cols.append(c)

    if len(border_cols) > 0:
        c = border_cols[0]
        if c == "touching_border" and df[c].dropna().isin([0, 1, True, False]).all():
            df = df[df[c].astype(int) == 0]
        else:
            vals = pd.to_numeric(df[c], errors="coerce").fillna(np.inf)
            df = df[vals <= (MAX_PCT_TOUCH + eps)]
    else:
        if verbose:
            print("WARNING: No border-touch column found. Cut-off particle removal NOT applied.")

    if "curl" in df.columns:
        vals = pd.to_numeric(df["curl"], errors="coerce").fillna(np.inf)
        df = df[vals < MAX_CURL]

    if "compactness" in df.columns:
        vals = pd.to_numeric(df["compactness"], errors="coerce").fillna(np.inf)
        df = df[vals < MAX_COMPACTNESS]

    if verbose:
        n1 = len(df)
        tag = f"{label} " if label else ""
        print(f"{tag}QC: kept {n1}/{n0} rows (removed {n0 - n1}).")
        if len(border_cols) > 0:
            print(f"  Border-touch column used: {border_cols[0]} with MAX_PCT_TOUCH={MAX_PCT_TOUCH}")

    return df


def build_labeled_from_total_and_chain(total_df, chain_df, verbose=False):
    total = total_df.copy().reset_index(drop=True)
    chain = chain_df.copy().reset_index(drop=True)

    key_cols = ["Time"]
    if ("img_num" in total.columns) and ("img_num" in chain.columns):
        key_cols = ["Time", "img_num"]
    else:
        if verbose:
            print("WARNING: 'img_num' not found in both files. Label merge uses Time only (possible label noise).")

    chain_keys = chain[key_cols].drop_duplicates()
    merged = total.merge(chain_keys, on=key_cols, how="left", indicator=True)
    merged["chain"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns=["_merge"])

    if verbose:
        print(f"Labeled dataset size: {len(merged)}")
        frac = merged["chain"].mean() if len(merged) else np.nan
        print(f"Chain fraction: {frac:.6f}")

    return merged


def build_feature_frame(labeled_df):
    df = labeled_df.copy()
    existing = [c for c in CANON_FEATURES if c in df.columns]
    if len(existing) == 0:
        raise ValueError(
            "None of the canonical features were found.\n"
            f"Expected one or more of: {CANON_FEATURES}\n"
            f"Actual columns (first 60): {list(df.columns[:60])}\n"
            "Fix: check column names in the file."
        )
    df = df.dropna(subset=existing + ["chain"]).copy()
    X = df[existing].astype(float)
    y = df["chain"].astype(int)
    return X, y, existing


# =========================
#   THRESHOLD SELECTION
# =========================

def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(precision[i]), float(recall[i]), float(f1[i])


# =========================
#   EVALUATION + PLOTTING
# =========================

def eval_at_threshold(name, y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name}:")
    print(classification_report(y_true, y_pred, target_names=["Not Chain", "Chain"]))
    return y_pred, cm


def _white_to_lightblue_cmap():
    return LinearSegmentedColormap.from_list(
        "white_lightblue",
        ["#ffffff", "#dbe9f6", "#9ec5e8", "#4a90d9"]
    )


def plot_confusions(cm_rf, cm_xgb, title, normalize=False):
    """
    Side-by-side confusion matrices for RF and XGB with a white->blue colormap.
    Colorbar is placed to the far right of both panels.
    """
    cmap = LinearSegmentedColormap.from_list(
        "white_blue",
        ["#ffffff", "#dbe9f6", "#9ec5e8", "#4a90d9", "#08519c"]
    )

    cms = [cm_rf.astype(float), cm_xgb.astype(float)]
    if normalize:
        cms_norm = []
        for cm in cms:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cms_norm.append(cm / row_sums)
        cms = cms_norm
        fmt = "{:.2f}"
    else:
        fmt = "{:d}"

    vmin = 0.0
    vmax = max(float(np.nanmax(cms[0])), float(np.nanmax(cms[1])))
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    # leave room on the right for the colorbar
    fig.subplots_adjust(right=0.88, wspace=0.35)

    im = None
    for ax, cm, t, cm_raw in zip(
        axes, cms, ["Random Forest", "XGBoost"], [cm_rf, cm_xgb]
    ):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{t} Confusion ({title})")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-chains", "Chains"])
        ax.set_yticklabels(["Non-chains", "Chains"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        for i in range(2):
            for j in range(2):
                txt = fmt.format(cm[i, j]) if normalize else fmt.format(int(cm_raw[i, j]))
                ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=13)

    # dedicated colorbar axis on the far right
    cax = fig.add_axes([0.90, 0.18, 0.015, 0.68])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Count" if not normalize else "Fraction")

    plt.show()


def plot_prediction_histograms(y_true, prob_rf, prob_xgb, title, thr_rf=None, thr_xgb=None, bins=30):
    """
    Predicted probability histograms for RF and XGB, split by true class,
    normalized as probability density (density=True) so shapes are comparable.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_rf = np.asarray(prob_rf).astype(float)
    prob_xgb = np.asarray(prob_xgb).astype(float)

    mask0 = (y_true == 0)
    mask1 = (y_true == 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200, sharex=True, sharey=True)

    axes[0].hist(prob_rf[mask0], bins=bins, range=(0, 1), alpha=0.6,
                 label="True: Not Chain", density=True)
    axes[0].hist(prob_rf[mask1], bins=bins, range=(0, 1), alpha=0.6,
                 label="True: Chain", density=True)
    if thr_rf is not None:
        axes[0].axvline(float(thr_rf), linestyle="--", linewidth=2,
                        label=f"Threshold = {float(thr_rf):.3f}")
    axes[0].set_title(f"Random Forest Probabilities ({title})")
    axes[0].set_xlabel("Predicted P(chain)")
    axes[0].set_ylabel("Probability density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].hist(prob_xgb[mask0], bins=bins, range=(0, 1), alpha=0.6,
                 label="True: Not Chain", density=True)
    axes[1].hist(prob_xgb[mask1], bins=bins, range=(0, 1), alpha=0.6,
                 label="True: Chain", density=True)
    if thr_xgb is not None:
        axes[1].axvline(float(thr_xgb), linestyle="--", linewidth=2,
                        label=f"Threshold = {float(thr_xgb):.3f}")
    axes[1].set_title(f"XGBoost Probabilities ({title})")
    axes[1].set_xlabel("Predicted P(chain)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_roc_pr_calibration(y_true, prob_rf, prob_xgb, title):
    fpr_rf, tpr_rf, _ = roc_curve(y_true, prob_rf)
    fpr_x, tpr_x, _ = roc_curve(y_true, prob_xgb)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_x = auc(fpr_x, tpr_x)

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(fpr_x, tpr_x, label=f"XGBoost (AUC={auc_x:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.title(f"ROC Comparison ({title})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    p_rf, r_rf, _ = precision_recall_curve(y_true, prob_rf)
    p_x, r_x, _ = precision_recall_curve(y_true, prob_xgb)
    ap_rf = average_precision_score(y_true, prob_rf)
    ap_x = average_precision_score(y_true, prob_xgb)

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(r_x, p_x, label=f"XGBoost (AP={ap_x:.3f})")
    plt.plot(r_rf, p_rf, label=f"Random Forest (AP={ap_rf:.3f})")
    plt.title(f"Precision-Recall Comparison ({title})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    prob_true_x, prob_pred_x = calibration_curve(y_true, prob_xgb, n_bins=10)
    prob_true_rf, prob_pred_rf = calibration_curve(y_true, prob_rf, n_bins=10)

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    plt.plot(prob_pred_x, prob_true_x, marker="x", label="XGBoost")
    plt.plot(prob_pred_rf, prob_true_rf, marker="o", label="Random Forest")
    plt.title(f"Calibration Curves ({title})")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# =========================
#   TUNING HELPERS
# =========================

def tune_random_forest(X_tr, y_tr):
    base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=False
    )

    param_dist = {
        "n_estimators": randint(200, 1200),
        "max_depth": randint(4, 30),
        "min_samples_split": randint(2, 30),
        "min_samples_leaf": randint(1, 15),
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced", "balanced_subsample"]
    }

    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=N_ITER_RF,
        scoring=F1_CHAIN_SCORER,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rs.fit(X_tr, y_tr)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


def tune_xgboost(X_tr, y_tr):
    pos = int(np.sum(y_tr == 1))
    neg = int(np.sum(y_tr == 0))
    spw = (neg / max(pos, 1))

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=spw
    )

    param_dist = {
        "n_estimators": randint(300, 2000),
        "learning_rate": uniform(0.01, 0.19),
        "max_depth": randint(2, 12),
        "min_child_weight": uniform(0.5, 8.0),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "gamma": uniform(0.0, 1.0),
        "reg_alpha": uniform(0.0, 2.0),
        "reg_lambda": uniform(0.5, 6.0)
    }

    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=N_ITER_XGB,
        scoring=F1_CHAIN_SCORER,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rs.fit(X_tr, y_tr)
    return rs.best_estimator_, rs.best_params_, rs.best_score_


# =========================
#   JTECH COMPARISON EXPORT
# =========================

def seconds_to_hhmmss_mmm(seconds):
    """Convert seconds from midnight to HH:MM:SS.mmm for cross-machine QA."""
    try:
        if pd.isna(seconds):
            return ""
        total_ms = int(round(float(seconds) * 1000.0))
    except Exception:
        return ""
    total_ms %= 24 * 3600 * 1000
    hh = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    mm = rem // 60_000
    rem %= 60_000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def canonical_image_number(values):
    """Create a stable integer-like image-number key without leading zeros."""
    out = pd.to_numeric(pd.Series(values), errors="coerce").round().astype("Int64")
    return out.astype("string")


def binary_metrics_row(method, y_true, y_prob, y_pred, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Method": method,
        "Threshold": float(threshold),
        "N": int(len(y_true)),
        "N_Chain_Manual": int(np.sum(y_true == 1)),
        "N_NonChain_Manual": int(np.sum(y_true == 0)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Chain": precision_score(y_true, y_pred, zero_division=0),
        "Recall_Chain": recall_score(y_true, y_pred, zero_division=0),
        "F1_Chain": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "ROC_AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "PR_AUC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
    }


def export_unseen_comparison_csvs(unseen_labeled, X_u, y_u,
                                   prob_rf_u, pred_rf_u, thr_rf,
                                   prob_xgb_u, pred_xgb_u, thr_xgb,
                                   used_features):
    """
    Export one row per ALL-IN unseen particle used by both tree classifiers.

    The local CSV is intentionally self-contained: it carries the manual label,
    descriptor values, model probabilities/predictions, and stable matching keys.
    After copying it to littlestorm, it can be merged with the CNN prediction CSV
    on Date + Image_Number_Key (with time retained as a QA field).
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    idx = X_u.index
    meta = unseen_labeled.loc[idx].copy()

    out = pd.DataFrame(index=idx)
    out["Date"] = UNSEEN_DATE
    out["Time_seconds"] = pd.to_numeric(meta["Time"], errors="coerce").values
    out["Time_UTC"] = [seconds_to_hhmmss_mmm(v) for v in out["Time_seconds"]]

    if "img_num" not in meta.columns:
        raise ValueError("UNSEEN table does not contain img_num; cannot build a reliable cross-machine key.")

    out["Image_Number_Raw"] = meta["img_num"].values
    out["Image_Number_Key"] = canonical_image_number(meta["img_num"].values).values
    out["Manual_Label"] = np.asarray(y_u).astype(int)

    for feature in used_features:
        out[feature] = X_u[feature].values

    out["RF_Probability"] = np.asarray(prob_rf_u, dtype=float)
    out["RF_Predicted_Label"] = np.asarray(pred_rf_u, dtype=int)
    out["RF_Threshold"] = float(thr_rf)
    out["XGB_Probability"] = np.asarray(prob_xgb_u, dtype=float)
    out["XGB_Predicted_Label"] = np.asarray(pred_xgb_u, dtype=int)
    out["XGB_Threshold"] = float(thr_xgb)

    # Useful matching diagnostic. Ideally Date + Image_Number_Key is unique.
    out["Duplicate_Image_Key"] = out.duplicated(["Date", "Image_Number_Key"], keep=False)

    ordered = [
        "Date", "Time_seconds", "Time_UTC", "Image_Number_Raw", "Image_Number_Key",
        "Duplicate_Image_Key", "Manual_Label",
    ] + list(used_features) + [
        "RF_Probability", "RF_Predicted_Label", "RF_Threshold",
        "XGB_Probability", "XGB_Predicted_Label", "XGB_Threshold",
    ]
    out = out[ordered].reset_index(drop=True)
    out.to_csv(EXPORT_PARTICLE_CSV, index=False)

    metrics = pd.DataFrame([
        binary_metrics_row("Random Forest", y_u, prob_rf_u, pred_rf_u, thr_rf),
        binary_metrics_row("XGBoost", y_u, prob_xgb_u, pred_xgb_u, thr_xgb),
    ])
    metrics.to_csv(EXPORT_METRICS_CSV, index=False)

    print("\n================ JTECH COMPARISON EXPORTS ================")
    print(f"Particle-level RF/XGB CSV: {EXPORT_PARTICLE_CSV}")
    print(f"Tree-model metrics CSV:    {EXPORT_METRICS_CSV}")
    print(f"Rows exported: {len(out)}")
    print(f"Duplicate Date+Image_Number_Key rows: {int(out['Duplicate_Image_Key'].sum())}")
    print("Copy the particle-level CSV to littlestorm for the exact common-particle merge with CNN output.")


# =========================
#           MAIN
# =========================

def main():
    print("\nReading TRAIN data...")

    train_total_raw = read_adpaa_table(TRAIN_TOTAL_FILE, verbose=False)
    train_chain_raw = read_adpaa_table(TRAIN_CHAIN_FILE, verbose=False)

    # Apply QC, including "no cut-off particles"
    train_total_raw = apply_qc_filters(train_total_raw, verbose=True, label="TRAIN TOTAL")
    train_chain_raw = apply_qc_filters(train_chain_raw, verbose=True, label="TRAIN CHAIN")

    train_labeled = build_labeled_from_total_and_chain(train_total_raw, train_chain_raw, verbose=True)
    X_all, y_all, used_features = build_feature_frame(train_labeled)

    print(f"\nTRAIN labeled dataset size: {len(y_all)}")
    print(f"TRAIN chain fraction: {y_all.mean():.6f}")
    print(f"Using features: {used_features}")

    # -------------------------
    # Split: 80/20 train/val
    # -------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    print("\nSplit summary (from TRAIN pool):")
    print(f"  TRAIN: {len(y_tr)}  chains={int((y_tr==1).sum())}  prev={y_tr.mean():.6f}")
    print(f"  VAL:   {len(y_val)}  chains={int((y_val==1).sum())}  prev={y_val.mean():.6f}")

    # -------------------------
    # Train or tune models (TRAIN only)
    # -------------------------
    if DO_TUNE:
        print("\nTUNING Random Forest on TRAIN split only (CV)...")
        rf, rf_best_params, rf_best_cv = tune_random_forest(X_tr, y_tr)
        print("\nRF best CV F1(chain): {:.4f}".format(rf_best_cv))
        print("RF best params:")
        print(rf_best_params)

        print("\nTUNING XGBoost on TRAIN split only (CV)...")
        xgb_clf, xgb_best_params, xgb_best_cv = tune_xgboost(X_tr, y_tr)
        print("\nXGB best CV F1(chain): {:.4f}".format(xgb_best_cv))
        print("XGB best params:")
        print(xgb_best_params)
    else:
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_tr, y_tr)

        pos = int(np.sum(y_tr == 1))
        neg = int(np.sum(y_tr == 0))
        spw = (neg / max(pos, 1))

        xgb_clf = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        xgb_clf.fit(X_tr, y_tr)

    # If tuned, RandomizedSearchCV already fit best_estimator_ on X_tr,
    # but we refit explicitly for safety.
    if DO_TUNE:
        rf.fit(X_tr, y_tr)
        xgb_clf.fit(X_tr, y_tr)

    # -------------------------
    # Threshold selection on VAL
    # -------------------------
    prob_rf_val = rf.predict_proba(X_val)[:, 1]
    thr_rf, p_rf, r_rf, f1_rf = best_f1_threshold(y_val.values, prob_rf_val)

    prob_xgb_val = xgb_clf.predict_proba(X_val)[:, 1]
    thr_xgb, p_xgb, r_xgb, f1_xgb = best_f1_threshold(y_val.values, prob_xgb_val)

    print("\nChosen thresholds (maximize F1 on VAL):")
    print(f"  XGBoost threshold:       {thr_xgb:.4f} (P={p_xgb:.3f}, R={r_xgb:.3f}, F1={f1_xgb:.3f})")
    print(f"  Random Forest threshold: {thr_rf:.4f} (P={p_rf:.3f}, R={r_rf:.3f}, F1={f1_rf:.3f})")

    # -------------------------
    # VAL evaluation + plots
    # -------------------------
    print("\n================ VALIDATION RESULTS (from TRAIN pool) ================")

    _, cm_rf_val = eval_at_threshold("RANDOM FOREST (VAL)", y_val, prob_rf_val, thr_rf)
    _, cm_xgb_val = eval_at_threshold("XGBOOST (VAL)", y_val, prob_xgb_val, thr_xgb)

    plot_confusions(cm_rf_val, cm_xgb_val, title="VAL", normalize=False)
    plot_prediction_histograms(y_val.values, prob_rf_val, prob_xgb_val,
                               title="VAL", thr_rf=thr_rf, thr_xgb=thr_xgb, bins=30)
    plot_roc_pr_calibration(y_val.values, prob_rf_val, prob_xgb_val, title="VAL")

    # -------------------------
    # Read UNSEEN data
    # -------------------------
    print("\nReading UNSEEN data...")

    unseen_total_raw = read_adpaa_table(UNSEEN_TOTAL_FILE, verbose=False)
    unseen_chain_raw = read_adpaa_table(UNSEEN_CHAIN_FILE, verbose=False)

    unseen_total_raw = apply_qc_filters(unseen_total_raw, verbose=True, label="UNSEEN TOTAL")
    unseen_chain_raw = apply_qc_filters(unseen_chain_raw, verbose=True, label="UNSEEN CHAIN")

    unseen_labeled = build_labeled_from_total_and_chain(unseen_total_raw, unseen_chain_raw, verbose=True)
    X_u, y_u, used_features_u = build_feature_frame(unseen_labeled)

    print(f"\nUNSEEN labeled dataset size: {len(y_u)}")
    print(f"UNSEEN chain fraction (from chain file): {y_u.mean():.6f}")
    print(f"UNSEEN using features: {used_features_u}")

    missing_in_unseen = [c for c in used_features if c not in X_u.columns]
    if missing_in_unseen:
        raise ValueError(f"UNSEEN missing required features: {missing_in_unseen}")

    X_u = X_u[used_features].copy()

    # -------------------------
    # UNSEEN evaluation (use VAL-derived thresholds)
    # -------------------------
    print("\n================ UNSEEN RESULTS (Jan 17 2022) ================")

    prob_rf_u = rf.predict_proba(X_u)[:, 1]
    prob_xgb_u = xgb_clf.predict_proba(X_u)[:, 1]

    pred_rf_u, cm_rf_u = eval_at_threshold("RANDOM FOREST (UNSEEN)", y_u, prob_rf_u, thr_rf)
    pred_xgb_u, cm_xgb_u = eval_at_threshold("XGBOOST (UNSEEN)", y_u, prob_xgb_u, thr_xgb)

    export_unseen_comparison_csvs(
        unseen_labeled=unseen_labeled,
        X_u=X_u,
        y_u=y_u.values,
        prob_rf_u=prob_rf_u,
        pred_rf_u=pred_rf_u,
        thr_rf=thr_rf,
        prob_xgb_u=prob_xgb_u,
        pred_xgb_u=pred_xgb_u,
        thr_xgb=thr_xgb,
        used_features=used_features,
    )

    plot_confusions(cm_rf_u, cm_xgb_u, title="UNSEEN", normalize=False)
    plot_prediction_histograms(y_u.values, prob_rf_u, prob_xgb_u,
                               title="UNSEEN", thr_rf=thr_rf, thr_xgb=thr_xgb, bins=30)
    plot_roc_pr_calibration(y_u.values, prob_rf_u, prob_xgb_u, title="UNSEEN")

    print("\nDone.")


if __name__ == "__main__":
    main()