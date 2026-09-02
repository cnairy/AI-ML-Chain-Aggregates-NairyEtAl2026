#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JTECH Appendix A audit/reproducibility script for descriptor-based classifiers.

FIXED VERSION: Matplotlib tick-label calls are written in a backward-compatible
form so the Appendix figures do not fail on older Matplotlib installations.

Purpose
-------
Re-run the three descriptor-based class-imbalance treatments used in the manuscript:
    1) No over- or under-sampling
    2) ADASYN
    3) SMOTE

For both Random Forest and XGBoost, this script preserves the modeling logic used in the
existing JTECH scripts and writes the information needed to document Appendix A:

- dataset/QC counts
- original train/validation counts
- resampled training counts for ADASYN and SMOTE
- hyperparameter search spaces
- best hyperparameters from RandomizedSearchCV
- best cross-validated chain-class F1
- validation-selected probability thresholds
- validation precision/recall/F1 and confusion counts
- independent Jan 17 2022 test precision/recall/F1/accuracy/ROC-AUC/PR-AUC
- independent-test confusion counts
- a six-panel independent-test confusion-matrix figure suitable for Appendix A
- independent-test precision-recall curves for the three treatments

IMPORTANT
---------
- The 17 January 2022 data are NEVER used for tuning or threshold selection.
- ADASYN/SMOTE are applied only to training data and, during RandomizedSearchCV,
  safely inside each CV training fold through an imbalanced-learn Pipeline.
- Thresholds are selected only from the untouched validation subset by maximizing
  chain-class F1.
- This script does not alter any existing files; it writes to a new output directory.

Based on the user's existing JTECH no-resampling, ADASYN, and SMOTE scripts.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    make_scorer,
)

import xgboost as xgb

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

# Development/training files
TRAIN_TOTAL_FILE = (
    "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/"
    "Filtered_Final_20250626/cpi.total-particle.properties.filtered.20250626.nasa"
)
TRAIN_CHAIN_FILE = (
    "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/"
    "F-T2chain_merged_20250627/cpi.FT2chain.properties.filtered.20250626.nasa"
)

# Independent Jan 17 2022 test files
UNSEEN_DIR = (
    "/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/"
    "xgb_trial_20250630"
)
UNSEEN_TOTAL_BASENAME = "22_01_17_10_01_49.cpi.total-particle.properties.filtered.raw"
UNSEEN_CHAIN_BASENAME = "22_01_17_10_03_37.cpi.T2chain.properties.filtered.raw"

UNSEEN_TOTAL_FILE = os.path.join(UNSEEN_DIR, UNSEEN_TOTAL_BASENAME)
UNSEEN_CHAIN_FILE = os.path.join(UNSEEN_DIR, UNSEEN_CHAIN_BASENAME)

# All new outputs go here.
OUTPUT_DIR = os.path.join(
    UNSEEN_DIR,
    "JTECH_model_comparison",
    "Appendix_A_diagnostics"
)

CANON_FEATURES = [
    "dmax", "circularity", "area_ratio", "complexity",
    "curl", "solidity", "compactness", "fine_detail"
]

# Same QC logic as the existing scripts.
MAX_PCT_TOUCH = 0.0
MAX_CURL = 9999.0
MAX_COMPACTNESS = 9999.0

# Same development split and tuning controls.
VAL_SIZE = 0.20
RANDOM_STATE = 1
N_ITER_RF = 30
N_ITER_XGB = 40
CV_FOLDS = 3
F1_CHAIN_SCORER = make_scorer(f1_score, pos_label=1)

# Same oversampler settings as the separate scripts.
ADASYN_PARAMS = dict(
    sampling_strategy="auto",
    n_neighbors=5,
    random_state=RANDOM_STATE,
)
SMOTE_PARAMS = dict(
    sampling_strategy="auto",
    k_neighbors=5,
    random_state=RANDOM_STATE,
)

# Treatments to run. Turn any of these False if you only need one later.
RUN_NO_RESAMPLING = True
RUN_ADASYN = True
RUN_SMOTE = True

# Figure settings
FIG_DPI = 300


# =============================================================================
# DATA READING / QC
# =============================================================================

def _find_table_header_idx(lines):
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("Time") and ("center_x" in s or "dmax" in s or "img_num" in s):
            return i
    return None


def read_adpaa_table(filepath):
    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    header_idx = _find_table_header_idx(lines)
    if header_idx is None:
        raise ValueError(
            f"Could not find table header in {filepath}. Expected a line beginning "
            "with 'Time' and containing center_x, dmax, or img_num."
        )

    df = pd.read_csv(
        filepath,
        skiprows=header_idx,
        sep=r"\s+",
        engine="python",
        on_bad_lines="skip",
    )

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).copy()

    rename_map = {
        "pct_touching": "pct_touch",
        "aspe_ratio": "aspect_ratio",
        "frac_dim": "fractal_dimension",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    for c in df.columns:
        if c != "Time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(axis=1, how="all")


def apply_qc_filters_with_counts(df, label):
    out = df.copy()
    n_raw = len(out)

    border_col = None
    for c in [
        "pct_touch", "pct_touching", "pct_border_touch",
        "border_touch", "touching_border"
    ]:
        if c in out.columns:
            border_col = c
            break

    before_border = len(out)
    if border_col is not None:
        if (
            border_col == "touching_border"
            and out[border_col].dropna().isin([0, 1, True, False]).all()
        ):
            out = out[out[border_col].astype(int) == 0]
        else:
            vals = pd.to_numeric(out[border_col], errors="coerce").fillna(np.inf)
            out = out[vals <= (MAX_PCT_TOUCH + 1e-12)]
    after_border = len(out)

    before_curl = len(out)
    if "curl" in out.columns:
        vals = pd.to_numeric(out["curl"], errors="coerce").fillna(np.inf)
        out = out[vals < MAX_CURL]
    after_curl = len(out)

    before_compact = len(out)
    if "compactness" in out.columns:
        vals = pd.to_numeric(out["compactness"], errors="coerce").fillna(np.inf)
        out = out[vals < MAX_COMPACTNESS]
    after_compact = len(out)

    row = {
        "Dataset": label,
        "Rows_raw": n_raw,
        "Border_column": border_col if border_col is not None else "NOT FOUND",
        "Removed_border_touch": before_border - after_border,
        "Removed_curl_filter": before_curl - after_curl,
        "Removed_compactness_filter": before_compact - after_compact,
        "Rows_after_QC": len(out),
        "MAX_PCT_TOUCH": MAX_PCT_TOUCH,
        "MAX_CURL": MAX_CURL,
        "MAX_COMPACTNESS": MAX_COMPACTNESS,
    }
    return out, row


def build_labeled_from_total_and_chain(total_df, chain_df):
    total = total_df.copy().reset_index(drop=True)
    chain = chain_df.copy().reset_index(drop=True)

    key_cols = ["Time"]
    if "img_num" in total.columns and "img_num" in chain.columns:
        key_cols = ["Time", "img_num"]

    chain_keys = chain[key_cols].drop_duplicates()
    merged = total.merge(chain_keys, on=key_cols, how="left", indicator=True)
    merged["chain"] = (merged["_merge"] == "both").astype(int)
    return merged.drop(columns=["_merge"])


def build_feature_frame_with_counts(labeled_df, dataset_name):
    df = labeled_df.copy()
    missing_features = [c for c in CANON_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"{dataset_name}: missing required features: {missing_features}\n"
            f"Available columns: {list(df.columns)}"
        )

    n_before = len(df)
    n_chain_before = int((df["chain"] == 1).sum())

    complete = df.dropna(subset=CANON_FEATURES + ["chain"]).copy()

    n_after = len(complete)
    n_chain_after = int((complete["chain"] == 1).sum())

    count_row = {
        "Dataset": dataset_name,
        "Rows_labeled_before_complete_feature_filter": n_before,
        "Chains_before_complete_feature_filter": n_chain_before,
        "Nonchains_before_complete_feature_filter": n_before - n_chain_before,
        "Removed_missing_required_feature": n_before - n_after,
        "Rows_used_for_modeling": n_after,
        "Chains_used_for_modeling": n_chain_after,
        "Nonchains_used_for_modeling": n_after - n_chain_after,
    }

    X = complete[CANON_FEATURES].astype(float)
    y = complete["chain"].astype(int)
    return X, y, count_row


# =============================================================================
# SEARCH SPACES / MODEL BUILDING
# =============================================================================

def rf_search_space(resampling):
    class_weights = (
        [None, "balanced", "balanced_subsample"]
        if resampling == "None"
        else [None]
    )
    return {
        "n_estimators": "randint(200, 1200)",
        "max_depth": "randint(4, 30)",
        "min_samples_split": "randint(2, 30)",
        "min_samples_leaf": "randint(1, 15)",
        "max_features": "sqrt | log2 | None",
        "class_weight": " | ".join(str(x) for x in class_weights),
    }


def xgb_search_space():
    return {
        "n_estimators": "randint(300, 2000)",
        "learning_rate": "uniform(0.01, 0.19)",
        "max_depth": "randint(2, 12)",
        "min_child_weight": "uniform(0.5, 8.0)",
        "subsample": "uniform(0.5, 0.5)",
        "colsample_bytree": "uniform(0.5, 0.5)",
        "gamma": "uniform(0.0, 1.0)",
        "reg_alpha": "uniform(0.0, 2.0)",
        "reg_lambda": "uniform(0.5, 6.0)",
    }


def make_sampler(treatment):
    if treatment == "ADASYN":
        return ADASYN(**ADASYN_PARAMS)
    if treatment == "SMOTE":
        return SMOTE(**SMOTE_PARAMS)
    return None


def tune_random_forest(X_tr, y_tr, treatment):
    base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=False,
    )

    sampler = make_sampler(treatment)

    if sampler is None:
        estimator = base
        prefix = ""
        class_weights = [None, "balanced", "balanced_subsample"]
    else:
        estimator = Pipeline([("sampler", sampler), ("clf", base)])
        prefix = "clf__"
        class_weights = [None]

    param_dist = {
        f"{prefix}n_estimators": randint(200, 1200),
        f"{prefix}max_depth": randint(4, 30),
        f"{prefix}min_samples_split": randint(2, 30),
        f"{prefix}min_samples_leaf": randint(1, 15),
        f"{prefix}max_features": ["sqrt", "log2", None],
        f"{prefix}class_weight": class_weights,
    }

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=N_ITER_RF,
        scoring=F1_CHAIN_SCORER,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False,
    )
    rs.fit(X_tr, y_tr)
    return rs.best_estimator_, rs.best_params_, float(rs.best_score_)


def tune_xgboost(X_tr, y_tr, treatment):
    if treatment == "None":
        pos = int(np.sum(y_tr == 1))
        neg = int(np.sum(y_tr == 0))
        scale_pos_weight = neg / max(pos, 1)
    else:
        scale_pos_weight = 1.0

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )

    sampler = make_sampler(treatment)
    if sampler is None:
        estimator = base
        prefix = ""
    else:
        estimator = Pipeline([("sampler", sampler), ("clf", base)])
        prefix = "clf__"

    param_dist = {
        f"{prefix}n_estimators": randint(300, 2000),
        f"{prefix}learning_rate": uniform(0.01, 0.19),
        f"{prefix}max_depth": randint(2, 12),
        f"{prefix}min_child_weight": uniform(0.5, 8.0),
        f"{prefix}subsample": uniform(0.5, 0.5),
        f"{prefix}colsample_bytree": uniform(0.5, 0.5),
        f"{prefix}gamma": uniform(0.0, 1.0),
        f"{prefix}reg_alpha": uniform(0.0, 2.0),
        f"{prefix}reg_lambda": uniform(0.5, 6.0),
    }

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=N_ITER_XGB,
        scoring=F1_CHAIN_SCORER,
        cv=CV_FOLDS,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False,
    )
    rs.fit(X_tr, y_tr)
    return (
        rs.best_estimator_,
        rs.best_params_,
        float(rs.best_score_),
        float(scale_pos_weight),
    )


# =============================================================================
# THRESHOLD / METRICS
# =============================================================================

def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        raise ValueError("No candidate thresholds were produced.")

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    i = int(np.nanargmax(f1))
    return (
        float(thresholds[i]),
        float(precision[i]),
        float(recall[i]),
        float(f1[i]),
    )


def metric_row(treatment, model_name, dataset_name, y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Treatment": treatment,
        "Model": model_name,
        "Dataset": dataset_name,
        "Threshold": float(threshold),
        "N": int(len(y_true)),
        "N_Chain": int(np.sum(y_true == 1)),
        "N_Nonchain": int(np.sum(y_true == 0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_Chain": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_Chain": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_Chain": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "ROC_AUC": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "PR_AUC": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2 else np.nan
        ),
    }


def strip_pipeline_prefix(params):
    cleaned = {}
    for k, v in params.items():
        for prefix in ("clf__", "sampler__"):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        if isinstance(v, np.generic):
            v = v.item()
        cleaned[k] = v
    return cleaned


def resampled_training_count(X_tr, y_tr, treatment):
    if treatment == "None":
        return {
            "Treatment": treatment,
            "N_before": int(len(y_tr)),
            "Chains_before": int((y_tr == 1).sum()),
            "Nonchains_before": int((y_tr == 0).sum()),
            "N_after": int(len(y_tr)),
            "Chains_after": int((y_tr == 1).sum()),
            "Nonchains_after": int((y_tr == 0).sum()),
        }

    sampler = make_sampler(treatment)
    X_res, y_res = sampler.fit_resample(X_tr, y_tr)
    y_res = np.asarray(y_res).astype(int)

    return {
        "Treatment": treatment,
        "N_before": int(len(y_tr)),
        "Chains_before": int((np.asarray(y_tr) == 1).sum()),
        "Nonchains_before": int((np.asarray(y_tr) == 0).sum()),
        "N_after": int(len(y_res)),
        "Chains_after": int((y_res == 1).sum()),
        "Nonchains_after": int((y_res == 0).sum()),
    }


# =============================================================================
# FIGURES
# =============================================================================

def plot_unseen_confusion_grid(metrics_df):
    df = metrics_df[metrics_df["Dataset"] == "Independent test"].copy()
    treatments = ["None", "ADASYN", "SMOTE"]
    models = ["Random Forest", "XGBoost"]

    fig, axes = plt.subplots(
        3, 2, figsize=(8.5, 10.0), dpi=FIG_DPI,
        constrained_layout=True
    )

    vmax = int(df[["TN", "FP", "FN", "TP"]].to_numpy().max())

    for i, treatment in enumerate(treatments):
        for j, model in enumerate(models):
            ax = axes[i, j]
            row = df[
                (df["Treatment"] == treatment) &
                (df["Model"] == model)
            ]
            if row.empty:
                ax.axis("off")
                continue

            row = row.iloc[0]
            cm = np.array([
                [row["TN"], row["FP"]],
                [row["FN"], row["TP"]],
            ], dtype=float)

            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-chain", "Chain"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Non-chain", "Chain"])
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")

            treatment_label = (
                "No resampling" if treatment == "None" else treatment
            )
            ax.set_title(f"{treatment_label} — {model}")

            for r in range(2):
                for c in range(2):
                    ax.text(
                        c, r, f"{int(cm[r, c])}",
                        ha="center", va="center"
                    )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("Count")

    png = os.path.join(
        OUTPUT_DIR,
        "Figure_A1_independent_test_confusion_matrices.png"
    )
    pdf = os.path.join(
        OUTPUT_DIR,
        "Figure_A1_independent_test_confusion_matrices.pdf"
    )
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_unseen_pr_curves(prob_store, y_unseen):
    treatments = ["None", "ADASYN", "SMOTE"]
    models = ["Random Forest", "XGBoost"]

    fig, axes = plt.subplots(
        1, 3, figsize=(11.5, 3.8), dpi=FIG_DPI,
        sharex=True, sharey=True, constrained_layout=True
    )

    for ax, treatment in zip(axes, treatments):
        for model in models:
            key = (treatment, model)
            if key not in prob_store:
                continue
            probs = prob_store[key]
            precision, recall, _ = precision_recall_curve(y_unseen, probs)
            ap = average_precision_score(y_unseen, probs)
            ax.plot(recall, precision, label=f"{model} (PR-AUC={ap:.3f})")

        label = "No resampling" if treatment == "None" else treatment
        ax.set_title(label)
        ax.set_xlabel("Recall")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Precision")
    axes[-1].legend(fontsize=8, loc="lower left")

    png = os.path.join(
        OUTPUT_DIR,
        "Appendix_independent_test_precision_recall_curves.png"
    )
    pdf = os.path.join(
        OUTPUT_DIR,
        "Appendix_independent_test_precision_recall_curves.pdf"
    )
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n============================================================")
    print("JTECH APPENDIX A — DESCRIPTOR-BASED CLASSIFIER AUDIT")
    print("============================================================")
    print(f"Output directory:\n{OUTPUT_DIR}\n")

    # -------------------------------------------------------------------------
    # Read and QC development data
    # -------------------------------------------------------------------------
    train_total_raw = read_adpaa_table(TRAIN_TOTAL_FILE)
    train_chain_raw = read_adpaa_table(TRAIN_CHAIN_FILE)

    train_total_qc, qc_train_total = apply_qc_filters_with_counts(
        train_total_raw, "Development TOTAL"
    )
    train_chain_qc, qc_train_chain = apply_qc_filters_with_counts(
        train_chain_raw, "Development CHAIN"
    )

    train_labeled = build_labeled_from_total_and_chain(
        train_total_qc, train_chain_qc
    )
    X_all, y_all, feature_count_dev = build_feature_frame_with_counts(
        train_labeled, "Development"
    )

    # One shared split for all three treatments, exactly as in the existing scripts.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all,
        y_all,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    split_rows = [
        {
            "Subset": "Development complete-feature pool",
            "N": len(y_all),
            "Chains": int((y_all == 1).sum()),
            "Nonchains": int((y_all == 0).sum()),
        },
        {
            "Subset": "Training",
            "N": len(y_tr),
            "Chains": int((y_tr == 1).sum()),
            "Nonchains": int((y_tr == 0).sum()),
        },
        {
            "Subset": "Validation",
            "N": len(y_val),
            "Chains": int((y_val == 1).sum()),
            "Nonchains": int((y_val == 0).sum()),
        },
    ]

    # -------------------------------------------------------------------------
    # Read and QC independent test
    # -------------------------------------------------------------------------
    unseen_total_raw = read_adpaa_table(UNSEEN_TOTAL_FILE)
    unseen_chain_raw = read_adpaa_table(UNSEEN_CHAIN_FILE)

    unseen_total_qc, qc_unseen_total = apply_qc_filters_with_counts(
        unseen_total_raw, "Independent test TOTAL"
    )
    unseen_chain_qc, qc_unseen_chain = apply_qc_filters_with_counts(
        unseen_chain_raw, "Independent test CHAIN"
    )

    unseen_labeled = build_labeled_from_total_and_chain(
        unseen_total_qc, unseen_chain_qc
    )
    X_u, y_u, feature_count_unseen = build_feature_frame_with_counts(
        unseen_labeled, "Independent test"
    )
    X_u = X_u[CANON_FEATURES].copy()

    # -------------------------------------------------------------------------
    # Save dataset/QC counts
    # -------------------------------------------------------------------------
    pd.DataFrame([
        qc_train_total,
        qc_train_chain,
        qc_unseen_total,
        qc_unseen_chain,
    ]).to_csv(
        os.path.join(OUTPUT_DIR, "dataset_QC_counts.csv"),
        index=False
    )

    pd.DataFrame([
        feature_count_dev,
        feature_count_unseen,
    ]).to_csv(
        os.path.join(OUTPUT_DIR, "complete_feature_counts.csv"),
        index=False
    )

    pd.DataFrame(split_rows).to_csv(
        os.path.join(OUTPUT_DIR, "development_split_counts.csv"),
        index=False
    )

    # -------------------------------------------------------------------------
    # Save hyperparameter search spaces
    # -------------------------------------------------------------------------
    search_rows = []
    for treatment in ["None", "ADASYN", "SMOTE"]:
        for param, space in rf_search_space(treatment).items():
            search_rows.append({
                "Treatment": treatment,
                "Model": "Random Forest",
                "Parameter": param,
                "Search_space": space,
            })
        for param, space in xgb_search_space().items():
            search_rows.append({
                "Treatment": treatment,
                "Model": "XGBoost",
                "Parameter": param,
                "Search_space": space,
            })

    pd.DataFrame(search_rows).to_csv(
        os.path.join(OUTPUT_DIR, "hyperparameter_search_spaces.csv"),
        index=False
    )

    # -------------------------------------------------------------------------
    # Treatments
    # -------------------------------------------------------------------------
    treatments = []
    if RUN_NO_RESAMPLING:
        treatments.append("None")
    if RUN_ADASYN:
        treatments.append("ADASYN")
    if RUN_SMOTE:
        treatments.append("SMOTE")

    all_metrics = []
    best_param_rows = []
    tuning_summary_rows = []
    threshold_rows = []
    resample_rows = []
    prob_store = {}

    for treatment in treatments:
        print("\n" + "=" * 72)
        print(f"TREATMENT: {treatment}")
        print("=" * 72)

        # Report resampled full-training counts for documentation only.
        resample_rows.append(
            resampled_training_count(X_tr, y_tr, treatment)
        )

        # ---------------------- RF ----------------------
        print("\nTuning Random Forest...")
        rf, rf_best_params, rf_best_cv = tune_random_forest(
            X_tr, y_tr, treatment
        )

        rf_best_clean = strip_pipeline_prefix(rf_best_params)
        for param, value in rf_best_clean.items():
            best_param_rows.append({
                "Treatment": treatment,
                "Model": "Random Forest",
                "Parameter": param,
                "Selected_value": value,
            })

        tuning_summary_rows.append({
            "Treatment": treatment,
            "Model": "Random Forest",
            "CV_folds": CV_FOLDS,
            "Random_search_iterations": N_ITER_RF,
            "Best_CV_F1_chain": rf_best_cv,
            "Scale_pos_weight": np.nan,
        })

        prob_rf_val = rf.predict_proba(X_val)[:, 1]
        thr_rf, p_rf, r_rf, f1_rf = best_f1_threshold(
            y_val.values, prob_rf_val
        )

        threshold_rows.append({
            "Treatment": treatment,
            "Model": "Random Forest",
            "Validation_selected_threshold": thr_rf,
            "Validation_precision_at_selected_threshold": p_rf,
            "Validation_recall_at_selected_threshold": r_rf,
            "Validation_F1_at_selected_threshold": f1_rf,
        })

        all_metrics.append(
            metric_row(
                treatment, "Random Forest", "Validation",
                y_val.values, prob_rf_val, thr_rf
            )
        )

        prob_rf_u = rf.predict_proba(X_u)[:, 1]
        all_metrics.append(
            metric_row(
                treatment, "Random Forest", "Independent test",
                y_u.values, prob_rf_u, thr_rf
            )
        )
        prob_store[(treatment, "Random Forest")] = prob_rf_u

        # ---------------------- XGB ----------------------
        print("\nTuning XGBoost...")
        xgb_clf, xgb_best_params, xgb_best_cv, spw = tune_xgboost(
            X_tr, y_tr, treatment
        )

        xgb_best_clean = strip_pipeline_prefix(xgb_best_params)
        for param, value in xgb_best_clean.items():
            best_param_rows.append({
                "Treatment": treatment,
                "Model": "XGBoost",
                "Parameter": param,
                "Selected_value": value,
            })

        tuning_summary_rows.append({
            "Treatment": treatment,
            "Model": "XGBoost",
            "CV_folds": CV_FOLDS,
            "Random_search_iterations": N_ITER_XGB,
            "Best_CV_F1_chain": xgb_best_cv,
            "Scale_pos_weight": spw,
        })

        prob_xgb_val = xgb_clf.predict_proba(X_val)[:, 1]
        thr_xgb, p_xgb, r_xgb, f1_xgb = best_f1_threshold(
            y_val.values, prob_xgb_val
        )

        threshold_rows.append({
            "Treatment": treatment,
            "Model": "XGBoost",
            "Validation_selected_threshold": thr_xgb,
            "Validation_precision_at_selected_threshold": p_xgb,
            "Validation_recall_at_selected_threshold": r_xgb,
            "Validation_F1_at_selected_threshold": f1_xgb,
        })

        all_metrics.append(
            metric_row(
                treatment, "XGBoost", "Validation",
                y_val.values, prob_xgb_val, thr_xgb
            )
        )

        prob_xgb_u = xgb_clf.predict_proba(X_u)[:, 1]
        all_metrics.append(
            metric_row(
                treatment, "XGBoost", "Independent test",
                y_u.values, prob_xgb_u, thr_xgb
            )
        )
        prob_store[(treatment, "XGBoost")] = prob_xgb_u

    # -------------------------------------------------------------------------
    # Save model/tuning outputs
    # -------------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, "validation_and_independent_test_metrics.csv"),
        index=False
    )

    pd.DataFrame(best_param_rows).to_csv(
        os.path.join(OUTPUT_DIR, "selected_hyperparameters.csv"),
        index=False
    )

    pd.DataFrame(tuning_summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "tuning_summary.csv"),
        index=False
    )

    pd.DataFrame(threshold_rows).to_csv(
        os.path.join(OUTPUT_DIR, "validation_selected_thresholds.csv"),
        index=False
    )

    pd.DataFrame(resample_rows).to_csv(
        os.path.join(OUTPUT_DIR, "resampling_counts.csv"),
        index=False
    )

    # Appendix-ready wide summary table.
    independent = metrics_df[
        metrics_df["Dataset"] == "Independent test"
    ].copy()
    appendix_cols = [
        "Treatment", "Model", "Threshold",
        "Precision_Chain", "Recall_Chain", "F1_Chain",
        "Accuracy", "TN", "FP", "FN", "TP",
        "ROC_AUC", "PR_AUC"
    ]
    independent[appendix_cols].to_csv(
        os.path.join(OUTPUT_DIR, "Appendix_A_model_performance_summary.csv"),
        index=False
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    if len(independent) > 0:
        plot_unseen_confusion_grid(metrics_df)

    if prob_store:
        plot_unseen_pr_curves(prob_store, y_u.values)

    # -------------------------------------------------------------------------
    # Human-readable text summary
    # -------------------------------------------------------------------------
    report_path = os.path.join(
        OUTPUT_DIR, "Appendix_A_run_summary.txt"
    )

    with open(report_path, "w") as f:
        f.write("JTECH Appendix A descriptor-based classifier audit\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Features: {', '.join(CANON_FEATURES)}\n")
        f.write(f"Validation fraction: {VAL_SIZE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"CV folds: {CV_FOLDS}\n")
        f.write(f"RF random-search iterations: {N_ITER_RF}\n")
        f.write(f"XGB random-search iterations: {N_ITER_XGB}\n")
        f.write(f"ADASYN params: {ADASYN_PARAMS}\n")
        f.write(f"SMOTE params: {SMOTE_PARAMS}\n\n")

        f.write("Development split counts\n")
        f.write(pd.DataFrame(split_rows).to_string(index=False))
        f.write("\n\nResampling counts\n")
        f.write(pd.DataFrame(resample_rows).to_string(index=False))
        f.write("\n\nTuning summary\n")
        f.write(pd.DataFrame(tuning_summary_rows).to_string(index=False))
        f.write("\n\nValidation-selected thresholds\n")
        f.write(pd.DataFrame(threshold_rows).to_string(index=False))
        f.write("\n\nIndependent-test performance\n")
        f.write(independent[appendix_cols].to_string(index=False))
        f.write("\n")

    print("\n============================================================")
    print("DONE")
    print("============================================================")
    print("Appendix A outputs written to:")
    print(OUTPUT_DIR)
    print("\nKey files:")
    print("  Appendix_A_model_performance_summary.csv")
    print("  selected_hyperparameters.csv")
    print("  tuning_summary.csv")
    print("  validation_selected_thresholds.csv")
    print("  dataset_QC_counts.csv")
    print("  complete_feature_counts.csv")
    print("  development_split_counts.csv")
    print("  resampling_counts.csv")
    print("  Figure_A1_independent_test_confusion_matrices.png/.pdf")
    print("  Appendix_independent_test_precision_recall_curves.png/.pdf")
    print("  Appendix_A_run_summary.txt")


if __name__ == "__main__":
    main()
