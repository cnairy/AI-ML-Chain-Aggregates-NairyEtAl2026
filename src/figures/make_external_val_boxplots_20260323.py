#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create external-validation figures from per-fold summary CSVs.

Reads by default:
  {base}/comparisons_30ep/external_val_summaries_30ep_20260323/per_fold_summary.csv
  {base}/comparisons_30ep/external_val_summaries_30ep_20260323/per_class_by_fold_bestF1.csv
  {base}/comparisons_30ep/external_val_summaries_30ep_20260323/per_class_by_fold_opthr.csv
  {base}/comparisons_30ep/external_val_summaries_30ep_20260323/per_class_by_fold_floor.csv

Outputs in that same folder:
  external_val_accuracy_bestF1_boxplot.png
  external_val_accuracy_opthr_boxplot.png
  external_val_accuracy_floor_boxplot.png

  external_val_PRF_bestF1_combined.png
  external_val_PRF_opthr_combined.png
  external_val_PRF_floor_combined.png

  external_val_PRF_fig5a_bestF1_combined.png
  external_val_PRF_fig5a_opthr_combined.png
  external_val_PRF_fig5a_floor_combined.png

  external_val_PRF_macro_bestF1_combined.png
  external_val_PRF_macro_opthr_combined.png
  external_val_PRF_macro_floor_combined.png

  external_val_PRF_chainonly_bestF1_combined.png
  external_val_PRF_chainonly_opthr_combined.png
  external_val_PRF_chainonly_floor_combined.png

Usage:
  python3 make_external_val_boxplots.py --base /path/to/nas_workspace/CNN/figures/30_epochs
"""

from typing import Optional, Tuple, List
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Appearance ----------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "axes.grid": True,
})

# Preferred fixed order for x-axis (left->right)
PREFERRED_ORDER = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG16", "VGG19"]

# Layout knobs
GROUP_SPACING = 1.4
PRF_BOX_WIDTH = 0.22
ACC_BOX_WIDTH = 0.45
PRF_OFFSET = 0.28

BAND_ALPHA = 0.08
BAND_COLOR = "#000000"

MEAN_PROPS = dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=3, alpha=0.75)
MEDIAN_PROPS = dict(color='black', linewidth=2)


# ---------- Utilities ----------
def _boxplot(ax, data, labels=None, positions=None, **kw):
    """Matplotlib 3.9+ 'labels' -> 'tick_labels' compat."""
    try:
        if labels is not None:
            return ax.boxplot(data, tick_labels=labels, positions=positions, **kw)
        return ax.boxplot(data, positions=positions, **kw)
    except TypeError:
        if labels is not None:
            return ax.boxplot(data, labels=labels, positions=positions, **kw)
        return ax.boxplot(data, positions=positions, **kw)


def alternating_bands(ax, centers, alpha=BAND_ALPHA, color=BAND_COLOR):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return

    spacing = GROUP_SPACING if centers.size == 1 else float(np.diff(centers).mean())
    xmin = centers[0] - spacing / 2
    xmax = centers[-1] + spacing / 2
    ax.set_xlim(xmin, xmax)

    for i, c in enumerate(centers):
        if i % 2 == 0:
            ax.axvspan(c - spacing / 2, c + spacing / 2, color=color, alpha=alpha, zorder=0)

    ax.autoscale(enable=False, axis='x')


def filter_order(models):
    present = [m for m in PREFERRED_ORDER if m in models]
    extras = [m for m in models if m not in PREFERRED_ORDER]
    return present + extras


def _prepare_centers(order):
    n_groups = len(order)
    centers = 1 + GROUP_SPACING * np.arange(n_groups, dtype=float)
    pos_f1 = centers - PRF_OFFSET
    pos_prec = centers
    pos_rec = centers + PRF_OFFSET
    return centers, pos_f1, pos_prec, pos_rec


def _recompute_f1(prec: pd.Series, rec: pd.Series) -> pd.Series:
    return 2 * (prec * rec) / (prec + rec + 1e-12)


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fix_f1(df: pd.DataFrame, p_col: str, r_col: str, f1_col: str, name: str):
    _coerce_numeric(df, [p_col, r_col, f1_col])
    f1_re = _recompute_f1(df[p_col], df[r_col])

    if f1_col not in df.columns:
        df[f1_col] = f1_re
        print(f"[INFO] {name}: no '{f1_col}' column; using recomputed F1 from P/R.")
    else:
        mismatch = (df[f1_col] - f1_re).abs() > 0.02
        if mismatch.any():
            n_bad = int(mismatch.sum())
            print(f"[WARN] {name}: {n_bad} rows where '{f1_col}' != 2PR/(P+R). Replacing with recomputed F1.")
            df.loc[mismatch, f1_col] = f1_re[mismatch]

    valid = (
        df[p_col].between(0, 1) &
        df[r_col].between(0, 1) &
        df[f1_col].between(0, 1)
    )
    n_drop = int((~valid).sum())
    if n_drop > 0:
        print(f"[WARN] {name}: dropping {n_drop} rows with out-of-range/NaN metrics.")
        df.drop(index=df.index[~valid], inplace=True)

    return df


# ---------- Per-class CSV loader ----------
def _load_per_class_df(comp_dir: Path, which: str) -> Optional[pd.DataFrame]:
    """
    Load per-class × per-fold CSV.
    Expected columns after light renaming:
      model, fold, class, precision, recall, f1
    """
    fname = f"per_class_by_fold_{which}.csv"
    p = comp_dir / fname
    if not p.exists():
        return None

    df = pd.read_csv(p)

    needed = {"model", "fold", "class", "precision", "recall", "f1"}
    if not needed.issubset(df.columns):
        colmap = {}
        for tgt, cands in {
            "model": ["model"],
            "fold": ["fold", "cv_fold", "kfold"],
            "class": ["class", "class_label", "label"],
            "precision": ["precision", "prec", f"prec@{which}"],
            "recall": ["recall", "rec", f"rec@{which}"],
            "f1": ["f1", f"f1@{which}", "F1", f"F1@{which}"],
        }.items():
            for c in cands:
                if c in df.columns:
                    colmap[c] = tgt
                    break

        df = df.rename(columns=colmap)
        if not needed.issubset(df.columns):
            print(f"[WARN] Loader: {fname} missing required columns after rename. Skipping.")
            return None

    df["model"] = df["model"].astype(str)
    df = _fix_f1(df, "precision", "recall", "f1", name=f"per-class:{which}")

    return df[["model", "fold", "class", "precision", "recall", "f1"]]


# ---------- helpers for class filtering & macro ----------
def _is_chain_class(label: str) -> bool:
    s = str(label).lower()
    if "chain" in s and "non" not in s:
        return True
    return s.startswith("1_") or s == "1" or s == "positive"


def _macro_by_fold(df_pc: pd.DataFrame) -> pd.DataFrame:
    grp = df_pc.groupby(["model", "fold"], as_index=False)[["precision", "recall", "f1"]].mean()
    grp["agg"] = "macro"
    return grp.rename(columns={"precision": "P", "recall": "R", "f1": "F1"})


def _chain_only_by_fold(df_pc: pd.DataFrame) -> pd.DataFrame:
    df_chain = df_pc[df_pc["class"].apply(_is_chain_class)].copy()
    df_chain["agg"] = "chain"
    return df_chain.rename(columns={"precision": "P", "recall": "R", "f1": "F1"})


def _collect_prf_arrays(df_in: pd.DataFrame, order: List[str]) -> Tuple[list, list, list]:
    f1 = [df_in.loc[df_in["model"] == m, "F1"].dropna().values for m in order]
    P = [df_in.loc[df_in["model"] == m, "P"].dropna().values for m in order]
    R = [df_in.loc[df_in["model"] == m, "R"].dropna().values for m in order]
    return f1, P, R


# ---------- Plot helpers ----------
def _plot_accuracy_boxplot(acc_data, order, title, out_path):
    centers, _, _, _ = _prepare_centers(order)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(order)), 7.0))

    _boxplot(
        ax, acc_data, labels=order, positions=centers,
        patch_artist=True, showmeans=True, widths=ACC_BOX_WIDTH,
        meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS
    )
    alternating_bands(ax, centers, alpha=BAND_ALPHA, color=BAND_COLOR)

    ax.set_title(title)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(which='major', axis='x')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def _plot_prf_combined(f1_data, prec_data, rec_data, order, title, out_path):
    centers, pos_f1, pos_prec, pos_rec = _prepare_centers(order)
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(order)), 7.8))
    alternating_bands(ax, centers, alpha=BAND_ALPHA, color=BAND_COLOR)

    bp1 = _boxplot(ax, f1_data, positions=pos_f1, patch_artist=True, showmeans=True,
                   widths=PRF_BOX_WIDTH, meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS)
    bp2 = _boxplot(ax, prec_data, positions=pos_prec, patch_artist=True, showmeans=True,
                   widths=PRF_BOX_WIDTH, meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS)
    bp3 = _boxplot(ax, rec_data, positions=pos_rec, patch_artist=True, showmeans=True,
                   widths=PRF_BOX_WIDTH, meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS)

    for patch in bp1['boxes']:
        patch.set_facecolor("#F4A3A3")   # light red
    for patch in bp2['boxes']:
        patch.set_facecolor("#9EC9FF")   # light blue
    for patch in bp3['boxes']:
        patch.set_facecolor("#A8DDA8")   # light green
    
    ax.set_xticks(centers)
    ax.set_xticklabels(order, rotation=25)
    ax.set_ylim(0.60, 1.0)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(which='major', axis='x')
    
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#F4A3A3", label="F1-score"),
        Patch(facecolor="#9EC9FF", label="Precision"),
        Patch(facecolor="#A8DDA8", label="Recall"),
    ], loc="lower left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ---------------- Main plotting ----------------
def main(base_dir: Path, subdir: str):
    comp_dir = base_dir / "comparisons_30ep" / subdir
    in_csv = comp_dir / "per_fold_summary.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Expected {in_csv} (run the evaluation script first).")

    df = pd.read_csv(in_csv)
    df.columns = [c.strip() for c in df.columns]

    metric_sets = {
        "bestF1": {
            "acc": "acc@bestF1",
            "prec": "prec@bestF1",
            "rec": "rec@bestF1",
            "f1": "f1@bestF1",
        },
        "opthr": {
            "acc": "acc@opthr",
            "prec": "prec@opthr",
            "rec": "rec@opthr",
            "f1": "f1@opthr",
        },
        "floor": {
            "acc": "acc@floor",
            "prec": "prec@floor",
            "rec": "rec@floor",
            "f1": "f1@floor",
        },
    }

    # Determine order once
    models_present = df["model"].dropna().unique().tolist()
    order = filter_order(models_present)

    # ----- per_fold_summary plots -----
    for which, cols in metric_sets.items():
        needed = ["model", cols["acc"], cols["prec"], cols["rec"], cols["f1"]]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] Skipping {which}: missing columns {missing}")
            continue

        df_use = df.copy()
        _coerce_numeric(df_use, [cols["acc"], cols["prec"], cols["rec"], cols["f1"]])
        df_use = _fix_f1(df_use, cols["prec"], cols["rec"], cols["f1"], name=f"per_fold_summary:{which}")

        acc_data = [df_use.loc[df_use["model"] == m, cols["acc"]].dropna().values for m in order]
        _plot_accuracy_boxplot(
            acc_data,
            order,
            title=f"External Validation — Accuracy @ {which} (No Averaging across folds)",
            out_path=comp_dir / f"external_val_accuracy_{which}_boxplot.png"
        )

        f1_data = [df_use.loc[df_use["model"] == m, cols["f1"]].dropna().values for m in order]
        prec_data = [df_use.loc[df_use["model"] == m, cols["prec"]].dropna().values for m in order]
        rec_data = [df_use.loc[df_use["model"] == m, cols["rec"]].dropna().values for m in order]

        _plot_prf_combined(
            f1_data, prec_data, rec_data, order,
            title=f"Variation across folds — Precision / Recall / F1 @ {which}\n(No Averaging across folds)",
            out_path=comp_dir / f"external_val_PRF_{which}_combined.png"
        )

    # ----- per_class_by_fold plots -----
    for which in ("bestF1", "opthr", "floor"):
        df_pc = _load_per_class_df(comp_dir, which)
        if df_pc is None or df_pc.empty:
            continue

        models_pc = df_pc["model"].dropna().unique().tolist()
        order_pc = filter_order(models_pc)

        # Fig-5a style: all per-class rows
        f1_pc = [df_pc.loc[df_pc["model"] == m, "f1"].dropna().values for m in order_pc]
        P_pc = [df_pc.loc[df_pc["model"] == m, "precision"].dropna().values for m in order_pc]
        R_pc = [df_pc.loc[df_pc["model"] == m, "recall"].dropna().values for m in order_pc]

        _plot_prf_combined(
            f1_pc, P_pc, R_pc, order_pc,
            title=f"Precision / Recall / F1 @ {which} (Per-class × Per-fold)",
            out_path=comp_dir / f"external_val_PRF_fig5a_{which}_combined.png"
        )

        # Macro across classes
        macro = _macro_by_fold(df_pc)
        models_macro = macro["model"].dropna().unique().tolist()
        order_macro = filter_order(models_macro)
        f1_m, P_m, R_m = _collect_prf_arrays(macro, order_macro)

        _plot_prf_combined(
            f1_m, P_m, R_m, order_macro,
            title=f"Precision / Recall / F1 @ {which} — MACRO across classes (per fold)",
            out_path=comp_dir / f"external_val_PRF_macro_{which}_combined.png"
        )

        # Chain only
        chain = _chain_only_by_fold(df_pc)
        if chain.empty:
            print(f"[WARN] No chain-class rows found for {which}; skipping chain-only plot.")
            continue

        models_chain = chain["model"].dropna().unique().tolist()
        order_chain = filter_order(models_chain)
        f1_c, P_c, R_c = _collect_prf_arrays(chain, order_chain)

        _plot_prf_combined(
            f1_c, P_c, R_c, order_chain,
            title= "Chain Aggregate Class Only (Variation in Folds)",
            out_path=comp_dir / f"external_val_PRF_chainonly_{which}_combined.png"
        )

    print(f"Saved figures to: {comp_dir}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True,
                    help="Base dir, e.g. /path/to/nas_workspace/CNN/figures/30_epochs")
    ap.add_argument("--subdir", type=str, default="external_val_summaries_30ep_20260323",
                    help="Subdirectory inside comparisons_30ep to read/write plots from")
    args = ap.parse_args()
    main(Path(args.base), args.subdir)