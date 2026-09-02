#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LOWER_IS_BETTER = {"logloss", "brier", "ece"}
DEFAULT_MODEL_ORDER = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG16", "VGG19"]


# ---------- CSV loading ----------
def _resolve_col(df, base, prefer="opthr"):
    cols = {c.lower(): c for c in df.columns}
    base_l = base.lower()

    if base_l in cols:
        return cols[base_l]

    pat = re.compile(rf"^{re.escape(base_l)}@(opthr|bestf1)$", re.IGNORECASE)
    matches = {}
    for lc, orig in cols.items():
        m = pat.match(lc)
        if m:
            matches[m.group(1).lower()] = orig

    order = {
        "opthr": ["opthr", "bestf1"],
        "bestF1": ["bestf1", "opthr"],
        "auto": ["opthr", "bestf1"],
    }[prefer]

    for key in order:
        if key in matches:
            return matches[key]

    return None


def read_per_fold_csv(path: Path, prefer="opthr") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path)

    model_col = next((c for c in df.columns if c.lower() == "model"), None)
    if model_col is None:
        raise ValueError("CSV needs a 'model' column.")

    fold_col = next((c for c in df.columns if c.lower() in {"fold", "cv_fold", "kfold", "split"}), None)

    roc_col = _resolve_col(df, "roc_auc", prefer)
    pr_col = _resolve_col(df, "pr_auc", prefer)
    log_col = (
        _resolve_col(df, "logloss", prefer)
        or _resolve_col(df, "nll", prefer)
        or _resolve_col(df, "cross_entropy", prefer)
    )
    brier_col = (
        _resolve_col(df, "brier", prefer)
        or _resolve_col(df, "brier_score", prefer)
        or _resolve_col(df, "brier_score_loss", prefer)
    )
    ece_col = (
        _resolve_col(df, "ece", prefer)
        or _resolve_col(df, "expected_calibration_error", prefer)
    )

    if roc_col is None or pr_col is None:
        raise ValueError("CSV must contain roc_auc* and pr_auc* (plain or @opthr/@bestF1).")

    keep = [
        ("model", model_col),
        ("fold", fold_col),
        ("roc_auc", roc_col),
        ("pr_auc", pr_col),
        ("logloss", log_col),
        ("brier", brier_col),
        ("ece", ece_col),
    ]

    out = pd.DataFrame()
    for std, src in keep:
        if src is not None and src in df.columns:
            out[std] = df[src]

    return out


# ---------- helpers ----------
def order_models(labels):
    return DEFAULT_MODEL_ORDER if all(m in labels for m in DEFAULT_MODEL_ORDER) else sorted(labels)


def percentiles(a, p):
    return np.nanpercentile(a, p) if len(a) else np.nan


def write_stats(df: pd.DataFrame, outdir: Path):
    metrics = ["roc_auc", "pr_auc", "logloss", "brier", "ece"]
    have = [c for c in metrics if c in df.columns]

    rows = []
    for model, sub in df.groupby("model"):
        for m in have:
            arr = sub[m].dropna().values
            rows.append({
                "model": model,
                "metric": m,
                "n": len(arr),
                "mean": np.nanmean(arr) if len(arr) else np.nan,
                "std": np.nanstd(arr) if len(arr) else np.nan,
                "median": np.nanmedian(arr) if len(arr) else np.nan,
                "min": np.nanmin(arr) if len(arr) else np.nan,
                "max": np.nanmax(arr) if len(arr) else np.nan,
                "p5": percentiles(arr, 5),
                "p95": percentiles(arr, 95),
            })

    pd.DataFrame(rows).sort_values(["metric", "model"]).to_csv(
        outdir / "per_metric_stats_by_model.csv", index=False
    )


# ---------- individual panel ----------
def boxplot_single_metric(df: pd.DataFrame, models, metric: str, outdir: Path):
    clean = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "logloss": "LogLoss",
        "brier": "Brier",
        "ece": "ECE",
    }[metric]

    display_title = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "logloss": "Log loss",
        "brier": "Brier score",
        "ece": "ECE",
    }[metric]

    series = [df.loc[df["model"] == m, metric].dropna().values for m in models]

    panel_w = max(6.0, 1.2 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(panel_w, 5.0), constrained_layout=True)

    ax.boxplot(
        series,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="#d9e1f2", alpha=0.55, linewidth=1.6),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(color="0.25", linewidth=1.4),
        capprops=dict(color="0.25", linewidth=1.4),
        flierprops=dict(
            marker="*",
            markersize=7,
            markerfacecolor="0.15",
            markeredgecolor="0.15",
            alpha=0.9,
        ),
    )

    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=35, ha="right")

    title = display_title + (" (lower is better)" if metric in LOWER_IS_BETTER else "")
    ax.set_title(title, pad=10)
    ax.set_ylabel("Value")
    ax.grid(True, which="both", axis="both", alpha=0.35)
    ax.margins(x=0.03)

    basefile = f"external_boxplot_{clean}"
    fig.savefig(outdir / f"{basefile}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{basefile}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------- 2x2 combined helpers ----------
def _nice_y_formatter(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return FuncFormatter(lambda x, p: f"{x:.3f}")

    vmax = np.nanmax(arr)
    if vmax < 0.02:
        fmt = "{x:.4f}"
    elif vmax < 1.0:
        fmt = "{x:.3f}"
    else:
        fmt = "{x:.2f}"

    return FuncFormatter(lambda x, p: fmt.format(x=x))


def _one_boxplot(ax, df, models, metric, show_xlabels=True):
    pos = np.arange(1, len(models) + 1)
    series = [df.loc[df["model"] == m, metric].dropna().values for m in models]

    ax.boxplot(
        series,
        positions=pos,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor="#d9e1f2", alpha=0.55, linewidth=1.4),
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="0.25", linewidth=1.2),
        capprops=dict(color="0.25", linewidth=1.2),
        flierprops=dict(
            marker="*",
            markersize=6,
            markerfacecolor="0.15",
            markeredgecolor="0.15",
            alpha=0.9,
        ),
    )

    ax.set_xlim(0.5, len(models) + 0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(models if show_xlabels else [], rotation=20, ha="center")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    all_vals = np.concatenate([a for a in series if len(a)]) if any(len(a) for a in series) else np.array([0, 1])
    ax.yaxis.set_major_formatter(_nice_y_formatter(all_vals))
    ax.grid(True, which="major", axis="both", alpha=0.3)
    ax.margins(x=0.02)


def _clean_title(metric):
    return {
        "brier": "Brier score",
        "logloss": "Log loss",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }[metric]


# ---------- 2x2 combined ----------
def combined_2x2(df: pd.DataFrame, models, outdir: Path, prefer: str):
    wanted = ["brier", "logloss", "roc_auc", "pr_auc"]
    if not all(m in df.columns for m in wanted):
        return

    fig, axes = plt.subplots(2, 2, figsize=(max(12, 2.0 * len(models)), 8.0), sharex=True)
    axes = axes.ravel()

    for i, metric in enumerate(wanted):
        ax = axes[i]
        show_x = (i >= 2)
        _one_boxplot(ax, df, models, metric, show_xlabels=show_x)
        ax.set_title(_clean_title(metric), pad=6)
        if i % 2 == 0:
            ax.set_ylabel("Value")

    suffix = {
        "opthr": "@opthr",
        "bestF1": "@bestF1",
        "auto": " (preferred selection)",
    }[prefer]

    fig.suptitle("Validation Metrics (per Fold)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if prefer == "opthr":
        stem = "external_opthr_boxplots_2x2"
    elif prefer == "bestF1":
        stem = "external_bestF1_boxplots_2x2"
    else:
        stem = "external_selected_boxplots_2x2"

    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=str)
    ap.add_argument("--csv", default=None, type=str)
    ap.add_argument("--combined", action="store_true", help="Also save a 2x2 combined figure")
    ap.add_argument("--prefer", choices=["opthr", "bestF1", "auto"], default="opthr")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    infile = (
        Path(args.csv)
        if args.csv
        else (base / "comparisons_30ep" / "external_val_summaries_30ep" / "per_fold_summary.csv")
    )
    outdir = base / "comparisons_30ep" / "external_30ep"
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_per_fold_csv(infile, prefer=args.prefer)
    write_stats(df, outdir)

    models = order_models(df["model"].unique().tolist())

    metrics_to_plot = ["roc_auc", "pr_auc", "logloss", "brier", "ece"]
    for m in [c for c in metrics_to_plot if c in df.columns]:
        boxplot_single_metric(df, models, m, outdir)

    if args.combined:
        combined_2x2(df, models, outdir, prefer=args.prefer)

    print(
        "Saved individual boxplots and stats"
        + ("; combined 2x2 figure saved." if args.combined else " (use --combined to enable combined figure).")
    )


if __name__ == "__main__":
    main()