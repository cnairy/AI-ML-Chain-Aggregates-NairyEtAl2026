#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Balanced CNN model+epoch selection (VALIDATION ONLY) with STRICT eligibility,
PLUS comparison to ResNet34 epoch 18.

Inputs (per REFIT*/plot_data):
- REQUIRED: confusion_matrix_per_epoch.txt
  Lines like:
    epoch=18 thr=0.681 TN=10984 FP=49 FN=30 TP=176
- OPTIONAL: epoch_metrics.csv with columns including:
    val_ece, val_brier, val_logloss, val_loss

Selection:
Stage A: Eligibility filters (STRICT)
  precision >= PREC_FLOOR
  recall    >= RECALL_MIN
  f1        >= F1_MIN

Stage B: Score among eligible:
  Core score:
    score_core = W_PREC*precision + W_F1*f1 + W_REC*recall
                 - W_FP_RATE*FP_rate - W_FN_RATE*FN_rate
    FP_rate = FP/(FP+TN)
    FN_rate = FN/(FN+TP)

  Validation error penalty (lower is better), robust-normalized globally:
    val_penalty = W_ECE*ece_n + W_BRIER*brier_n + W_LOGLOSS*logloss_n + W_LOSS*loss_n
    score_total = score_core - val_penalty

Comparison:
- Compare BEST_OVERALL vs ResNet34 epoch 18 (validation only)
- Prints both rows + deltas for key metrics and error metrics

Outputs:
- BEST_MODEL_EPOCH_REPORT_VALIDATION_ONLY.txt  (clean report + TOP10 + comparison block)
- TOP10_overall_candidates_VALIDATION_ONLY.csv
- best_epoch_per_model_balanced_VALIDATION_ONLY.csv
- ranked_epochs_<model>_<refit>_VALIDATION_ONLY.csv (optional)
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# USER SETTINGS
# -----------------------------
RUN_ROOTS = {
    "ResNet18": "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-ResNet18_training_evalutation_20251021-hyp-cv-30ep",
    "ResNet34": "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-ResNet34_training_evalutation_20251021d-hyp-cv-30ep",
    "ResNet50": "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-ResNet50_training_evalutation_20251104d-hyp-cv-30ep",
    "ResNet101": "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-ResNet101_training_evalutation_20251105-hyp-cv-30ep",
    "VGG16": "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-VGG16_training_evalutation_20251014-hyp-cv-30ep",
    "VGG19": "/path/to/nas_workspace/CNN/figures/30_epochs/JTech-VGG19_training_evalutation_20251106-hyp-cv-30ep",
}

OUT_ROOT = "/path/to/nas_workspace/CNN/figures/30_epochs/_best_model_selection_outputs_VALIDATION_ONLY"

# Eligibility filters (STRICT)
F1_MIN = 0.75
RECALL_MIN = 0.75
PREC_FLOOR = 0.90
STRICT_ELIGIBILITY = True

# Core score weights
W_PREC = 0.45
W_F1 = 0.35
W_REC = 0.20
W_FP_RATE = 0.15
W_FN_RATE = 0.05

# Validation error weights (penalty; lower is better)
W_ECE = 0.15
W_BRIER = 0.15
W_LOGLOSS = 0.20
W_LOSS = 0.10
VAL_METRICS = ["val_ece", "val_brier", "val_logloss", "val_loss"]

# If True, require ALL val metrics present for an epoch to be considered.
REQUIRE_VAL_METRICS = False

# Report settings
TOPK_OVERALL = 10
SAVE_RANKED_EPOCHS_PER_REFIT = True

# Comparison target
COMPARE_MODEL = "ResNet34"
COMPARE_EPOCH = 18


# -----------------------------
# HELPERS
# -----------------------------
CM_TXT_RE = re.compile(
    r"epoch\s*=\s*(\d+)\s+thr\s*=\s*([0-9]*\.?[0-9]+)\s+TN\s*=\s*(\d+)\s+FP\s*=\s*(\d+)\s+FN\s*=\s*(\d+)\s+TP\s*=\s*(\d+)",
    re.IGNORECASE,
)


def is_valid_plot_dir(plot_dir: Path) -> bool:
    return plot_dir.is_dir() and plot_dir.name == "plot_data" and (plot_dir / "confusion_matrix_per_epoch.txt").exists()


def discover_plot_dirs_recursive(root_or_plot: Path) -> List[Path]:
    if root_or_plot.is_dir() and root_or_plot.name == "plot_data":
        return [root_or_plot] if is_valid_plot_dir(root_or_plot) else []
    if not root_or_plot.exists():
        return []
    hits: List[Path] = []
    for f in root_or_plot.rglob("confusion_matrix_per_epoch.txt"):
        plot_dir = f.parent
        if is_valid_plot_dir(plot_dir):
            hits.append(plot_dir)
    return sorted(set(hits), key=lambda x: str(x))


def load_val_confusion_txt_from_plot_dir(plot_dir: Path) -> pd.DataFrame:
    p = plot_dir / "confusion_matrix_per_epoch.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    rows: List[Dict] = []
    with open(p, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = CM_TXT_RE.search(line)
            if not m:
                continue

            epoch = int(m.group(1))
            thr = float(m.group(2))
            tn = int(m.group(3))
            fp = int(m.group(4))
            fn = int(m.group(5))
            tp = int(m.group(6))

            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-12)

            rows.append(
                dict(
                    epoch=epoch,
                    threshold_used=thr,
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    tp=tp,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                    fp_rate=fp / (fp + tn + 1e-12),
                    fn_rate=fn / (fn + tp + 1e-12),
                )
            )

    if not rows:
        raise RuntimeError(f"Parsed 0 epoch rows from: {p}")

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    df["epoch"] = df["epoch"].astype(int)
    return df


def load_epoch_metrics_from_plot_dir(plot_dir: Path) -> Optional[pd.DataFrame]:
    p = plot_dir / "epoch_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "epoch" not in df.columns:
        return None
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df[pd.notna(df["epoch"])].copy()
    df["epoch"] = df["epoch"].astype(int)
    return df


def attach_val_metrics(val_df: pd.DataFrame, plot_dir: Path) -> pd.DataFrame:
    out = val_df.copy()
    for k in VAL_METRICS:
        if k not in out.columns:
            out[k] = np.nan

    vdf = load_epoch_metrics_from_plot_dir(plot_dir)
    if vdf is None or vdf.empty:
        return out

    keep = ["epoch"]
    for c in VAL_METRICS + ["val_roc_auc", "val_pr_auc"]:
        if c in vdf.columns:
            keep.append(c)

    vdf = vdf[keep].drop_duplicates(subset=["epoch"])
    out = out.merge(vdf, on="epoch", how="left")

    for k in VAL_METRICS:
        if k not in out.columns:
            out[k] = np.nan

    return out


def compute_core_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_core"] = (
        W_PREC * out["precision"]
        + W_F1 * out["f1"]
        + W_REC * out["recall"]
        - W_FP_RATE * out["fp_rate"]
        - W_FN_RATE * out["fn_rate"]
    )
    return out


def add_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["eligible"] = (
        (out["precision"] >= PREC_FLOOR)
        & (out["recall"] >= RECALL_MIN)
        & (out["f1"] >= F1_MIN)
    ).astype(int)
    return out


def robust_minmax_norm(series: pd.Series, p_low=5.0, p_high=95.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.nan, index=series.index)

    lo = np.nanpercentile(s.values, p_low)
    hi = np.nanpercentile(s.values, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.nan, index=series.index)

    x = (s - lo) / (hi - lo)
    return x.clip(lower=0.0, upper=1.0)


def compute_total_score(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["val_ece_n"] = robust_minmax_norm(out["val_ece"])
    out["val_brier_n"] = robust_minmax_norm(out["val_brier"])
    out["val_logloss_n"] = robust_minmax_norm(out["val_logloss"])
    out["val_loss_n"] = robust_minmax_norm(out["val_loss"])

    ece_n = out["val_ece_n"].fillna(0.0)
    bri_n = out["val_brier_n"].fillna(0.0)
    log_n = out["val_logloss_n"].fillna(0.0)
    los_n = out["val_loss_n"].fillna(0.0)

    out["val_penalty"] = (W_ECE * ece_n) + (W_BRIER * bri_n) + (W_LOGLOSS * log_n) + (W_LOSS * los_n)
    out["score_total"] = out["score_core"] - out["val_penalty"]
    return out


def rank_epochs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if REQUIRE_VAL_METRICS:
        mask = np.ones(len(out), dtype=bool)
        for k in VAL_METRICS:
            mask &= pd.notna(out[k]).values
        out = out[mask].copy()

    if out.empty:
        return out

    eligible = out[out["eligible"] == 1].copy()

    if STRICT_ELIGIBILITY:
        if eligible.empty:
            return eligible
        cand = eligible
    else:
        cand = eligible if not eligible.empty else out

    sort_cols = [
        "score_total",
        "score_core",
        "precision",
        "f1",
        "recall",
        "fp",
        "fn",
        "val_logloss",
        "val_brier",
        "val_ece",
        "val_loss",
    ]
    ascending = [False, False, False, False, False, True, True, True, True, True, True]

    for c in sort_cols:
        if c not in cand.columns:
            cand[c] = np.nan

    return cand.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)


def safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")


def fmt_float(x, nd=6) -> str:
    try:
        if x is None:
            return ""
        if not np.isfinite(float(x)):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def make_table(df: pd.DataFrame, cols: List[Tuple[str, str]], max_rows: int) -> List[str]:
    sub = df.copy().head(max_rows)

    headers = [h for _, h in cols]
    widths: List[int] = []
    for col, header in cols:
        maxlen = len(header)
        for v in sub[col].tolist():
            s = str(v)
            if len(s) > maxlen:
                maxlen = len(s)
        widths.append(maxlen)

    def row_to_line(vals: List[str]) -> str:
        return "  " + "  ".join(v.ljust(w) for v, w in zip(vals, widths))

    lines: List[str] = []
    lines.append(row_to_line(headers))
    lines.append("  " + "  ".join(("-" * w) for w in widths))
    for _, r in sub.iterrows():
        lines.append(row_to_line([str(r[c]) for c, _ in cols]))
    return lines


def _pick_first_resnet34_plotdir(refit_frames: List[Tuple[str, str, Path, pd.DataFrame]]) -> Optional[Path]:
    # Prefer a REFIT folder that contains "REFIT" in name
    candidates = [plot_dir for (m, _, plot_dir, _) in refit_frames if m == "ResNet34"]
    if not candidates:
        return None
    candidates_refit = [p for p in candidates if p.parent.name.lower().startswith("refit")]
    if candidates_refit:
        return candidates_refit[0]
    return candidates[0]


def _row_for_model_epoch(pool_scored: pd.DataFrame, model: str, epoch: int, plot_dir: Optional[Path] = None) -> Optional[pd.Series]:
    sub = pool_scored[pool_scored["model"] == model].copy()
    if plot_dir is not None:
        sub = sub[sub["plot_dir"] == str(plot_dir)]
    r = sub[sub["epoch"] == int(epoch)]
    if r.empty:
        return None
    return r.iloc[0]


def _fmt_line(label: str, r: pd.Series) -> str:
    parts = [
        f"{label}: model={r['model']} refit={r['refit_name']} epoch={int(r['epoch']):02d} thr={float(r['threshold_used']):.6f}",
        f"prec={float(r['precision']):.6f} rec={float(r['recall']):.6f} f1={float(r['f1']):.6f}",
        f"TN={int(r['tn'])} FP={int(r['fp'])} FN={int(r['fn'])} TP={int(r['tp'])}",
        f"score_total={float(r['score_total']):.6f} core={float(r['score_core']):.6f} pen={float(r['val_penalty']):.6f}",
    ]
    # add validation metrics if present
    vm = []
    for k in VAL_METRICS:
        if k in r.index and np.isfinite(r[k]):
            vm.append(f"{k}={float(r[k]):.6f}")
    if vm:
        parts.append(" ".join(vm))
    return " | ".join(parts)


def _delta_lines(a: pd.Series, b: pd.Series, label: str) -> List[str]:
    """
    Deltas (a minus b)
    """
    keys = ["precision", "recall", "f1", "fp", "fn", "tp", "tn", "score_total", "score_core", "val_penalty"] + VAL_METRICS
    out: List[str] = [label]
    for k in keys:
        if k not in a.index or k not in b.index:
            continue
        va = a[k]
        vb = b[k]
        if pd.isna(va) or pd.isna(vb):
            continue
        if k in ["fp", "fn", "tp", "tn"]:
            out.append(f"  d_{k} = {int(va) - int(vb):+d}")
        else:
            out.append(f"  d_{k} = {float(va) - float(vb):+.6f}")
    return out


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    now = datetime.today().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUT_ROOT) / f"best_cnn_epoch_selection_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_models: List[str] = []
    model_refit_counts: Dict[str, int] = {}

    refit_frames: List[Tuple[str, str, Path, pd.DataFrame]] = []
    pool_rows: List[pd.DataFrame] = []

    for model, root in sorted(RUN_ROOTS.items()):
        root_p = Path(root)
        plot_dirs = discover_plot_dirs_recursive(root_p)
        model_refit_counts[model] = len(plot_dirs)

        if not plot_dirs:
            missing_models.append(f"{model} (no plot_data/confusion_matrix_per_epoch.txt under: {root})")
            continue

        for plot_dir in plot_dirs:
            refit_name = plot_dir.parent.name

            df = load_val_confusion_txt_from_plot_dir(plot_dir)
            df = attach_val_metrics(df, plot_dir)
            df = compute_core_score(df)
            df = add_eligibility(df)

            df["model"] = model
            df["refit_name"] = refit_name
            df["plot_dir"] = str(plot_dir)

            refit_frames.append((model, refit_name, plot_dir, df))
            pool_rows.append(df)

    if not pool_rows:
        raise RuntimeError("No validation epochs loaded. Check RUN_ROOTS and file locations.")

    pool = pd.concat(pool_rows, ignore_index=True)

    # Global eligibility sanity print
    global_eligible = pool[pool["eligible"] == 1]
    print(f"GLOBAL eligible epochs: {len(global_eligible)} / {len(pool)} "
          f"(PREC_FLOOR={PREC_FLOOR}, RECALL_MIN={RECALL_MIN}, F1_MIN={F1_MIN})")

    pool_scored = compute_total_score(pool)

    # Rank per refit, collect best per refit
    best_per_refit_rows: List[Dict] = []
    excluded_refits: List[str] = []

    for model, refit_name, plot_dir, _ in refit_frames:
        df = pool_scored[(pool_scored["model"] == model) & (pool_scored["refit_name"] == refit_name)].copy()
        df = df.sort_values("epoch").reset_index(drop=True)

        ranked = rank_epochs(df)

        if ranked.empty:
            excluded_refits.append(f"{model} | {refit_name} (no eligible epochs under current filters)")
            continue

        if SAVE_RANKED_EPOCHS_PER_REFIT:
            ranked_csv = out_dir / f"ranked_epochs_{safe_slug(model)}_{safe_slug(refit_name)}_VALIDATION_ONLY.csv"
            ranked.to_csv(ranked_csv, index=False)

        best_per_refit_rows.append(ranked.iloc[0].to_dict())

    if not best_per_refit_rows:
        msg = "No eligible epochs found anywhere under the current filters.\n"
        msg += f"Try lowering PREC_FLOOR (currently {PREC_FLOOR}) or relaxing RECALL_MIN/F1_MIN.\n"
        raise RuntimeError(msg)

    df_best_refit = pd.DataFrame(best_per_refit_rows).copy()

    # Top10 overall among best-per-refit
    df_best_refit = df_best_refit.sort_values(
        by=["score_total", "score_core", "precision", "f1", "recall", "fp", "fn", "val_logloss", "val_brier", "val_ece", "val_loss"],
        ascending=[False, False, False, False, False, True, True, True, True, True, True],
    ).reset_index(drop=True)

    top10 = df_best_refit.head(TOPK_OVERALL).copy()
    top10_csv = out_dir / "TOP10_overall_candidates_VALIDATION_ONLY.csv"
    top10.to_csv(top10_csv, index=False)

    # Best per model
    per_model_best_rows: List[Dict] = []
    for model in sorted(df_best_refit["model"].unique().tolist()):
        sub = df_best_refit[df_best_refit["model"] == model].copy()
        if sub.empty:
            continue
        per_model_best_rows.append(sub.iloc[0].to_dict())

    df_model_best = pd.DataFrame(per_model_best_rows)
    df_model_best = df_model_best.sort_values(
        by=["score_total", "score_core", "precision", "f1", "recall", "fp", "fn", "val_logloss", "val_brier", "val_ece", "val_loss"],
        ascending=[False, False, False, False, False, True, True, True, True, True, True],
    ).reset_index(drop=True)

    per_model_csv = out_dir / "best_epoch_per_model_balanced_VALIDATION_ONLY.csv"
    df_model_best.to_csv(per_model_csv, index=False)

    best_overall = df_model_best.iloc[0].to_dict()

    # -----------------------------
    # Comparison block: BEST vs ResNet34 epoch 18
    # -----------------------------
    compare_lines: List[str] = []
    best_row = _row_for_model_epoch(
        pool_scored,
        model=str(best_overall["model"]),
        epoch=int(best_overall["epoch"]),
        plot_dir=Path(str(best_overall["plot_dir"])) if str(best_overall.get("plot_dir", "")).strip() else None,
    )

    if best_row is None:
        compare_lines.append("COMPARISON BLOCK: Could not reconstruct BEST_OVERALL row from pool_scored (unexpected).")
    else:
        # For ResNet34 epoch 18, pick a specific plot_dir (first REFIT) to avoid ambiguity if multiple REFITs exist.
        res34_plot = _pick_first_resnet34_plotdir(refit_frames)
        res34_row = _row_for_model_epoch(pool_scored, "ResNet34", COMPARE_EPOCH, plot_dir=res34_plot)

        compare_lines.append("COMPARISON: BEST_OVERALL vs ResNet34 epoch 18 (VALIDATION ONLY)")
        compare_lines.append(f"  Eligibility: precision>={PREC_FLOOR:.3f}, recall>={RECALL_MIN:.3f}, f1>={F1_MIN:.3f} (STRICT={STRICT_ELIGIBILITY})")
        compare_lines.append("")

        compare_lines.append(_fmt_line("BEST_OVERALL", best_row))

        if res34_row is None:
            compare_lines.append(f"ResNet34 epoch {COMPARE_EPOCH:02d}: NOT FOUND in parsed validation files.")
            compare_lines.append("If you have multiple ResNet34 REFITs, ensure epoch 18 exists in confusion_matrix_per_epoch.txt.")
        else:
            compare_lines.append(_fmt_line(f"ResNet34_E{COMPARE_EPOCH:02d}", res34_row))
            compare_lines.append("")
            compare_lines.extend(_delta_lines(res34_row, best_row, f"Deltas (ResNet34_E{COMPARE_EPOCH:02d} minus BEST_OVERALL):"))

        compare_lines.append("")

    # -----------------------------
    # CLEAN REPORT
    # -----------------------------
    lines: List[str] = []
    lines.append("BEST MODEL + EPOCH SELECTION (VALIDATION ONLY)")
    lines.append(f"Run timestamp: {now}")
    lines.append("")
    lines.append("ELIGIBILITY (STRICT)")
    lines.append(f"  precision >= {PREC_FLOOR:.3f}")
    lines.append(f"  recall    >= {RECALL_MIN:.3f}")
    lines.append(f"  f1        >= {F1_MIN:.3f}")
    lines.append(f"  STRICT_ELIGIBILITY = {STRICT_ELIGIBILITY}")
    lines.append("")
    lines.append("SCORING")
    lines.append(f"  score_core  = {W_PREC}*precision + {W_F1}*f1 + {W_REC}*recall - {W_FP_RATE}*FP_rate - {W_FN_RATE}*FN_rate")
    lines.append(f"  val_penalty = {W_ECE}*ECE_n + {W_BRIER}*Brier_n + {W_LOGLOSS}*LogLoss_n + {W_LOSS}*Loss_n")
    lines.append("  score_total = score_core - val_penalty")
    lines.append("  FP_rate = FP/(FP+TN), FN_rate = FN/(FN+TP)")
    lines.append("  Validation metric normalization: robust min-max using p05..p95 across ALL epochs (clipped to [0,1])")
    lines.append(f"  REQUIRE_VAL_METRICS = {REQUIRE_VAL_METRICS}")
    lines.append("")
    lines.append("DISCOVERY SUMMARY")
    for m in sorted(RUN_ROOTS.keys()):
        lines.append(f"  {m}: found {model_refit_counts.get(m, 0)} plot_data dirs")
    if missing_models:
        lines.append("")
        lines.append("MODELS WITH MISSING VALIDATION FILES")
        for msg in missing_models:
            lines.append(f"  - {msg}")
    if excluded_refits:
        lines.append("")
        lines.append("REFITS EXCLUDED (NO ELIGIBLE EPOCHS UNDER CURRENT FILTERS)")
        for msg in excluded_refits:
            lines.append(f"  - {msg}")
    lines.append("")

    lines.append("BEST OVERALL (STRICT-ELIGIBLE)")
    lines.append(f"  Model    : {best_overall['model']}")
    lines.append(f"  REFIT    : {best_overall['refit_name']}")
    lines.append(f"  Epoch    : {int(best_overall['epoch']):02d}")
    lines.append(f"  Thr      : {float(best_overall['threshold_used']):.6f}")
    lines.append(
        f"  Metrics  : prec={float(best_overall['precision']):.6f}  rec={float(best_overall['recall']):.6f}  f1={float(best_overall['f1']):.6f}"
    )
    lines.append(
        f"  Counts   : TN={int(best_overall['tn'])} FP={int(best_overall['fp'])} FN={int(best_overall['fn'])} TP={int(best_overall['tp'])}"
    )
    lines.append(f"  Core     : {float(best_overall['score_core']):.6f}")
    lines.append(f"  Penalty  : {float(best_overall['val_penalty']):.6f}")
    lines.append(f"  Total    : {float(best_overall['score_total']):.6f}")
    lines.append(f"  Plot dir : {best_overall['plot_dir']}")
    lines.append("")

    lines.append(f"TOP {TOPK_OVERALL} OVERALL (BEST EPOCH PER REFIT)")
    top_print = top10.copy()
    top_print.insert(0, "rank", np.arange(1, len(top_print) + 1))

    for c in ["score_total", "score_core", "val_penalty", "precision", "recall", "f1", "threshold_used",
              "val_logloss", "val_brier", "val_ece", "val_loss"]:
        if c in top_print.columns:
            top_print[c] = top_print[c].apply(lambda x: fmt_float(x, 6))

    cols = [
        ("rank", "rank"),
        ("model", "model"),
        ("refit_name", "refit"),
        ("epoch", "ep"),
        ("threshold_used", "thr"),
        ("score_total", "score_total"),
        ("score_core", "core"),
        ("val_penalty", "pen"),
        ("precision", "prec"),
        ("recall", "rec"),
        ("f1", "f1"),
        ("fp", "FP"),
        ("fn", "FN"),
        ("val_logloss", "val_logloss"),
        ("val_brier", "val_brier"),
        ("val_ece", "val_ece"),
        ("val_loss", "val_loss"),
    ]
    cols = [(c, h) for c, h in cols if c in top_print.columns]
    lines.extend(make_table(top_print, cols, TOPK_OVERALL))
    lines.append("")

    lines.extend(compare_lines)

    report_path = out_dir / "BEST_MODEL_EPOCH_REPORT_VALIDATION_ONLY.txt"
    write_text(report_path, lines)

    print("")
    print("==============================================")
    print("DONE (VALIDATION ONLY, STRICT)")
    print(f"Outputs written to: {out_dir}")
    print(f"- {per_model_csv.name}")
    print(f"- {top10_csv.name}")
    print(f"- {report_path.name}")
    print("==============================================")
    print("")
    print("Best overall (validation, strict):")
    print(f"  {best_overall['model']} | {best_overall['refit_name']} | epoch {int(best_overall['epoch']):02d}")
    print(
        f"  score_total={float(best_overall['score_total']):.6f} core={float(best_overall['score_core']):.6f} "
        f"penalty={float(best_overall['val_penalty']):.6f} prec={float(best_overall['precision']):.6f} "
        f"rec={float(best_overall['recall']):.6f} f1={float(best_overall['f1']):.6f} "
        f"FP={int(best_overall['fp'])} FN={int(best_overall['fn'])}"
    )
    print("")
    print(f"Comparison written in report: BEST vs {COMPARE_MODEL} epoch {COMPARE_EPOCH}")


if __name__ == "__main__":
    main()
