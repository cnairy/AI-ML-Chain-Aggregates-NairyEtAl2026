#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script Name: external_validate_all_models-30ep_20260323.py

import csv
import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights,
    VGG16_Weights, VGG19_Weights
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, average_precision_score,
    accuracy_score, classification_report, log_loss, brier_score_loss
)

# --------------------------
# Config / constants
# --------------------------
THRESHOLD_FLOOR = 0.667
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization rules
RESNET_MEAN, RESNET_STD = 0.485, 0.229
VGG_MEAN, VGG_STD = 0.278238, 0.180433  # from training script

# Model name parsing patterns
MODEL_PATTERNS = {
    "ResNet18":  re.compile(r"ResNet18", re.I),
    "ResNet34":  re.compile(r"ResNet34", re.I),
    "ResNet50":  re.compile(r"ResNet50", re.I),
    "ResNet101": re.compile(r"ResNet101", re.I),
    "VGG16":     re.compile(r"VGG-?16", re.I),
    "VGG19":     re.compile(r"VGG-?19", re.I),
}


# --------------------------
# Utilities
# --------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def soft_sigmoid(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).cpu().numpy().reshape(-1)


def infer_model_name_from_dir(d: Path) -> str:
    s = d.name
    for name, pat in MODEL_PATTERNS.items():
        if pat.search(s):
            return name
    return s


def list_candidate_model_dirs(base: Path) -> List[Path]:
    return [p for p in base.iterdir() if p.is_dir() and p.name.startswith("JTech-")]


def list_fold_dirs(model_dir: Path) -> List[Path]:
    out = []
    for p in model_dir.iterdir():
        if p.is_dir() and p.name.startswith("cvHP"):
            out.append(p)
    return sorted(out)


def load_operating_threshold(fold_dir: Path, default_thr: float) -> float:
    f = fold_dir / "operating_threshold.txt"
    if not f.exists():
        return float(default_thr)
    try:
        txt = f.read_text().strip().splitlines()
        for line in txt:
            if line.lower().startswith("threshold="):
                return float(line.split("=", 1)[1].strip())
    except Exception:
        pass
    return float(default_thr)


def pr_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thr_floor=THRESHOLD_FLOOR) -> Dict[str, float]:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    eps = 1e-8

    if thr.size == 0:
        thr = np.array([0.5], dtype=float)
        prec = np.array([1.0, 1.0])
        rec = np.array([0.0, 1.0])

    P = prec[:-1]
    R = rec[:-1]
    T = thr

    mask = T >= thr_floor
    if not np.any(mask):
        Pm = np.array([P.max() if P.size else 0.0])
        Rm = np.array([R.max() if R.size else 0.0])
        Tm = np.array([thr_floor])
    else:
        Pm, Rm, Tm = P[mask], R[mask], T[mask]

    F1m = 2 * (Pm * Rm) / (Pm + Rm + eps)
    idx = int(np.nanargmax(F1m))
    thr_best = float(Tm[idx])

    return {"thr_bestF1": thr_best}


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    preds = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, preds)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    try:
        nll = float(log_loss(y_true.astype(int), y_prob, labels=[0, 1]))
    except ValueError:
        nll = float("nan")

    try:
        brier = float(brier_score_loss(y_true.astype(int), y_prob))
    except ValueError:
        brier = float("nan")

    rep = classification_report(
        y_true,
        preds,
        labels=[0, 1],
        target_names=["0_non-chains", "1_T2_chains"],
        zero_division=0,
        output_dict=True
    )

    prec = float(rep["weighted avg"]["precision"])
    rec = float(rep["weighted avg"]["recall"])
    f1 = float(rep["weighted avg"]["f1-score"])

    return dict(
        acc=acc,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=prec,
        recall=rec,
        f1=f1,
        logloss=nll,
        brier=brier,
    )


# --------------------------
# Model builders (match training heads)
# --------------------------
def make_resnet(which: str, dropout: float = 0.30):
    weights_map = {
        "ResNet18": ResNet18_Weights.DEFAULT,
        "ResNet34": ResNet34_Weights.DEFAULT,
        "ResNet50": ResNet50_Weights.DEFAULT,
        "ResNet101": ResNet101_Weights.DEFAULT,
    }
    base = getattr(models, which.lower())(weights=weights_map[which])

    old_w = base.conv1.weight.data.clone()
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        w = old_w
        luma = 0.2989 * w[:, 0, :, :] + 0.5870 * w[:, 1, :, :] + 0.1140 * w[:, 2, :, :]
        new_conv1.weight[:, 0, :, :] = luma
    base.conv1 = new_conv1

    in_feats = base.fc.in_features
    base.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feats, 1))
    return base.to(DEVICE)


def make_vgg(which: str, dropout: float = 0.30):
    weights_map = {
        "VGG16": VGG16_Weights.DEFAULT,
        "VGG19": VGG19_Weights.DEFAULT,
    }
    base = getattr(models, which.lower())(weights=weights_map[which])

    old_w = base.features[0].weight.data.clone()
    new_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    with torch.no_grad():
        w = old_w
        luma = 0.2989 * w[:, 0, :, :] + 0.5870 * w[:, 1, :, :] + 0.1140 * w[:, 2, :, :]
        new_conv1.weight[:, 0, :, :] = luma
    base.features[0] = new_conv1

    base.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512 * 7 * 7, 1))
    return base.to(DEVICE)


def build_model(model_name: str):
    if model_name.startswith("ResNet"):
        return make_resnet(model_name)
    if model_name.startswith("VGG"):
        return make_vgg(model_name)
    raise ValueError(f"Unknown model: {model_name}")


def make_transform(model_name: str):
    if model_name.startswith("ResNet"):
        mean, std = RESNET_MEAN, RESNET_STD
    else:
        mean, std = VGG_MEAN, VGG_STD

    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])


# --------------------------
# Evaluation
# --------------------------
def eval_one_fold(
    model_name: str,
    fold_dir: Path,
    val_root: Path,
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Loads best checkpoint from `fold_dir`, evaluates on `val_root`.
    Returns dict of per-fold metrics & thresholds.
    """
    ckpt = None
    for p in fold_dir.iterdir():
        if p.is_file() and p.name.startswith("cnn_chain_classifier_best_") and p.name.endswith(".pt"):
            ckpt = p
            break
    if ckpt is None:
        raise FileNotFoundError(f"No best checkpoint in {fold_dir}")

    tf = make_transform(model_name)
    val_ds = datasets.ImageFolder(str(val_root), transform=tf)
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = build_model(model_name)
    state = torch.load(str(ckpt), map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_labels.append(y.numpy().reshape(-1))

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels).astype(int)

    thr_info = pr_best_thresholds(y_true, y_prob)
    thr_best = float(thr_info["thr_bestF1"])
    thr_op = load_operating_threshold(fold_dir, default_thr=thr_best)
    thr_floor = float(THRESHOLD_FLOOR)

    m_best = metrics_at_threshold(y_true, y_prob, thr_best)
    m_op = metrics_at_threshold(y_true, y_prob, thr_op)
    m_floor = metrics_at_threshold(y_true, y_prob, thr_floor)

    m_best.update({"thr": thr_best})
    m_op.update({"thr": thr_op})
    m_floor.update({"thr": thr_floor})

    return dict(
        model=model_name,
        fold=fold_dir.name,
        y_true=y_true,
        y_prob=y_prob,
        best=m_best,
        op=m_op,
        floor=m_floor,
    )


def write_headers_if_needed(p: Path, header: List[str]):
    if not p.exists():
        with open(p, "w", newline="") as f:
            csv.writer(f).writerow(header)


def main(args):
    base = Path(args.base)
    val_root = Path(args.val)
    out_dir = base / "comparisons_30ep" / "external_val_summaries_30ep_20260323"
    ensure_dir(out_dir)

    per_fold_csv = out_dir / "per_fold_summary.csv"
    per_class_best_csv = out_dir / "per_class_by_fold_bestF1.csv"
    per_class_op_csv = out_dir / "per_class_by_fold_opthr.csv"
    per_class_floor_csv = out_dir / "per_class_by_fold_floor.csv"

    write_headers_if_needed(per_fold_csv, [
        "model", "fold",
        "thr_bestF1", "acc@bestF1", "prec@bestF1", "rec@bestF1", "f1@bestF1", "pr_auc@bestF1", "roc_auc@bestF1", "logloss@bestF1", "brier@bestF1",
        "thr_opthr", "acc@opthr", "prec@opthr", "rec@opthr", "f1@opthr", "pr_auc@opthr", "roc_auc@opthr", "logloss@opthr", "brier@opthr",
        "thr_floor", "acc@floor", "prec@floor", "rec@floor", "f1@floor", "pr_auc@floor", "roc_auc@floor", "logloss@floor", "brier@floor"
    ])
    write_headers_if_needed(per_class_best_csv, ["model", "fold", "class", "precision", "recall", "f1", "support", "threshold_type"])
    write_headers_if_needed(per_class_op_csv, ["model", "fold", "class", "precision", "recall", "f1", "support", "threshold_type"])
    write_headers_if_needed(per_class_floor_csv, ["model", "fold", "class", "precision", "recall", "f1", "support", "threshold_type"])

    model_dirs = list_candidate_model_dirs(base)
    if not model_dirs:
        raise RuntimeError(f"No JTech-* model folders found under {base}")

    for mdir in model_dirs:
        model_name = infer_model_name_from_dir(mdir)
        print(f"\n=== {model_name} :: {mdir.name} ===")
        fold_dirs = list_fold_dirs(mdir)
        if not fold_dirs:
            print("  (no cvHP* fold dirs found)")
            continue

        for fd in fold_dirs:
            try:
                res = eval_one_fold(model_name, fd, val_root, batch_size=args.batch)
            except Exception as e:
                print(f"  [SKIP] {fd.name}: {e}")
                continue

            with open(per_fold_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    model_name, fd.name,

                    f"{res['best']['thr']:.6f}",
                    f"{res['best']['acc']:.6f}",
                    f"{res['best']['precision']:.6f}",
                    f"{res['best']['recall']:.6f}",
                    f"{res['best']['f1']:.6f}",
                    f"{res['best']['pr_auc']:.6f}",
                    f"{res['best']['roc_auc']:.6f}",
                    f"{res['best']['logloss']:.6f}",
                    f"{res['best']['brier']:.6f}",

                    f"{res['op']['thr']:.6f}",
                    f"{res['op']['acc']:.6f}",
                    f"{res['op']['precision']:.6f}",
                    f"{res['op']['recall']:.6f}",
                    f"{res['op']['f1']:.6f}",
                    f"{res['op']['pr_auc']:.6f}",
                    f"{res['op']['roc_auc']:.6f}",
                    f"{res['op']['logloss']:.6f}",
                    f"{res['op']['brier']:.6f}",

                    f"{res['floor']['thr']:.6f}",
                    f"{res['floor']['acc']:.6f}",
                    f"{res['floor']['precision']:.6f}",
                    f"{res['floor']['recall']:.6f}",
                    f"{res['floor']['f1']:.6f}",
                    f"{res['floor']['pr_auc']:.6f}",
                    f"{res['floor']['roc_auc']:.6f}",
                    f"{res['floor']['logloss']:.6f}",
                    f"{res['floor']['brier']:.6f}",
                ])

            y_true = res["y_true"]
            y_prob = res["y_prob"]

            # Best-F1 per-class rows
            preds_b = (y_prob >= res["best"]["thr"]).astype(int)
            rep_b = classification_report(
                y_true,
                preds_b,
                labels=[0, 1],
                target_names=["0_non-chains", "1_T2_chains"],
                zero_division=0,
                output_dict=True
            )
            with open(per_class_best_csv, "a", newline="") as f:
                w = csv.writer(f)
                for cname in ["0_non-chains", "1_T2_chains"]:
                    d = rep_b[cname]
                    w.writerow([
                        model_name,
                        fd.name,
                        cname,
                        f"{d['precision']:.6f}",
                        f"{d['recall']:.6f}",
                        f"{d['f1-score']:.6f}",
                        int(d["support"]),
                        "bestF1"
                    ])

            # Operating-threshold per-class rows
            preds_o = (y_prob >= res["op"]["thr"]).astype(int)
            rep_o = classification_report(
                y_true,
                preds_o,
                labels=[0, 1],
                target_names=["0_non-chains", "1_T2_chains"],
                zero_division=0,
                output_dict=True
            )
            with open(per_class_op_csv, "a", newline="") as f:
                w = csv.writer(f)
                for cname in ["0_non-chains", "1_T2_chains"]:
                    d = rep_o[cname]
                    w.writerow([
                        model_name,
                        fd.name,
                        cname,
                        f"{d['precision']:.6f}",
                        f"{d['recall']:.6f}",
                        f"{d['f1-score']:.6f}",
                        int(d["support"]),
                        "opthr"
                    ])

            # Threshold-floor per-class rows
            preds_f = (y_prob >= res["floor"]["thr"]).astype(int)
            rep_f = classification_report(
                y_true,
                preds_f,
                labels=[0, 1],
                target_names=["0_non-chains", "1_T2_chains"],
                zero_division=0,
                output_dict=True
            )
            with open(per_class_floor_csv, "a", newline="") as f:
                w = csv.writer(f)
                for cname in ["0_non-chains", "1_T2_chains"]:
                    d = rep_f[cname]
                    w.writerow([
                        model_name,
                        fd.name,
                        cname,
                        f"{d['precision']:.6f}",
                        f"{d['recall']:.6f}",
                        f"{d['f1-score']:.6f}",
                        int(d["support"]),
                        "floor"
                    ])

            print(
                f"  {fd.name} — "
                f"bestF1 thr={res['best']['thr']:.3f}, "
                f"opthr={res['op']['thr']:.3f}, "
                f"floor={res['floor']['thr']:.3f}"
            )

    print(
        f"\nDone. Wrote:\n"
        f"  - {per_fold_csv}\n"
        f"  - {per_class_best_csv}\n"
        f"  - {per_class_op_csv}\n"
        f"  - {per_class_floor_csv}\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=str,
        required=True,
        help="Root figures dir with JTech-* folders (e.g., /nas/.../CNN/figures/30_epochs)"
    )
    ap.add_argument(
        "--val",
        type=str,
        required=True,
        help="External validation split root, with subfolders 0_non-chains/ and 1_T2_chains/"
    )
    ap.add_argument("--batch", type=int, default=64, help="Eval batch size (default 64)")
    args = ap.parse_args()
    main(args)