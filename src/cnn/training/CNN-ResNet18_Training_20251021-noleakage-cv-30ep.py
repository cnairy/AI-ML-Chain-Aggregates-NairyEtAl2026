#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import copy
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_curve, auc,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# ------------------------------ #
#        GLOBAL SWITCHES         #
# ------------------------------ #
DO_HYPERPARAM_SEARCH   = False      # Grid over enabled dims (see *_ENABLE flags below)
DO_KFOLD               = True       # Stratified K-Fold over your 'train/' split (no leakage)
N_SPLITS               = 5
SELECT_BEST_BY         = "mean_F1_best"    # 'mean_val_loss' | 'mean_pr_auc' | 'mean_roc_auc' | 'mean_F1_best'
DO_REFIT_ON_FULL_TRAIN = True       # Refit best config on full train/ evaluate on original val/

# Repro
torch.backends.cudnn.benchmark = True

# ------------------------------ #
#       BASE CONFIG VALUES       #
# ------------------------------ #
data_dir      = '/path/to/nas_workspace/CNN/Addtional_CPI_Images/CPI_images_aug_split_noleakage/'
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(16)

# Epochs
num_epochs    = 30  # per run
# Operating constraints
THRESHOLD_FLOOR = 0.667
OPERATING_RULE  = "best_f1"  # or "max_precision_then_best_f1"

# Output base
today_str    = datetime.today().strftime('%Y%m%d')
base_out_dir = f'/path/to/nas_workspace/CNN/figures/30_epochs/JTech-ResNet18_training_evalutation_{today_str}-hyp-cv-30ep'
os.makedirs(base_out_dir, exist_ok=True)

# ---- NEW: runtime log setup ----
RUN_TIME_LOG = os.path.join(base_out_dir, 'run_time.log')
def _fmt_hms(seconds: float):
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log_runtime(section: str, start_ts: float, end_ts: float):
    dur = end_ts - start_ts
    with open(RUN_TIME_LOG, 'a') as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {section} | {dur:.3f} s ({_fmt_hms(dur)})\n")

# Write header at script entry
with open(RUN_TIME_LOG, 'a') as f:
    f.write("\n" + "="*72 + "\n")
    f.write(f"RUN START: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write(f"Device={device}, Epochs={num_epochs}, DO_KFOLD={DO_KFOLD}, DO_REFIT_ON_FULL_TRAIN={DO_REFIT_ON_FULL_TRAIN}\n")

# ------------------------------ #
#    EASY ON/OFF FOR GRID DIMS   #
# ------------------------------ #
LR_HEAD_ENABLE        = False
LR_BACKBONE_ENABLE    = True
WEIGHT_DECAY_ENABLE   = True
DROPOUT_ENABLE        = False
WARMUP_ENABLE         = True
UNFREEZE_PLAN_ENABLE  = True
OPTIMIZER_ENABLE      = False
SCHEDULER_ENABLE      = True
BATCH_SIZE_ENABLE     = False
LABEL_SMOOTH_ENABLE   = True
INIT_MODE_ENABLE      = True
USE_AUG_ENABLE        = False

# ------------------------------ #
#       DEFAULT (NO-SEARCH)      #
# ------------------------------ #
DEFAULT_HP = dict(
    learning_rate = 1e-4,
    backbone_lr   = 5e-5,
    weight_decay  = 2e-4,
    dropout       = 0.30,
    warmup_epochs = 6,
    unfreeze_plan = "layer3_4",    # "layer4" | "layer3_4" | "all"
    optimizer     = "adamw",       # "adamw" | "sgd_nesterov"
    scheduler     = "plateau",     # "plateau" | "cosine"
    batch_size    = 64,
    label_smooth  = 0.00,
    init_mode     = "luma",        # "mean" | "luma"
    use_online_aug= True
)

# ------------------------------ #
#     SMALL, SANE SEARCH RANGES  #
# ------------------------------ #
GRID_LR_HEAD       = [1e-4, 2e-4]
GRID_LR_BACKBONE   = [5e-5, 1e-5]
GRID_WEIGHT_DECAY  = [2e-4, 1e-4]
GRID_DROPOUT       = [0.30, 0.40]
GRID_WARMUP_EPOCHS = [3, 6]
GRID_UNFREEZE_PLAN = ["layer4", "layer3_4"]
GRID_OPTIMIZER     = ["adamw", "sgd_nesterov"]
GRID_SCHEDULER     = ["plateau", "cosine"]
GRID_BATCH_SIZE    = [32, 64, 128]
GRID_LABEL_SMOOTH  = [0.0, 0.05]
GRID_INIT_MODE     = ["mean", "luma"]
GRID_USE_AUG       = [True]

# ------------------------------ #
#        TRANSFORMS (BASE)       #
# ------------------------------ #
base_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])

mild_aug_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees=180, expand=True, fill=108),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.Normalize((0.485,), (0.229,))
])

# ------------------------------ #
#      UTILS / SMALL HELPERS     #
# ------------------------------ #
def save_points_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

def get_lrs(opt):
    return [g.get('lr', None) for g in opt.param_groups]

def metrics_at_threshold(y_true, y_prob, thr, eps=1e-8):
    preds = (y_prob >= thr).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def pr_best_thresholds(y_true, y_prob):
    precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_prob)
    eps = 1e-8
    if len(pr_thresholds) == 0:
        pr_thresholds = np.array([0.5], dtype=float)
        precision_arr = np.array([1.0, 1.0])
        recall_arr    = np.array([0.0, 1.0])
    P = precision_arr[:-1]
    R = recall_arr[:-1]
    T = pr_thresholds
    mask = T >= THRESHOLD_FLOOR
    if np.any(mask):
        Pm, Rm, Tm = P[mask], R[mask], T[mask]
        F1m = 2 * (Pm * Rm) / (Pm + Rm + eps)
        idx_bf = int(np.nanargmax(F1m))
        thr_bestF1 = float(Tm[idx_bf])
        prec_bestF1, rec_bestF1, bestF1_val = metrics_at_threshold(y_true, y_prob, thr_bestF1)
        maxP = np.max(Pm)
        idxs = np.where(np.isclose(Pm, maxP))[0]
        if idxs.size > 1:
            f1_sub = [metrics_at_threshold(y_true, y_prob, float(Tm[i]))[2] for i in idxs]
            idx_mp = idxs[int(np.argmax(f1_sub))]
        else:
            idx_mp = idxs[0]
        thr_precMax_bestF1 = float(Tm[idx_mp])
        prec_precMax, rec_precMax, f1_precMax = metrics_at_threshold(y_true, y_prob, thr_precMax_bestF1)
    else:
        thr_bestF1 = thr_precMax_bestF1 = THRESHOLD_FLOOR
        prec_bestF1, rec_bestF1, bestF1_val = metrics_at_threshold(y_true, y_prob, THRESHOLD_FLOOR)
        prec_precMax, rec_precMax, f1_precMax = prec_bestF1, rec_bestF1, bestF1_val
        P, R, T = np.array([]), np.array([]), np.array([])
    return dict(
        thr_bestF1=thr_bestF1, prec_bestF1=prec_bestF1, rec_bestF1=rec_bestF1, F1_best=bestF1_val,
        thr_precMax_bestF1=thr_precMax_bestF1, prec_precMax=prec_precMax, rec_precMax=rec_precMax, F1_precMax=f1_precMax,
        P=P, R=R, T=T
    )

def weighted_ece(probs, y_true, n_bins=10):
    probs = np.clip(probs, 1e-7, 1-1e-7)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i+1]
        m = (probs >= left) & ((probs < right) if i < n_bins-1 else (probs <= right))
        if not np.any(m):
            continue
        gap = abs(probs[m].mean() - y_true[m].mean())
        ece += gap * m.mean()
    return float(ece)

# ------------------------------ #
#         MODEL / OPT / SCHED    #
# ------------------------------ #
def build_model(dropout, init_mode="mean"):
    base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    old_w = base.conv1.weight.data.clone()  # [64,3,7,7]
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        if init_mode == "luma":
            w = old_w
            luma = 0.2989*w[:,0,:,:] + 0.5870*w[:,1,:,:] + 0.1140*w[:,2,:,:]
            new_conv1.weight[:,0,:,:] = luma
        else:
            new_conv1.weight[:] = old_w.mean(dim=1, keepdim=True)
    base.conv1 = new_conv1
    in_feats = base.fc.in_features
    base.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feats, 1))
    return base.to(device)

def freeze_for_warmup(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True

def unfreeze_and_add_params(module, optimizer, lr_backbone, wd):
    new_params = []
    for p in module.parameters():
        if not p.requires_grad:
            p.requires_grad = True
            new_params.append(p)
    if new_params:
        optimizer.add_param_group({'params': new_params, 'lr': lr_backbone, 'weight_decay': wd})

def apply_unfreeze_plan(model, optimizer, plan, lr_backbone, wd):
    if plan == "layer4":
        modules = [model.layer4]
    elif plan == "layer3_4":
        modules = [model.layer3, model.layer4]
    else:  # "all"
        modules = [model.layer1, model.layer2, model.layer3, model.layer4]
    for m in modules:
        unfreeze_and_add_params(m, optimizer, lr_backbone, wd)

def build_optimizer(model, hp):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if hp['optimizer'] == "sgd_nesterov":
        return optim.SGD(params, lr=hp['learning_rate'], momentum=0.9, nesterov=True, weight_decay=hp['weight_decay'])
    else:
        return optim.AdamW(params, lr=hp['learning_rate'], weight_decay=hp['weight_decay'])

def build_scheduler(optimizer, hp):
    if hp['scheduler'] == "cosine":
        T_max = max(1, num_epochs - hp['warmup_epochs'])
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
    else:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, threshold=5e-4,
            threshold_mode='abs', cooldown=1, min_lr=1e-6
        )

# ------------------------------ #
#        DATASET HELPERS         #
# ------------------------------ #
def build_single_split_loaders(hp):
    train_tf = mild_aug_transform if hp['use_online_aug'] else base_transform
    test_tf  = base_transform
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=test_tf)
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True,  num_workers=16, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=hp['batch_size'], shuffle=False, num_workers=8,  pin_memory=True)
    return train_loader, val_loader

def build_kfold_samplers(n_splits):
    base_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=None)
    y = [label for _, label in base_ds.samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(np.arange(len(y)), y))
    return base_ds, folds

def loaders_for_fold(base_ds, fold_indices, hp):
    tr_idx, va_idx = fold_indices
    train_tf = mild_aug_transform if hp['use_online_aug'] else base_transform
    test_tf  = base_transform
    train_ds_tf = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds_tf   = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=test_tf)
    train_subset = Subset(train_ds_tf, tr_idx)
    val_subset   = Subset(val_ds_tf,   va_idx)
    train_loader = DataLoader(train_subset, batch_size=hp['batch_size'], shuffle=True,  num_workers=16, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=hp['batch_size'], shuffle=False, num_workers=8,  pin_memory=True)
    return train_loader, val_loader

# ------------------------------ #
#     TRAIN/EVAL ONE FULL RUN    #
# ------------------------------ #
def train_one_run(run_name, train_loader, val_loader, hp, root_out_dir=base_out_dir):
    out_dir = os.path.join(root_out_dir, run_name)
    plot_dir = os.path.join(out_dir, 'plot_data')
    os.makedirs(plot_dir, exist_ok=True)
    log_file = os.path.join(out_dir, 'training_output.log')
    best_model_path = os.path.join(out_dir, 'cnn_chain_classifier_best_ResNet18.pt')
    operating_thr_file = os.path.join(out_dir, 'operating_threshold.txt')

    # Build model + optimizer + scheduler
    model = build_model(hp['dropout'], init_mode=hp['init_mode'])
    freeze_for_warmup(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, hp)
    scheduler = build_scheduler(optimizer, hp)

    # Histories
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []
    val_auc_hist, val_ap_hist      = [], []
    best_val_loss = float('inf')
    best_metrics = {'best_val_loss': float('inf'), 'val_roc_auc': float('nan'),
                    'val_pr_auc': float('nan'), 'thr_operating': None, 'F1_best': float('nan')}

    # CSV headers (includes NLL/Brier/ECE)
    scalar_csv = os.path.join(plot_dir, 'epoch_metrics.csv')
    with open(scalar_csv, 'w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['epoch','train_loss','val_loss','train_acc@0.5','val_acc@0.5',
                    'val_roc_auc','val_pr_auc','thr_bestF1','prec_bestF1','rec_bestF1','F1_best',
                    'thr_precMax_bestF1','prec_precMax','rec_precMax','F1_precMax',
                    'val_logloss','val_brier','val_ece'])

    # Header
    with open(log_file, 'a') as f:
        f.write(f"\n===== Training Session Started: {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        f.write(f"Device: {device}\nEpochs: {num_epochs}\nHP: {hp}\n")

    for epoch in range(num_epochs):
        if epoch == hp['warmup_epochs']:
            apply_unfreeze_plan(model, optimizer, hp['unfreeze_plan'], hp['backbone_lr'], hp['weight_decay'])

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc="Train Batches", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            if hp['label_smooth'] > 0:
                ls = hp['label_smooth']
                labels_smoothed = labels*(1.0 - ls) + 0.5*ls
            else:
                labels_smoothed = labels

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels_smoothed)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds_05 = (probs > 0.5).float()
                train_correct += (preds_05.eq(labels)).sum().item()
                train_total += labels.numel()

        train_loss /= len(train_loader)
        train_acc_05 = train_correct / max(1, train_total)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc_05)

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val Batches", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
                logits = model(imgs)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                val_loss += loss.item()
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        val_loss_hist.append(val_loss)
        y_prob = torch.cat(all_probs).numpy().flatten()
        y_true = torch.cat(all_labels).numpy().flatten()

        preds_05 = (y_prob > 0.5).astype(int)
        val_acc_05 = accuracy_score(y_true, preds_05)
        val_acc_hist.append(val_acc_05)

        thr_info = pr_best_thresholds(y_true, y_prob)
        operating_thr_epoch = (thr_info['thr_bestF1']
                               if OPERATING_RULE == "best_f1"
                               else thr_info['thr_precMax_bestF1'])
        final_preds = (y_prob >= operating_thr_epoch).astype(int)

        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
        except ValueError:
            roc_auc = float('nan'); fpr, tpr = np.array([0,1]), np.array([0,1])
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = float('nan')

        val_auc_hist.append(roc_auc)
        val_ap_hist.append(pr_auc)

        try:
            val_logloss = float(log_loss(y_true.astype(int), y_prob, labels=[0,1]))
        except ValueError:
            val_logloss = float('nan')
        try:
            val_brier = float(brier_score_loss(y_true.astype(int), y_prob))
        except ValueError:
            val_brier = float('nan')
        val_ece = float(weighted_ece(y_prob, y_true, n_bins=10))

        roc_csv = os.path.join(plot_dir, f'roc_epoch_{epoch+1:02d}.csv')
        save_points_csv(roc_csv, ['fpr','tpr'], list(zip(fpr, tpr)))
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        cal_csv = os.path.join(plot_dir, f'calibration_epoch_{epoch+1:02d}.csv')
        save_points_csv(cal_csv, ['prob_pred','prob_true'], list(zip(prob_pred, prob_true)))
        pr_csv = os.path.join(plot_dir, f'pr_epoch_{epoch+1:02d}.csv')
        save_points_csv(pr_csv, ['threshold','precision','recall'], list(zip(thr_info['T'], thr_info['P'], thr_info['R'])))

        with open(scalar_csv, 'a', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow([
                epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_acc_05:.6f}", f"{val_acc_05:.6f}",
                f"{roc_auc:.6f}", f"{pr_auc:.6f}",
                f"{thr_info['thr_bestF1']:.6f}", f"{thr_info['prec_bestF1']:.6f}", f"{thr_info['rec_bestF1']:.6f}", f"{thr_info['F1_best']:.6f}",
                f"{thr_info['thr_precMax_bestF1']:.6f}", f"{thr_info['prec_precMax']:.6f}", f"{thr_info['rec_precMax']:.6f}", f"{thr_info['F1_precMax']:.6f}",
                f"{val_logloss:.6f}", f"{val_brier:.6f}", f"{val_ece:.6f}"
            ])

        per_epoch_path = os.path.join(out_dir, f"model_epoch_{epoch+1:02d}_valloss_{val_loss:.6f}.pt")
        torch.save(model.state_dict(), per_epoch_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            with open(operating_thr_file, 'w') as f:
                f.write(f"rule={OPERATING_RULE}\nthreshold={operating_thr_epoch:.6f}\n")
            best_metrics = {'best_val_loss': float(val_loss), 'val_roc_auc': float(roc_auc),
                            'val_pr_auc': float(pr_auc), 'thr_operating': float(operating_thr_epoch),
                            'F1_best': float(thr_info['F1_best'])}

        cm_op = confusion_matrix(y_true, final_preds)
        fig, ax = plt.subplots(figsize=(8,6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_op, display_labels=['non-chains','T2_chains'])
        disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=True)
        plt.title(f"Val Confusion Matrix (op thr={operating_thr_epoch:.3f}) – Epoch {epoch+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'confusion_matrix_val_operating_epoch_{epoch+1:02d}.png'), dpi=300)
        plt.close(fig)

        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"Train Acc@0.5 {train_acc_05:.4f} | Val Acc@0.5 {val_acc_05:.4f}\n")
            f.write(f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}\n")
            f.write(f"Val ROC-AUC {roc_auc:.4f} | PR-AUC {pr_auc:.4f}\n")
            f.write(f"LogLoss (NLL) {val_logloss:.4f} | Brier {val_brier:.4f} | ECE {val_ece:.4f}\n")
            f.write(
                f"thr_bestF1={thr_info['thr_bestF1']:.4f} "
                f"(P={thr_info['prec_bestF1']:.4f}, R={thr_info['rec_bestF1']:.4f}, F1={thr_info['F1_best']:.4f})\n"
            )
            f.write(
                f"thr_precMax_bestF1={thr_info['thr_precMax_bestF1']:.4f} "
                f"(P={thr_info['prec_precMax']:.4f}, R={thr_info['rec_precMax']:.4f}, F1={thr_info['F1_precMax']:.4f})\n"
            )
            f.write(f"Operating thr ({OPERATING_RULE}): {operating_thr_epoch:.4f}\n\n")

        if hasattr(scheduler, 'step'):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                prev_lrs = get_lrs(optimizer); scheduler.step(val_loss); new_lrs = get_lrs(optimizer)
                if new_lrs != prev_lrs:
                    with open(log_file, 'a') as f: f.write(f"LR reduced: {prev_lrs} -> {new_lrs}\n\n")
            else:
                scheduler.step()

    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(train_loss_hist, label='Train'); plt.plot(val_loss_hist, label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('CNN Loss Curve')
    plt.savefig(os.path.join(out_dir, 'loss_curve.png')); plt.close()

    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(val_auc_hist, label='Val ROC-AUC')
    plt.xlabel('Epoch'); plt.ylabel('ROC-AUC'); plt.legend(); plt.title('Validation ROC-AUC Curve')
    plt.savefig(os.path.join(out_dir, 'val_auc_curve.png')); plt.close()

    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(val_ap_hist, label='Val PR-AUC')
    plt.xlabel('Epoch'); plt.ylabel('PR-AUC'); plt.legend(); plt.title('Validation PR-AUC Curve')
    plt.savefig(os.path.join(out_dir, 'val_prauc_curve.png')); plt.close()

    torch.save(model.state_dict(), os.path.join(out_dir, 'cnn_chain_classifier_ResNet18_final.pt'))
    return best_metrics, out_dir

# ------------------------------ #
#           MAIN DRIVER          #
# ------------------------------ #
if __name__ == "__main__":
    script_start = time.perf_counter()  # NEW: total timer start

    # Build the search grid respecting enable flags
    grids = dict(
        learning_rate = GRID_LR_HEAD       if LR_HEAD_ENABLE       else [DEFAULT_HP['learning_rate']],
        backbone_lr   = GRID_LR_BACKBONE   if LR_BACKBONE_ENABLE   else [DEFAULT_HP['backbone_lr']],
        weight_decay  = GRID_WEIGHT_DECAY  if WEIGHT_DECAY_ENABLE  else [DEFAULT_HP['weight_decay']],
        dropout       = GRID_DROPOUT       if DROPOUT_ENABLE       else [DEFAULT_HP['dropout']],
        warmup_epochs = GRID_WARMUP_EPOCHS if WARMUP_ENABLE        else [DEFAULT_HP['warmup_epochs']],
        unfreeze_plan = GRID_UNFREEZE_PLAN if UNFREEZE_PLAN_ENABLE else [DEFAULT_HP['unfreeze_plan']],
        optimizer     = GRID_OPTIMIZER     if OPTIMIZER_ENABLE     else [DEFAULT_HP['optimizer']],
        scheduler     = GRID_SCHEDULER     if SCHEDULER_ENABLE     else [DEFAULT_HP['scheduler']],
        batch_size    = GRID_BATCH_SIZE    if BATCH_SIZE_ENABLE    else [DEFAULT_HP['batch_size']],
        label_smooth  = GRID_LABEL_SMOOTH  if LABEL_SMOOTH_ENABLE  else [DEFAULT_HP['label_smooth']],
        init_mode     = GRID_INIT_MODE     if INIT_MODE_ENABLE     else [DEFAULT_HP['init_mode']],
        use_online_aug= GRID_USE_AUG       if USE_AUG_ENABLE       else [DEFAULT_HP['use_online_aug']],
    )

    if DO_HYPERPARAM_SEARCH:
        keys, values = zip(*grids.items())
        hp_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        hp_configs = [copy.deepcopy(DEFAULT_HP)]

    if not DO_KFOLD:
        # Single-split path (no CV block)
        hp = hp_configs[0]
        run_name = ("single_split_"
                    f"lrh{hp['learning_rate']}_lrb{hp['backbone_lr']}_wd{hp['weight_decay']}"
                    f"_do{hp['dropout']}_wu{hp['warmup_epochs']}_{hp['unfreeze_plan']}"
                    f"_{hp['optimizer']}_{hp['scheduler']}_bs{hp['batch_size']}"
                    f"_ls{hp['label_smooth']}_{hp['init_mode']}_aug{int(hp['use_online_aug'])}")
        # Treat this as the "final model" timing section
        final_refit_start = time.perf_counter()
        train_loader, val_loader = build_single_split_loaders(hp)
        metrics, out_dir = train_one_run(run_name, train_loader, val_loader, hp, root_out_dir=base_out_dir)
        final_refit_end = time.perf_counter()
        log_runtime("FINAL_MODEL_REFIT_TOTAL", final_refit_start, final_refit_end)

        print("Done single split. Best metrics:", metrics)

    else:
        # K-Fold over TRAIN split only
        cv_block_start = time.perf_counter()  # NEW: CV timer start
        base_ds, folds = build_kfold_samplers(N_SPLITS)
        summary_path = os.path.join(base_out_dir, f'cv_summary_{N_SPLITS}fold.csv')
        with open(summary_path, 'w', newline='') as fsum:
            w = csv.writer(fsum)
            w.writerow([
                'config_tag','learning_rate','backbone_lr','weight_decay','dropout',
                'warmup_epochs','unfreeze_plan','optimizer','scheduler','batch_size',
                'label_smooth','init_mode','use_online_aug',
                'mean_val_loss','mean_roc_auc','mean_pr_auc','mean_F1_best'
            ])

        best_config = None
        best_score = -np.inf if SELECT_BEST_BY != 'mean_val_loss' else np.inf

        for ci, hp in enumerate(hp_configs, start=1):
            fold_metrics = []
            for k, fold_idx in enumerate(folds, start=1):
                run_tag = (f"cvHP{ci}_fold{k}_lrh{hp['learning_rate']}_lrb{hp['backbone_lr']}"
                           f"_wd{hp['weight_decay']}_do{hp['dropout']}_wu{hp['warmup_epochs']}"
                           f"_{hp['unfreeze_plan']}_{hp['optimizer']}_{hp['scheduler']}"
                           f"_bs{hp['batch_size']}_ls{hp['label_smooth']}_{hp['init_mode']}"
                           f"_aug{int(hp['use_online_aug'])}")
                train_loader, val_loader = loaders_for_fold(base_ds, fold_idx, hp)
                metrics, out_dir = train_one_run(run_tag, train_loader, val_loader, hp, root_out_dir=base_out_dir)
                fold_metrics.append(metrics)
                print(f"[CONFIG {ci}/{len(hp_configs)}] Fold {k}/{N_SPLITS} done. {metrics}")

            # Aggregate
            mean_val_loss = float(np.mean([m['best_val_loss'] for m in fold_metrics]))
            mean_roc_auc  = float(np.nanmean([m['val_roc_auc']  for m in fold_metrics]))
            mean_pr_auc   = float(np.nanmean([m['val_pr_auc']   for m in fold_metrics]))
            mean_F1_best  = float(np.nanmean([m['F1_best']      for m in fold_metrics]))

            config_tag = (f"HP{ci}_lrh{hp['learning_rate']}_lrb{hp['backbone_lr']}_wd{hp['weight_decay']}"
                          f"_do{hp['dropout']}_wu{hp['warmup_epochs']}_{hp['unfreeze_plan']}"
                          f"_{hp['optimizer']}_{hp['scheduler']}_bs{hp['batch_size']}"
                          f"_ls{hp['label_smooth']}_{hp['init_mode']}_aug{int(hp['use_online_aug'])}")

            with open(summary_path, 'a', newline='') as fsum:
                w = csv.writer(fsum)
                w.writerow([
                    config_tag, hp['learning_rate'], hp['backbone_lr'], hp['weight_decay'], hp['dropout'],
                    hp['warmup_epochs'], hp['unfreeze_plan'], hp['optimizer'], hp['scheduler'], hp['batch_size'],
                    hp['label_smooth'], hp['init_mode'], hp['use_online_aug'],
                    f"{mean_val_loss:.6f}", f"{mean_roc_auc:.6f}", f"{mean_pr_auc:.6f}", f"{mean_F1_best:.6f}"
                ])

            # Select best config by criterion
            if SELECT_BEST_BY == 'mean_val_loss':
                score = -mean_val_loss
            elif SELECT_BEST_BY == 'mean_pr_auc':
                score = mean_pr_auc
            elif SELECT_BEST_BY == 'mean_roc_auc':
                score = mean_roc_auc
            else:
                score = mean_F1_best

            if score > best_score:
                best_score = score
                best_config = (hp, config_tag)

        print("=== CV complete ===")
        print("Best config by", SELECT_BEST_BY, ":", best_config)
        cv_block_end = time.perf_counter()  # NEW: CV timer end
        log_runtime("CROSS_VALIDATION_TOTAL", cv_block_start, cv_block_end)

        # Optional refit on full TRAIN and evaluate on original VAL
        if DO_REFIT_ON_FULL_TRAIN and best_config is not None:
            hp, cfg_tag = best_config
            final_refit_start = time.perf_counter()  # NEW: final model timer start

            train_tf = mild_aug_transform if hp['use_online_aug'] else base_transform
            full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
            val_ds     = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=base_transform)
            train_loader = DataLoader(full_train, batch_size=hp['batch_size'], shuffle=True,  num_workers=16, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=hp['batch_size'], shuffle=False, num_workers=8,  pin_memory=True)

            run_name = f"REFIT_FULLTRAIN_best_{cfg_tag}"
            metrics, out_dir = train_one_run(run_name, train_loader, val_loader, hp, root_out_dir=base_out_dir)
            print("Refit on full train finished. Metrics:", metrics)

            final_refit_end = time.perf_counter()  # NEW: final model timer end
            log_runtime("FINAL_MODEL_REFIT_TOTAL", final_refit_start, final_refit_end)

    # ---- total runtime log ----
    script_end = time.perf_counter()
    log_runtime("FULL_PIPELINE_TOTAL", script_start, script_end)
    with open(RUN_TIME_LOG, 'a') as f:
        f.write(f"RUN END:   {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("="*72 + "\n")
