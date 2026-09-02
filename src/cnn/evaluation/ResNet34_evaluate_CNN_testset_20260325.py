#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, ConfusionMatrixDisplay, auc
)

# ========== CONFIGURATION ==========
today_str = datetime.today().strftime('%Y%m%d')

test_dir = '/path/to/nas_workspace/CNN/Addtional_CPI_Images/testing_dataset_20220117/'
model_path = (
    '/path/to/nas_workspace/CNN/figures/30_epochs/'
    'JTech-ResNet34_training_evalutation_20251021d-hyp-cv-30ep/'
    'REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/'
    'model_epoch_18_valloss_0.014471.pt'
)

output_dir = f'./eval_outputs_ResNet34_{today_str}/'
misclass_dir = f'/path/to/nas_workspace/CNN/Addtional_CPI_Images/misclassified_chains_testset_ResNet34_{today_str}/'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(misclass_dir, exist_ok=True)
os.makedirs(os.path.join(misclass_dir, 'false_pos'), exist_ok=True)
os.makedirs(os.path.join(misclass_dir, 'false_neg'), exist_ok=True)
os.makedirs(os.path.join(misclass_dir, 'true_pos'), exist_ok=True)
os.makedirs(os.path.join(misclass_dir, 'true_neg'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])

# ========== ENFORCE CORRECT CLASS MAPPING ==========
desired_class_to_idx = {'0_non-chains': 0, '1_T2_chains': 1}
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_dataset.class_to_idx = desired_class_to_idx
test_dataset.classes = ['0_non-chains', '1_T2_chains']

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
image_paths = [s[0] for s in test_dataset.samples]
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# ========== LOAD RESNET34 (MATCH TRAINING ARCH) ==========
model = models.resnet34(weights=None)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

in_feats = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.30),
    torch.nn.Linear(in_feats, 1)
)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ========== INFERENCE ==========
all_labels, all_probs, all_preds = [], [], []
misclassified_info = []

OPERATING_THRESHOLD = 0.671

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > OPERATING_THRESHOLD).astype(int)

        start_idx = i * test_loader.batch_size
        for j in range(len(labels)):
            label = labels[j].item()
            prob = probs[j]
            pred = preds[j]
            path = image_paths[start_idx + j]

            all_labels.append(label)
            all_probs.append(prob)
            all_preds.append(pred)

            if pred != label:
                if pred == 1 and label == 0:
                    shutil.copy(path, os.path.join(misclass_dir, 'false_pos', os.path.basename(path)))
                    misclassified_info.append(("False Positive", path))
                elif pred == 0 and label == 1:
                    shutil.copy(path, os.path.join(misclass_dir, 'false_neg', os.path.basename(path)))
                    misclassified_info.append(("False Negative", path))
            else:
                if pred == 1 and label == 1:
                    shutil.copy(path, os.path.join(misclass_dir, 'true_pos', os.path.basename(path)))
                elif pred == 0 and label == 0:
                    shutil.copy(path, os.path.join(misclass_dir, 'true_neg', os.path.basename(path)))

# ========== METRICS ==========
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_preds = np.array(all_preds)

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc_score = roc_auc_score(all_labels, all_probs)

# ========== PRECISION-RECALL DATA ==========
precision_arr, recall_arr, thresholds = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall_arr, precision_arr)

# ========== BEST THRESHOLD (FROM TEST SET PR CURVE) ==========
f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold for F1 (test set): {best_threshold:.4f}")
print(f"Precision: {precision_arr[best_idx]:.4f}, Recall: {recall_arr[best_idx]:.4f}, F1: {f1_scores[best_idx]:.4f}")
print(f"ROC-AUC: {auc_score:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(classification_report(all_labels, all_preds, target_names=['non-chains', 'T2_chains']))

# ========== SAVE TEXT METRICS ==========
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    f.write("=== Classification Report ===\n")
    f.write(classification_report(all_labels, all_preds, target_names=['non-chains', 'T2_chains']))
    f.write("\n=== Confusion Matrix ===\n")
    f.write(str(confusion_matrix(all_labels, all_preds)))
    f.write(
        f"\n\nAccuracy: {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall: {rec:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"ROC-AUC: {auc_score:.4f}\n"
        f"PR-AUC: {pr_auc:.4f}\n"
        f"Operating Threshold: {OPERATING_THRESHOLD:.3f}\n"
        f"Best Test-Set F1 Threshold: {best_threshold:.4f}\n"
        f"Best Test-Set Precision: {precision_arr[best_idx]:.4f}\n"
        f"Best Test-Set Recall: {recall_arr[best_idx]:.4f}\n"
        f"Best Test-Set F1: {f1_scores[best_idx]:.4f}\n"
    )
    f.write("\n=== Misclassified Examples ===\n")
    for typ, path in misclassified_info:
        f.write(f"{typ},{path}\n")

# ========== ROC CURVE ==========
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='orange', label=f"ROC-AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
plt.close()

# ========== CONFUSION MATRIX ==========
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-chains", "Chains"])
disp.plot(cmap='Blues', values_format='d')
for text in disp.text_.ravel():
    text.set_fontsize(15)

disp.ax_.set_xlabel("Predicted label", fontsize=13)
disp.ax_.set_ylabel("True label", fontsize=13)
disp.ax_.tick_params(axis='both', labelsize=13)

plt.title("Confusion Matrix", fontsize = 13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
plt.close()

# ========== PRECISION-RECALL CURVE ==========
plt.figure(figsize=(8, 5))
plt.plot(recall_arr, precision_arr, color='orange', label=f"PR-AUC = {pr_auc:.4f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-AUC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
plt.close()

# ========== PREDICTION PROBABILITY HISTOGRAM ==========
plt.figure(figsize=(8, 5))
plt.hist(
    [all_probs[i] for i in range(len(all_probs)) if all_labels[i] == 0],
    bins=50, alpha=0.6, label='Non-chains', color='tab:blue'
)
plt.hist(
    [all_probs[i] for i in range(len(all_probs)) if all_labels[i] == 1],
    bins=50, alpha=0.6, label='T2_chains', color='tab:orange'
)
plt.axvline(x=OPERATING_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {OPERATING_THRESHOLD:.3f}')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.yscale('log')
plt.title('Prediction Probability Histogram')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_histogram.png'), dpi=300)
plt.close()

# ========== NORMALIZED PREDICTION PROBABILITY HISTOGRAM ==========
plt.figure(figsize=(8, 5))
plt.hist(
    [all_probs[i] for i in range(len(all_probs)) if all_labels[i] == 0],
    bins=50, alpha=0.6, label='Non-chains', color='tab:blue', density=True
)
plt.hist(
    [all_probs[i] for i in range(len(all_probs)) if all_labels[i] == 1],
    bins=50, alpha=0.6, label='T2_chains', color='tab:orange', density=True
)
plt.axvline(x=OPERATING_THRESHOLD, color='red', linestyle='--', label=f'Threshold = {OPERATING_THRESHOLD:.3f}')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Normalized Prediction Probability Histogram')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'prediction_histogram_normalized.png'), dpi=300)
plt.close()

print(f"Evaluation complete. Metrics and figures saved to: {output_dir}")
print(f"Misclassified images saved to: {misclass_dir}")