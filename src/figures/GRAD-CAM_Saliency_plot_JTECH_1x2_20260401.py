#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# ==== PATHS ====
model_path = '/path/to/nas_workspace/CNN/figures/30_epochs/JTech-ResNet34_training_evalutation_20251021d-hyp-cv-30ep/REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_wu6_layer3_4_adamw_plateau_bs64_ls0.0_luma_aug1/model_epoch_18_valloss_0.014471.pt'

# image_path = '/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200125_133603/Analysis/CNN-ResNet34_trial2/CNN_T2_chains/20200125/IMPACTS_HawkeyeCPI_20200125204257204313205_001441_C1.png'
# image_path = '/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200125_133603/Hawkeye-CPI_C1_Images/20200125193836_C1/IMPACTS_HawkeyeCPI_20200125193836193839378_000419_C1.png'
# image_path = '/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200125_133603/Analysis/CNN-ResNet34_noleakage_JTECH-epoch18/CNN_T2_chains/IMPACTS_HawkeyeCPI_20200125183211192101111_010724_C1.png'
# image_path = '/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200125_133603/Analysis/CNN-ResNet18_noleakage_JTECH-epoch22/CNN_T2_chains/IMPACTS_HawkeyeCPI_20200125194040194224895_006944_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2022/20220117_080140/CNN_T2_chains/IMPACTS_HawkeyeCPI_20220117113441113532604_001431_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2020/20200125_133603/CNN_T2_chains/IMPACTS_HawkeyeCPI_20200125223056223059842_000306_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2020/20200220_192119/CNN_T2_chains/IMPACTS_HawkeyeCPI_20200220213859213900239_000083_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2023/20230209_172935/CNN_T2_chains/IMPACTS_HawkeyeCPI_20230209204506204606327_015014_C1.png'
image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2023/20230214_211703/CNN_T2_chains/IMPACTS_HawkeyeCPI_20230215011358011404335_000237_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2023/20230113_041654/CNN_T2_chains/IMPACTS_HawkeyeCPI_20230113073203073907139_003532_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2023/20230129_133405/CNN_T2_chains/IMPACTS_HawkeyeCPI_20230129155323155344168_003130_C1.png'
# image_path = '/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS/2023/20230129_133405/CNN_T2_chains/IMPACTS_HawkeyeCPI_20230129141213141545612_005240_C1.png'

# ==== EASY TITLE CONTROL ====
particle_title = "Dendrite"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== INPUT TRANSFORM (match training) ====
tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,)),
])

# ==== LOAD IMAGE ====
pil_img = Image.open(image_path).convert('L')

# Raw original image for top display only
orig_gray = np.array(pil_img)

# 224x224 image for model-space attribution
pil_bg = pil_img.resize((224, 224), Image.BILINEAR)
bg_gray = np.array(pil_bg).astype(np.float32) / 255.0

# Tensor for model
input_tensor = tfm(pil_img).unsqueeze(0).to(device)
input_tensor.requires_grad_(True)

# ==== BUILD MODEL LIKE TRAINING ====
def build_trained_resnet34(dropout_p=0.30, init_mode="luma"):
    base = models.resnet34(weights=ResNet34_Weights.DEFAULT)

    old_w = base.conv1.weight.data.clone()
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        if init_mode == "luma":
            luma = 0.2989 * old_w[:, 0, :, :] + 0.5870 * old_w[:, 1, :, :] + 0.1140 * old_w[:, 2, :, :]
            new_conv1.weight[:, 0, :, :] = luma
        else:
            new_conv1.weight[:] = old_w.mean(dim=1, keepdim=True)

    base.conv1 = new_conv1
    in_feats = base.fc.in_features
    base.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_feats, 1))
    return base

model = build_trained_resnet34(dropout_p=0.30, init_mode="luma").to(device)

# ==== LOAD CHECKPOINT ====
ckpt = torch.load(model_path, map_location=device)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
if missing or unexpected:
    print("[load_state_dict] missing keys:", missing)
    print("[load_state_dict] unexpected keys:", unexpected)
model.eval()

# ================================
#         GRAD x INPUT MAP
# ================================
model.zero_grad(set_to_none=True)
if input_tensor.grad is not None:
    input_tensor.grad.zero_()

logit = model(input_tensor)[0, 0]
prob = torch.sigmoid(logit).item()
logit.backward()

print(f"logit: {logit.item():.6f}")
print(f"probability: {prob:.6f}")

input_grad = input_tensor.grad.detach()[0, 0].cpu().numpy()
input_img_norm = input_tensor.detach()[0, 0].cpu().numpy()

grad_x_input = np.abs(input_grad * input_img_norm)
grad_x_input = gaussian_filter(grad_x_input, sigma=0.8)

g_low, g_high = np.percentile(grad_x_input, [1, 99.5])
g_den = max(g_high - g_low, 1e-12)
grad_x_input_norm = np.clip((grad_x_input - g_low) / g_den, 0, 1)

# ==== PLOT ====
fig, axes = plt.subplots(
    2, 1,
    figsize=(2.8, 5.8),
    gridspec_kw={'height_ratios': [1, 1]}
)

plt.subplots_adjust(
    left=0.24,
    right=0.98,
    top=0.83,
    bottom=0.04,
    hspace=0.08
)

# Top: raw original image
axes[0].imshow(orig_gray, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
# axes[0].set_ylabel("Original Image", fontsize=16, rotation=90, labelpad=6)
axes[0].set_xticks([])
axes[0].set_yticks([])
for spine in axes[0].spines.values():
    spine.set_visible(False)

# Bottom: Grad × Input only
axes[1].imshow(bg_gray, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
axes[1].imshow(grad_x_input_norm, cmap='hot', alpha=0.60, interpolation='nearest')
# axes[1].set_ylabel("Saliency Map", fontsize=16, rotation=90, labelpad=6)
axes[1].set_xticks([])
axes[1].set_yticks([])
for spine in axes[1].spines.values():
    spine.set_visible(False)

# Position title/subtitle relative to top image panel
fig.canvas.draw()
top_bbox = axes[0].get_position()
subplot_center_x = (top_bbox.x0 + top_bbox.x1) / 2.0
title_y = top_bbox.y1 + 0.040
prob_y  = top_bbox.y1 + 0.013

fig.text(
    subplot_center_x,
    title_y,
    particle_title,
    ha='center',
    va='bottom',
    fontsize=16
)

fig.text(
    subplot_center_x,
    prob_y,
    f"P = {prob:.2f}",
    ha='center',
    va='bottom',
    fontsize=13
)

# ==== SAVE FIGURE ====
safe_particle_title = "".join(
    c if c.isalnum() or c in (" ", "_", "-") else "_" for c in particle_title
).strip().replace(" ", "_")

out_name = f"{safe_particle_title}_explainability.png"
plt.savefig(out_name, dpi=300, bbox_inches='tight')

print(f"Saved figure to: {out_name}")

plt.show()