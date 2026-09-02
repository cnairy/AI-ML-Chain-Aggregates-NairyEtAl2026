#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn

# ============================================================
#                GLOBAL TOGGLES / CONFIG
# ============================================================

# Toggle: run a single flight vs all flights
RUN_ALL_FLIGHTS = True  # False = single flight, True = loop over all flights

# Base directory for all CNN prediction outputs (Paper 3)
BASE_OUTPUT_DIR = "/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS_Filtered"

# Model checkpoint (ResNet34) – adjust if you switch epoch/checkpoint
model_path = (
    "/path/to/nas_workspace/CNN/figures/30_epochs/"
    "JTech-ResNet34_training_evalutation_20251021d-hyp-cv-30ep/"
    "REFIT_FULLTRAIN_best_HP1_lrh0.0001_lrb5e-05_wd0.0002_do0.3_wu6_layer3_4_"
    "adamw_plateau_bs64_ls0.0_luma_aug1/model_epoch_18_valloss_0.014471.pt"
)

# Operating threshold for chain classification
THRESHOLD = 0.671

# ============================================================
#                   FLIGHT PATH DEFINITIONS
# ============================================================

# Single-flight mode: choose one flight here
SINGLE_FLIGHT_DIR = (
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/"
    "FlightData/20200125_133603/Hawkeye-CPI_C1_Images"
)

# All flights you want to loop over when RUN_ALL_FLIGHTS = True
ALL_FLIGHT_DIRS = [
    # ---------- 2020 ----------
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200118_130356/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200125_133603/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200201_060849/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200205_181400/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200207_134955/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200213_054812/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200218_170335/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200220_192119/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200224_173407/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData/20200225_1/Hawkeye-CPI_C1_Good_Images",

    # ---------- 2022 ----------
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220114_1/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220117_080140/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220119_105103/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220129_184039/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220203_130550/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220204_125652/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220208_1/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220213_113157/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220217_162158/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220219_115230/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2022/Aircraft/P3_N426NA/FlightData/20220225_1/Hawkeye-CPI_C1_Good_Images",

    # ---------- 2023 ----------
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230113_041654/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230115_1/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230119_202459/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230123_114903/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230125_184916/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230129_133405/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230205_133304/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230209_172935/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230212_132810/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230214_211703/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230216_150059/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230223_114748/Hawkeye-CPI_C1_Good_Images",
    "/path/to/impacts_workspace/IMPACTS2023/Aircraft/P3_N426NA/FlightData/20230228_090908/Hawkeye-CPI_C1_Good_Images",
]

# ============================================================
#                    DEVICE / MODEL SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(16)

# Build ResNet34 architecture matching training
model = models.resnet34(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
in_feats = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_feats, 1))  # logits

state = torch.load(model_path, map_location=device)
if not isinstance(state, dict) or "state_dict" in state:
    state = state.get("state_dict", state)

model.load_state_dict(state, strict=True)
model = model.to(device).eval()

# ============================================================
#                    TRANSFORMS / DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,)),
])

class CPIFiles(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = []
        for r, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(".png"):
                    self.paths.append(os.path.join(r, fn))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("L")
        x = self.transform(img) if self.transform else img
        fname = os.path.basename(path)
        return x, path, fname

# ============================================================
#                         HELPERS
# ============================================================

def parse_date_time_from_fname(fname):
    """
    Parse date (YYYY-MM-DD), time (HH:MM:SS.mmm), and image number
    from a CPI filename like:
      IMPACTS_HawkeyeCPI_20200205235114235131546_002205_C1.png
    where parts[2] is '20200205235114235131546'.

    Date: first 8 chars of parts[2] -> 20200205 -> 2020-02-05
    Time: next 9 chars -> HHMMSSmmm -> HH:MM:SS.mmm
    """
    try:
        parts = fname.split("_")
        time_raw = parts[2]  # e.g., '20200205235114235131546'

        # Extract date
        date_raw = time_raw[:8]  # 'YYYYMMDD'
        date_str = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

        # Extract time info (last 9 characters as before)
        hhmmss = time_raw[-9:-3]
        ms = time_raw[-3:]
        time_str = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}.{ms}"

        # Image number
        image_number = parts[3].split(".")[0]
    except Exception:
        date_str, time_str, image_number = "", "", ""
    return date_str, time_str, image_number

def get_year_and_flight_id(flight_dir):
    """
    Extract year (2020/2022/2023) and flight_id (e.g., 20230223_114748 or 20220114_1)
    from a full flight_dir path.
    """
    parts = flight_dir.split(os.sep)
    year_str = None
    for p in parts:
        if p.startswith("IMPACTS") and len(p) >= 11:
            year_str = p[-4:]
            break
    if year_str is None:
        raise ValueError(f"Could not infer year from path: {flight_dir}")

    # Flight id is the FlightData subdirectory name
    flight_root = os.path.dirname(flight_dir)  # .../FlightData/<flight_id>
    flight_id = os.path.basename(flight_root)

    return year_str, flight_id

def run_inference_for_flight(flight_dir):
    """Run ResNet34 inference for a single flight."""
    if not os.path.isdir(flight_dir):
        print(f"[WARN] Flight directory does not exist, skipping: {flight_dir}")
        return

    year_str, flight_id = get_year_and_flight_id(flight_dir)

    # Build output dirs: base/year/flight_id/
    output_dir = os.path.join(BASE_OUTPUT_DIR, year_str, flight_id)
    chain_dir = os.path.join(output_dir, "CNN_T2_chains")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chain_dir, exist_ok=True)

    # --- NEW: clear any existing files in CNN_T2_chains for a clean rerun ---
    for fn in os.listdir(chain_dir):
        fpath = os.path.join(chain_dir, fn)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"[WARN] Could not remove {fpath}: {e}")
    # ------------------------------------------------------------------------

    print(f"\n=== Running flight {flight_id} (year {year_str}) ===")
    print(f"Input:  {flight_dir}")
    print(f"Output: {output_dir}")

    ds = CPIFiles(flight_dir, transform=transform)
    if len(ds) == 0:
        print(f"[WARN] No PNG files found in {flight_dir}, skipping.")
        return

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

    records = []
    desc = f"{flight_id} ({len(ds)} imgs)"
    pbar = tqdm(total=len(ds), desc=f"Classifying Images {desc}", unit="img")

    with torch.inference_mode():
        for xb, paths, fnames in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

            for prob, pred, path, fname in zip(probs, preds, paths, fnames):
                if pred == 1:
                    shutil.copy(path, os.path.join(chain_dir, fname))
                date_str, time_str, image_number = parse_date_time_from_fname(fname)
                records.append({
                    "Date": date_str,
                    "Time": time_str,
                    "Image_Number": image_number,
                    "Predicted_Label": int(pred),
                    "Probability": float(round(prob, 4)),
                    "File_Name": fname,
                })

            pbar.update(len(fnames))

    pbar.close()

    # Save CSV per flight, with Date as the first column
    df = pd.DataFrame(records)
    cols = ["Date", "Time", "Image_Number", "Predicted_Label", "Probability", "File_Name"]
    df = df[cols]

    csv_path = os.path.join(output_dir, "CNN-ResNet34-ep18_predictions.csv")
    df.to_csv(csv_path, index=False)

    print(f"Finished flight {flight_id}.")
    print(f"  Chains saved to: {chain_dir}")
    print(f"  Prediction CSV:  {csv_path}")

# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    if RUN_ALL_FLIGHTS:
        flight_list = ALL_FLIGHT_DIRS
    else:
        flight_list = [SINGLE_FLIGHT_DIR]

    # Outer progress bar over flights
    for fd in tqdm(flight_list, desc="Processing flights", unit="flight"):
        run_inference_for_flight(fd)
