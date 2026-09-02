#!/bin/env python3
"""
  NAME:
    convert_CNN-preds_2nasa.py

  PURPOSE:
    This script converts the Conv. Neural Network (CNN) predictions into NASA
    format.

  CALLS:
    ADPAA
    Numpy
    Pandas

  EXAMPLE:
    The command:
        convert_CNN-preds_2nasa.py <CNN_PREDICTIONS_DATA_FILE.csv>

        convert_CNN-preds_2nasa.py /path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS

    creates:
        YY_MM_DD_HH_MI_SS.cnn.chain_non-chain.preds.1Hz

  MODIFICATIONS:
    Christian Nairy <christian.nairy@und.edu> - 2025/07/28:
      Written...
    Christian Nairy <christian.nairy@und.edu> - 2025/12/04:
      Adjusted code to perform this script on multiple flights concurrently.
    Christian Nairy <christian.nairy@und.edu> - 2026/06/09:
      Added USE_FILTERED_IMAGES toggle to optionally process
      CNN_Predictions_IMPACTS_Filtered directory.

  COPYRIGHT:
    2025 David Delene

    This program is distributed under terms of the GNU General Public License

    This file is part of Airborne Data Processing and Analysis (ADPAA).

    ADPAA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ADPAA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ADPAA.  If not, see <http://www.gnu.org/licenses/>.
"""
###START###

import os
import sys
from adpaa_python3 import ADPAA
import numpy as np
from datetime import datetime
import pandas as pd
import warnings
import traceback

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# ============================================================
# User settings
# ============================================================

# If True, the script will ignore the command-line input path and process:
#   /path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS_Filtered
#
# If False, the script will use the file or directory provided on the command line.
USE_FILTERED_IMAGES = True

CNN_PREDICTIONS_DIR = "/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS"
CNN_PREDICTIONS_FILTERED_DIR = "/path/to/nas_workspace/Paper3/CNN_Predictions_IMPACTS_Filtered"


# Thresholds for deciding a true midnight rollover vs garbage
# NOTE:
# These are currently not used because the script now uses full datetimes
# and simply drops rows where the datetime goes backwards.
ROLLOVER_PREV_MIN = 22 * 3600  # previous time >= 22:00:00
ROLLOVER_CURR_MAX = 2 * 3600   # current time <= 02:00:00


# ----------------------------------------------------------------------
# Helper: usage message
# ----------------------------------------------------------------------
def help_message():
    print(
        "\nSyntax:\n"
        "  convert_CNN-preds_2nasa.py <CNN_PREDICTIONS_DATA_FILE.csv>\n"
        "  convert_CNN-preds_2nasa.py <DIRECTORY_CONTAINING_PREDICTIONS>\n\n"
        "Toggle option:\n"
        "  If USE_FILTERED_IMAGES = True, the script will process:\n"
        f"    {CNN_PREDICTIONS_FILTERED_DIR}\n"
        "  and the command-line input path will be ignored.\n"
    )


# ----------------------------------------------------------------------
# Core conversion for a single CNN prediction CSV
# ----------------------------------------------------------------------
def convert_cnn_file(cnn_file):
    """
    Convert a single CNN prediction CSV file to ADPAA/NASA format.

    Assumes columns:
      Date, Time, Image_Number, Predicted_Label, Probability, File_Name

    Parameters
    ----------
    cnn_file : str
        Full path to the CNN prediction CSV file.

    Returns
    -------
    (bool, str or None, int, int)
        (ok_flag, error_text, skipped_invalid_time, skipped_garbage_time).
        ok_flag is True if processing succeeded, False if an error occurred.
        error_text contains the traceback if an error occurred, otherwise None.
    """
    cnn_file = os.path.abspath(cnn_file)
    dirpath = os.path.dirname(cnn_file)
    fname = os.path.basename(cnn_file)

    print(f"\nProcessing: {cnn_file}")

    # Counters for skipped indices
    skipped_invalid_time = 0
    skipped_garbage_time = 0

    # Ensure ADPAA output is written in the same directory as the CSV
    orig_cwd = os.getcwd()

    try:
        if dirpath:
            os.chdir(dirpath)

        # ------------------------------------------------------------------
        # Load CNN prediction data using pandas
        # ------------------------------------------------------------------
        df = pd.read_csv(fname)

        required_cols = [
            "Date",
            "Time",
            "Predicted_Label",
            "Probability",
            "File_Name",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in {fname}")

        # ------------------------------------------------------------------
        # Combine Date + Time into full UTC datetime
        # ------------------------------------------------------------------
        dt_str = (
            df["Date"].astype(str).str.strip()
            + " "
            + df["Time"].astype(str).str.strip()
        )

        # Let pandas infer format; handle both with and without fractional seconds
        df["datetime"] = pd.to_datetime(dt_str, errors="coerce", utc=True)

        # Skip rows with invalid datetime
        invalid_mask = df["datetime"].isna()
        skipped_invalid_time = int(invalid_mask.sum())

        if skipped_invalid_time > 0:
            print(f"  Skipping {skipped_invalid_time} rows with invalid datetime")

        df = df[~invalid_mask].reset_index(drop=True)

        if df.empty:
            raise ValueError("No valid datetimes found after filtering invalid rows.")

        # ------------------------------------------------------------------
        # Drop garbage rows where time goes backwards
        #
        # Example:
        #   12:00:00 -> 00:40:00 -> 12:00:01
        #
        # The 00:40:00 row is treated as garbage because the datetime goes
        # backwards relative to the previous valid datetime.
        # ------------------------------------------------------------------
        dt_values = df["datetime"].to_numpy()
        keep_mask = np.ones(len(dt_values), dtype=bool)

        prev_dt = dt_values[0]

        for i in range(1, len(dt_values)):
            curr_dt = dt_values[i]

            if curr_dt < prev_dt:
                skipped_garbage_time += 1
                keep_mask[i] = False

                print(
                    f"  Skipping garbage datetime at index {i}: "
                    f"{curr_dt} (prev={prev_dt})"
                )
            else:
                prev_dt = curr_dt

        df = df[keep_mask].reset_index(drop=True)

        if df.empty:
            raise ValueError("No data left after dropping garbage datetime rows.")

        # ------------------------------------------------------------------
        # Compute seconds from midnight on day measurements started
        #
        # This naturally handles crossing midnight because datetime is
        # strictly increasing; no manual rollover logic needed.
        # ------------------------------------------------------------------
        dt0 = df["datetime"].iloc[0]
        midnight0 = dt0.normalize()  # 00:00 of the first date, UTC

        sfm = (df["datetime"] - midnight0).dt.total_seconds().to_numpy()

        # Extract prediction/probability arrays
        pred = df["Predicted_Label"].astype(float).to_numpy()
        prob = df["Probability"].astype(float).to_numpy()

        # ------------------------------------------------------------------
        # Extract DATE for ADPAA header and TIME for output filename
        # ------------------------------------------------------------------
        yyyy = f"{dt0.year:04d}"
        yy = f"{dt0.year % 100:02d}"
        mm = f"{dt0.month:02d}"
        dd = f"{dt0.day:02d}"
        hh = f"{dt0.hour:02d}"
        mi = f"{dt0.minute:02d}"
        ss = f"{dt0.second:02d}"

        # ------------------------------------------------------------------
        # Aggregate to 1 Hz by rounding down to nearest second
        # ------------------------------------------------------------------
        sfm_rounded = np.floor(sfm).astype(float)

        df_agg = pd.DataFrame(
            {
                "Time": sfm_rounded,
                "Prediction": pred,
                "Probability": prob,
            }
        )

        df_1hz = (
            df_agg.groupby("Time")
            .agg(
                {
                    "Prediction": "sum",
                    "Probability": lambda x: np.nan
                    if (x[x > 0.01].empty)
                    else x[x > 0.01].mean(),
                }
            )
            .reset_index()
        )

        sfm_1hz = df_1hz["Time"].to_numpy()
        pred_1hz = df_1hz["Prediction"].to_numpy()
        prob_1hz = df_1hz["Probability"].to_numpy()

        pred_1hz[np.isnan(pred_1hz)] = 999999.9999
        prob_1hz[np.isnan(prob_1hz)] = 999999.9999

        # ------------------------------------------------------------------
        # Metadata for ADPAA
        # ------------------------------------------------------------------
        date_str = datetime.today().strftime("%Y-%m-%d")

        out = ADPAA()
        out.DREV = 0
        out.NLHEAD = 22
        out.FFI = 1001
        out.ONAME = "Delene, David"
        out.ORG = "University of North Dakota"
        out.SNAME = "Atmospheric Science Dept."
        out.MNAME = "IMPACTS Hawkeye-CPI Particle Chain/non-chain CNN Prediction"
        out.IVOL = 1
        out.VVOL = 1
        out.DATE = f"{yyyy} {mm} {dd}"
        out.RDATE = date_str
        out.DX = 1.0
        out.XNAME = (
            "Time [second]; UT seconds from midnight on day "
            "measurements started."
        )
        out.NV = 2
        out.VSCAL = ["     1.0000", "     1.0000"]
        out.VMISS = ["999999.9999", "999999.9999"]
        out.VNAME = [
            "pred: CNN Chain Aggregate Prediction [0=non-chain; 1=chain]",
            "prob: CNN Chain Aggregate Prediction Probability [Thresh = 0.666]",
        ]
        out.DTYPE = "Floats. NaN converted to 999999.9999"
        out.VFREQ = "1Hz (aggregated from async)"
        out.VDESC = ["Time", "Prediction", "Probability"]
        out.VUNITS = ["s", "count", "probability"]

        out.data = {
            "Time": sfm_1hz,
            "Prediction": pred_1hz,
            "Probability": prob_1hz,
        }

        # Do not change output file naming.
        out.name = f"{yy}_{mm}_{dd}_{hh}_{mi}_{ss}.cnn.chain_non-chain.preds.1Hz"

        out.WriteFile()

        print(f"  -> {fname} converted to {out.name} in {dirpath}")
        print(
            f"  Skipped indices for this file: "
            f"invalid_time={skipped_invalid_time}, "
            f"garbage_time={skipped_garbage_time}"
        )

        return True, None, skipped_invalid_time, skipped_garbage_time

    except Exception:
        err_text = traceback.format_exc()
        print(f"  ERROR processing {cnn_file}")
        return False, err_text, skipped_invalid_time, skipped_garbage_time

    finally:
        os.chdir(orig_cwd)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Determine input target
    # ------------------------------------------------------------------
    if USE_FILTERED_IMAGES:
        target = CNN_PREDICTIONS_FILTERED_DIR

        print(
            "\nUSE_FILTERED_IMAGES = True"
            "\nProcessing filtered CNN predictions directory:"
            f"\n  {target}"
        )

    else:
        if any(param in sys.argv for param in ["-h", "--help"]) or len(sys.argv) < 2:
            help_message()

            if len(sys.argv) < 2:
                sys.exit("Error: Missing input file or directory.")

            sys.exit(0)

        target = sys.argv[1]

    errors = []
    summary_records = []

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------
    if os.path.isfile(target):

        ok, err_text, skipped_invalid, skipped_garbage = convert_cnn_file(target)
        abs_path = os.path.abspath(target)

        summary_records.append(
            {
                "file": abs_path,
                "skipped_invalid": skipped_invalid,
                "skipped_garbage": skipped_garbage,
            }
        )

        if not ok:
            errors.append((abs_path, err_text))

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------
    elif os.path.isdir(target):

        base_dir = os.path.abspath(target)
        pattern = "CNN-ResNet34-ep18_predictions.csv"

        print(f"\nSearching for '{pattern}' under: {base_dir}")

        cnn_files = []

        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f == pattern:
                    cnn_files.append(os.path.join(root, f))

        if not cnn_files:
            print(f"No '{pattern}' files found under: {base_dir}")
            sys.exit(1)

        print(f"Found {len(cnn_files)} prediction file(s). Beginning conversion...")

        for f in sorted(cnn_files):
            ok, err_text, skipped_invalid, skipped_garbage = convert_cnn_file(f)
            abs_path = os.path.abspath(f)

            summary_records.append(
                {
                    "file": abs_path,
                    "skipped_invalid": skipped_invalid,
                    "skipped_garbage": skipped_garbage,
                }
            )

            if not ok:
                errors.append((abs_path, err_text))

        print("\nAll conversions attempted.")

    else:
        help_message()
        sys.exit("Error: Input must be a CSV file or a directory.")

    # ------------------------------------------------------------------
    # Write skipped-index summary file
    # ------------------------------------------------------------------
    if summary_records:

        if os.path.isdir(target):
            base_for_summary = os.path.abspath(target)
        else:
            base_for_summary = os.path.dirname(os.path.abspath(target))

        summary_path = os.path.join(
            base_for_summary,
            "CNN_pred_skipped_indices_summary.txt",
        )

        with open(summary_path, "w") as f:
            f.write("Summary of skipped indices per flight\n")
            f.write(
                f"Generated: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(
                "Flight_Dir,CSV_Path,Skipped_Invalid_Time,"
                "Skipped_Garbage_Time,Total_Skipped\n"
            )

            for rec in summary_records:
                file_path = rec["file"]
                flight_dir = os.path.basename(os.path.dirname(file_path))

                try:
                    rel_path = os.path.relpath(file_path, base_for_summary)
                except ValueError:
                    rel_path = file_path

                total_skipped = rec["skipped_invalid"] + rec["skipped_garbage"]

                f.write(
                    f"{flight_dir},{rel_path},"
                    f"{rec['skipped_invalid']},{rec['skipped_garbage']},"
                    f"{total_skipped}\n"
                )

        print(f"\nSkipped-index summary written to: {summary_path}")

    # ------------------------------------------------------------------
    # Final error summary
    # ------------------------------------------------------------------
    if errors:

        print("\nThe following files had errors during processing:")

        for epath, etext in errors:
            print(f"\n  {epath}\n")

            for line in etext.rstrip("\n").splitlines():
                print("    " + line)

    else:
        print("\nAll files processed successfully with no errors.")