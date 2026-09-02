#!/usr/bin/env python3

import cv2
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
import time
from datetime import datetime
from tqdm import tqdm


# ============================================================
# User settings
# ============================================================

flightdata_root = Path(
    "/path/to/impacts_workspace/IMPACTS2020/Aircraft/P3_N426NA/FlightData"
)

flight_directories = [
    "20200112_131625",
    "20200118_130356",
    "20200125_133603",
    "20200201_060849",
    "20200205_181400",
    "20200207_134955",
    "20200213_054812",
    "20200218_170335",
    "20200220_192119",
    "20200224_173407",
    "20200225_1",
]

input_directory_name = "Hawkeye-CPI_C1_Images"
bad_output_directory_name = "Hawkeye-CPI_C1_Bad_Images"
good_output_directory_name = "Hawkeye-CPI_C1_Good_Images"

# Optional combined summary across all processed flights
combined_summary_csv = flightdata_root / "cpi_good_bad_image_filtering_summary_ALL_FLIGHTS.csv"

# Lower values mean blurrier images.
# Images with Laplacian variance below this value are classified as blurry.
blur_threshold = 30.0

# Edge-strength blur tests.
edge_strength_p99_threshold = 30.0
edge_strength_p95_threshold = 19.0

# Empty-image detection settings.
empty_contrast_threshold = 18.0
empty_min_particle_area_px = 40
empty_min_dark_fraction = 0.0005

# Image extensions to search for
image_extensions = [".png", ".PNG"]

# Number of CPU workers.
# I recommend 8 for many small PNG files because using all cores can overwhelm disk I/O.
# You can increase this later if it runs smoothly.
n_workers = 8

# Number of tasks given to each worker at a time.
# Larger chunks reduce multiprocessing overhead.
chunksize = 100


# ============================================================
# Utility helpers
# ============================================================

def print_time_message(message):
    """
    Print a message with a timestamp and flush immediately.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================
# Metric computation
# ============================================================

def analyze_image(
    image_path_str,
    input_root_str,
    flight_name,
    blur_threshold,
    edge_strength_p99_threshold,
    edge_strength_p95_threshold,
    empty_contrast_threshold,
    empty_min_particle_area_px,
    empty_min_dark_fraction
):
    """
    Read one image once, then compute:
    1. Laplacian variance sharpness score
    2. Sobel edge-strength metrics
    3. Empty-image metrics

    Lower sharpness_score = blurrier image.
    Lower edge_strength_p95/p99 = smoother / blurrier / less detailed image.
    """

    image_path = Path(image_path_str)
    input_root = Path(input_root_str)

    relative_path = image_path.relative_to(input_root)

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {
            "flight": flight_name,
            "input_path": str(image_path),
            "relative_path": str(relative_path),
            "sharpness_score": None,
            "edge_strength_mean": None,
            "edge_strength_p95": None,
            "edge_strength_p99": None,
            "is_blurry": None,
            "is_low_edge_strength_p95": None,
            "is_low_edge_strength_p99": None,
            "is_low_edge_strength": None,
            "background_level": None,
            "largest_dark_area_px": None,
            "dark_fraction": None,
            "is_empty": None,
            "is_bad": None,
            "is_good": None,
            "bad_reason": "failed_to_read",
            "status": "failed_to_read"
        }

    # --------------------------------------------------------
    # Blur metric: variance of Laplacian
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpness_score = float(laplacian.var())

    is_blurry = sharpness_score < blur_threshold

    # --------------------------------------------------------
    # Edge-strength metric: Sobel gradient magnitude
    # --------------------------------------------------------

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    edge_strength_mean = float(np.mean(gradient_magnitude))
    edge_strength_p95 = float(np.percentile(gradient_magnitude, 95))
    edge_strength_p99 = float(np.percentile(gradient_magnitude, 99))

    is_low_edge_strength_p95 = edge_strength_p95 < edge_strength_p95_threshold
    is_low_edge_strength_p99 = edge_strength_p99 < edge_strength_p99_threshold

    is_low_edge_strength = (
        is_low_edge_strength_p95 or
        is_low_edge_strength_p99
    )

    # --------------------------------------------------------
    # Empty-image metric
    # --------------------------------------------------------

    img_smooth = cv2.GaussianBlur(img, (3, 3), 0)

    background = float(np.percentile(img_smooth, 90))

    dark_mask = (background - img_smooth) > empty_contrast_threshold
    dark_mask = dark_mask.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark_mask,
        connectivity=8
    )

    if num_labels <= 1:
        largest_area_px = 0
    else:
        largest_area_px = int(stats[1:, cv2.CC_STAT_AREA].max())

    dark_fraction = float(np.count_nonzero(dark_mask) / dark_mask.size)

    is_empty = (
        largest_area_px < empty_min_particle_area_px and
        dark_fraction < empty_min_dark_fraction
    )

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    is_bad = (
        is_blurry or
        is_low_edge_strength or
        is_empty
    )

    is_good = not is_bad

    bad_reasons = []

    if is_blurry:
        bad_reasons.append("blurry_laplacian")

    if is_low_edge_strength_p95:
        bad_reasons.append("low_edge_strength_p95")

    if is_low_edge_strength_p99:
        bad_reasons.append("low_edge_strength_p99")

    if is_empty:
        bad_reasons.append("empty")

    if len(bad_reasons) == 0:
        bad_reason = "good"
    else:
        bad_reason = ";".join(bad_reasons)

    return {
        "flight": flight_name,
        "input_path": str(image_path),
        "relative_path": str(relative_path),
        "sharpness_score": sharpness_score,
        "edge_strength_mean": edge_strength_mean,
        "edge_strength_p95": edge_strength_p95,
        "edge_strength_p99": edge_strength_p99,
        "is_blurry": bool(is_blurry),
        "is_low_edge_strength_p95": bool(is_low_edge_strength_p95),
        "is_low_edge_strength_p99": bool(is_low_edge_strength_p99),
        "is_low_edge_strength": bool(is_low_edge_strength),
        "background_level": background,
        "largest_dark_area_px": largest_area_px,
        "dark_fraction": dark_fraction,
        "is_empty": bool(is_empty),
        "is_bad": bool(is_bad),
        "is_good": bool(is_good),
        "bad_reason": bad_reason,
        "status": "ok"
    }


def analyze_image_from_tuple(args):
    """
    Wrapper needed for executor.map with multiple arguments.
    """

    return analyze_image(*args)


# ============================================================
# Copy helper
# ============================================================

def copy_preserving_structure(image_path, source_root, destination_root):
    """
    Copy image to destination while preserving the directory structure
    relative to source_root.

    This only copies images. It does not delete, move, or modify originals.
    """

    relative_path = image_path.relative_to(source_root)
    destination_path = destination_root / relative_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, destination_path)

    return destination_path


# ============================================================
# Plotting helper
# ============================================================

def make_good_bad_histograms(
    summary_df,
    flight_dir,
    blur_threshold,
    edge_strength_p99_threshold,
    edge_strength_p95_threshold
):
    """
    Save diagnostic PNG plots comparing good and bad images for one flight.
    """

    valid_df = summary_df.dropna(
        subset=["sharpness_score", "edge_strength_p95", "edge_strength_p99"]
    ).copy()

    if valid_df.empty:
        print_time_message("No valid quality metrics available for histogram.")
        return

    good_df = valid_df.loc[valid_df["is_good"] == True].copy()
    bad_df = valid_df.loc[valid_df["is_bad"] == True].copy()

    good_scores = good_df["sharpness_score"]
    bad_scores = bad_df["sharpness_score"]

    good_edges_p95 = good_df["edge_strength_p95"]
    bad_edges_p95 = bad_df["edge_strength_p95"]

    good_edges_p99 = good_df["edge_strength_p99"]
    bad_edges_p99 = bad_df["edge_strength_p99"]

    saved_paths = []

    # --------------------------------------------------------
    # 1. Sharpness count histogram, linear x-axis
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        good_scores,
        bins=100,
        alpha=0.6,
        color="blue",
        label=f"Good images, n={len(good_scores)}"
    )

    ax.hist(
        bad_scores,
        bins=100,
        alpha=0.6,
        color="red",
        label=f"Bad images, n={len(bad_scores)}"
    )

    ax.axvline(
        blur_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Blur threshold = {blur_threshold}"
    )

    ax.set_xlabel("Laplacian variance sharpness score")
    ax.set_ylabel("Number of CPI images")
    ax.set_title("CPI image sharpness scores: good vs bad images")
    ax.legend()

    fig.tight_layout()

    sharpness_hist_path = flight_dir / "cpi_good_bad_sharpness_histogram.png"
    fig.savefig(sharpness_hist_path, dpi=300)
    plt.close(fig)
    saved_paths.append(sharpness_hist_path)

    # --------------------------------------------------------
    # 2. Sharpness count histogram, log x-axis
    # --------------------------------------------------------

    good_scores_positive = good_scores[good_scores > 0]
    bad_scores_positive = bad_scores[bad_scores > 0]

    all_positive_scores = valid_df.loc[
        valid_df["sharpness_score"] > 0,
        "sharpness_score"
    ]

    log_bins = None

    if len(all_positive_scores) > 0:

        log_bins = np.logspace(
            np.log10(all_positive_scores.min()),
            np.log10(all_positive_scores.max()),
            100
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.hist(
            good_scores_positive,
            bins=log_bins,
            alpha=0.6,
            color="blue",
            label=f"Good images, n={len(good_scores_positive)}"
        )

        ax.hist(
            bad_scores_positive,
            bins=log_bins,
            alpha=0.6,
            color="red",
            label=f"Bad images, n={len(bad_scores_positive)}"
        )

        ax.axvline(
            blur_threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Blur threshold = {blur_threshold}"
        )

        ax.set_xscale("log")
        ax.set_xlabel("Laplacian variance sharpness score, log scale")
        ax.set_ylabel("Number of CPI images")
        ax.set_title("CPI image sharpness scores: good vs bad images, log scale")
        ax.legend()

        fig.tight_layout()

        sharpness_log_hist_path = flight_dir / "cpi_good_bad_sharpness_histogram_logx.png"
        fig.savefig(sharpness_log_hist_path, dpi=300)
        plt.close(fig)
        saved_paths.append(sharpness_log_hist_path)

    # --------------------------------------------------------
    # 3. Sharpness normalized histogram, linear x-axis
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        good_scores,
        bins=100,
        density=True,
        alpha=0.6,
        color="blue",
        label=f"Good images, n={len(good_scores)}"
    )

    ax.hist(
        bad_scores,
        bins=100,
        density=True,
        alpha=0.6,
        color="red",
        label=f"Bad images, n={len(bad_scores)}"
    )

    ax.axvline(
        blur_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Blur threshold = {blur_threshold}"
    )

    ax.set_xlabel("Laplacian variance sharpness score")
    ax.set_ylabel("Normalized density")
    ax.set_title("Normalized CPI image sharpness scores: good vs bad images")
    ax.legend()

    fig.tight_layout()

    sharpness_norm_hist_path = flight_dir / "cpi_good_bad_sharpness_histogram_normalized.png"
    fig.savefig(sharpness_norm_hist_path, dpi=300)
    plt.close(fig)
    saved_paths.append(sharpness_norm_hist_path)

    # --------------------------------------------------------
    # 4. Sharpness normalized histogram, log x-axis
    # --------------------------------------------------------

    if len(all_positive_scores) > 0 and log_bins is not None:

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.hist(
            good_scores_positive,
            bins=log_bins,
            density=True,
            alpha=0.6,
            color="blue",
            label=f"Good images, n={len(good_scores_positive)}"
        )

        ax.hist(
            bad_scores_positive,
            bins=log_bins,
            density=True,
            alpha=0.6,
            color="red",
            label=f"Bad images, n={len(bad_scores_positive)}"
        )

        ax.axvline(
            blur_threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Blur threshold = {blur_threshold}"
        )

        ax.set_xscale("log")
        ax.set_xlabel("Laplacian variance sharpness score, log scale")
        ax.set_ylabel("Normalized density")
        ax.set_title("Normalized CPI image sharpness scores: good vs bad images, log scale")
        ax.legend()

        fig.tight_layout()

        sharpness_norm_log_hist_path = flight_dir / "cpi_good_bad_sharpness_histogram_normalized_logx.png"
        fig.savefig(sharpness_norm_log_hist_path, dpi=300)
        plt.close(fig)
        saved_paths.append(sharpness_norm_log_hist_path)

    # --------------------------------------------------------
    # 5. Edge-strength p99 count histogram
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        good_edges_p99,
        bins=100,
        alpha=0.6,
        color="blue",
        label=f"Good images, n={len(good_edges_p99)}"
    )

    ax.hist(
        bad_edges_p99,
        bins=100,
        alpha=0.6,
        color="red",
        label=f"Bad images, n={len(bad_edges_p99)}"
    )

    ax.axvline(
        edge_strength_p99_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"p99 edge threshold = {edge_strength_p99_threshold}"
    )

    ax.set_xlabel("Sobel edge strength p99")
    ax.set_ylabel("Number of CPI images")
    ax.set_title("CPI image edge strength p99: good vs bad images")
    ax.legend()

    fig.tight_layout()

    edge_p99_hist_path = flight_dir / "cpi_good_bad_edge_strength_p99_histogram.png"
    fig.savefig(edge_p99_hist_path, dpi=300)
    plt.close(fig)
    saved_paths.append(edge_p99_hist_path)

    # --------------------------------------------------------
    # 6. Edge-strength p99 normalized histogram
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        good_edges_p99,
        bins=100,
        density=True,
        alpha=0.6,
        color="blue",
        label=f"Good images, n={len(good_edges_p99)}"
    )

    ax.hist(
        bad_edges_p99,
        bins=100,
        density=True,
        alpha=0.6,
        color="red",
        label=f"Bad images, n={len(bad_edges_p99)}"
    )

    ax.axvline(
        edge_strength_p99_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"p99 edge threshold = {edge_strength_p99_threshold}"
    )

    ax.set_xlabel("Sobel edge strength p99")
    ax.set_ylabel("Normalized density")
    ax.set_title("Normalized CPI image edge strength p99: good vs bad images")
    ax.legend()

    fig.tight_layout()

    edge_p99_norm_hist_path = flight_dir / "cpi_good_bad_edge_strength_p99_histogram_normalized.png"
    fig.savefig(edge_p99_norm_hist_path, dpi=300)
    plt.close(fig)
    saved_paths.append(edge_p99_norm_hist_path)

    # --------------------------------------------------------
    # 7. Edge-strength p95 count histogram
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        good_edges_p95,
        bins=100,
        alpha=0.6,
        color="blue",
        label=f"Good images, n={len(good_edges_p95)}"
    )

    ax.hist(
        bad_edges_p95,
        bins=100,
        alpha=0.6,
        color="red",
        label=f"Bad images, n={len(bad_edges_p95)}"
    )

    ax.axvline(
        edge_strength_p95_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"p95 edge threshold = {edge_strength_p95_threshold}"
    )

    ax.set_xlabel("Sobel edge strength p95")
    ax.set_ylabel("Number of CPI images")
    ax.set_title("CPI image edge strength p95: good vs bad images")
    ax.legend()

    fig.tight_layout()

    edge_p95_hist_path = flight_dir / "cpi_good_bad_edge_strength_p95_histogram.png"
    fig.savefig(edge_p95_hist_path, dpi=300)
    plt.close(fig)
    saved_paths.append(edge_p95_hist_path)

    # --------------------------------------------------------
    # 8. Edge-strength p95 normalized histogram
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        good_edges_p95,
        bins=100,
        density=True,
        alpha=0.6,
        color="blue",
        label=f"Good images, n={len(good_edges_p95)}"
    )

    ax.hist(
        bad_edges_p95,
        bins=100,
        density=True,
        alpha=0.6,
        color="red",
        label=f"Bad images, n={len(bad_edges_p95)}"
    )

    ax.axvline(
        edge_strength_p95_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"p95 edge threshold = {edge_strength_p95_threshold}"
    )

    ax.set_xlabel("Sobel edge strength p95")
    ax.set_ylabel("Normalized density")
    ax.set_title("Normalized CPI image edge strength p95: good vs bad images")
    ax.legend()

    fig.tight_layout()

    edge_p95_norm_hist_path = flight_dir / "cpi_good_bad_edge_strength_p95_histogram_normalized.png"
    fig.savefig(edge_p95_norm_hist_path, dpi=300)
    plt.close(fig)
    saved_paths.append(edge_p95_norm_hist_path)

    # --------------------------------------------------------
    # 9. Sharpness vs edge-strength p99 scatter plot
    # --------------------------------------------------------

    scatter_df = valid_df.copy()

    max_scatter_points = 30000

    if len(scatter_df) > max_scatter_points:
        scatter_df = scatter_df.sample(
            n=max_scatter_points,
            random_state=42
        )

    scatter_good = scatter_df.loc[scatter_df["is_good"] == True]
    scatter_bad = scatter_df.loc[scatter_df["is_bad"] == True]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        scatter_good["sharpness_score"],
        scatter_good["edge_strength_p99"],
        s=6,
        alpha=0.25,
        color="blue",
        label="Good images"
    )

    ax.scatter(
        scatter_bad["sharpness_score"],
        scatter_bad["edge_strength_p99"],
        s=6,
        alpha=0.25,
        color="red",
        label="Bad images"
    )

    ax.axvline(
        blur_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Blur threshold = {blur_threshold}"
    )

    ax.axhline(
        edge_strength_p99_threshold,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"p99 edge threshold = {edge_strength_p99_threshold}"
    )

    ax.set_xlabel("Laplacian variance sharpness score")
    ax.set_ylabel("Sobel edge strength p99")
    ax.set_title("Sharpness score vs edge strength p99")
    ax.legend()

    fig.tight_layout()

    scatter_p99_path = flight_dir / "cpi_sharpness_vs_edge_strength_p99_scatter.png"
    fig.savefig(scatter_p99_path, dpi=300)
    plt.close(fig)
    saved_paths.append(scatter_p99_path)

    # --------------------------------------------------------
    # 10. Sharpness vs edge-strength p95 scatter plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        scatter_good["sharpness_score"],
        scatter_good["edge_strength_p95"],
        s=6,
        alpha=0.25,
        color="blue",
        label="Good images"
    )

    ax.scatter(
        scatter_bad["sharpness_score"],
        scatter_bad["edge_strength_p95"],
        s=6,
        alpha=0.25,
        color="red",
        label="Bad images"
    )

    ax.axvline(
        blur_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Blur threshold = {blur_threshold}"
    )

    ax.axhline(
        edge_strength_p95_threshold,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"p95 edge threshold = {edge_strength_p95_threshold}"
    )

    ax.set_xlabel("Laplacian variance sharpness score")
    ax.set_ylabel("Sobel edge strength p95")
    ax.set_title("Sharpness score vs edge strength p95")
    ax.legend()

    fig.tight_layout()

    scatter_p95_path = flight_dir / "cpi_sharpness_vs_edge_strength_p95_scatter.png"
    fig.savefig(scatter_p95_path, dpi=300)
    plt.close(fig)
    saved_paths.append(scatter_p95_path)

    # --------------------------------------------------------
    # 11. Edge strength p95 vs p99, zoomed decision region
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        scatter_good["edge_strength_p95"],
        scatter_good["edge_strength_p99"],
        s=6,
        alpha=0.25,
        color="blue",
        label="Good images"
    )

    ax.scatter(
        scatter_bad["edge_strength_p95"],
        scatter_bad["edge_strength_p99"],
        s=6,
        alpha=0.25,
        color="red",
        label="Bad images"
    )

    ax.axvline(
        edge_strength_p95_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"p95 threshold = {edge_strength_p95_threshold}"
    )

    ax.axhline(
        edge_strength_p99_threshold,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"p99 threshold = {edge_strength_p99_threshold}"
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 150)

    ax.set_xlabel("Sobel edge strength p95")
    ax.set_ylabel("Sobel edge strength p99")
    ax.set_title("Edge strength p95 vs p99, zoomed decision region")
    ax.legend()

    fig.tight_layout()

    edge_p95_p99_scatter_path = flight_dir / "cpi_edge_strength_p95_vs_p99_scatter_zoomed.png"
    fig.savefig(edge_p95_p99_scatter_path, dpi=300)
    plt.close(fig)
    saved_paths.append(edge_p95_p99_scatter_path)

    # --------------------------------------------------------
    # 12. Edge-threshold sensitivity plot for p95 and p99
    # --------------------------------------------------------

    thresholds = np.arange(5, 81, 1)

    p95_percent_flagged = []
    p99_percent_flagged = []

    for threshold in thresholds:
        p95_n_flagged = int((valid_df["edge_strength_p95"] < threshold).sum())
        p99_n_flagged = int((valid_df["edge_strength_p99"] < threshold).sum())

        p95_percent_flagged.append(100.0 * p95_n_flagged / len(valid_df))
        p99_percent_flagged.append(100.0 * p99_n_flagged / len(valid_df))

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        thresholds,
        p95_percent_flagged,
        linewidth=2,
        label="p95"
    )

    ax.plot(
        thresholds,
        p99_percent_flagged,
        linewidth=2,
        label="p99"
    )

    ax.axvline(
        edge_strength_p95_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"p95 threshold = {edge_strength_p95_threshold}"
    )

    ax.axvline(
        edge_strength_p99_threshold,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"p99 threshold = {edge_strength_p99_threshold}"
    )

    ax.set_xlabel("Candidate edge strength threshold")
    ax.set_ylabel("Images flagged by edge strength only (%)")
    ax.set_title("Sensitivity of bad-image flagging to edge-strength thresholds")
    ax.legend()

    fig.tight_layout()

    sensitivity_path = flight_dir / "cpi_edge_threshold_sensitivity_p95_p99.png"
    fig.savefig(sensitivity_path, dpi=300)
    plt.close(fig)
    saved_paths.append(sensitivity_path)

    # --------------------------------------------------------
    # 13. Bad-reason count plot
    # --------------------------------------------------------

    if "bad_reason" in summary_df.columns:

        bad_reason_counts = (
            summary_df["bad_reason"]
            .value_counts(dropna=False)
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.barh(
            bad_reason_counts.index.astype(str),
            bad_reason_counts.values
        )

        ax.set_xlabel("Number of CPI images")
        ax.set_ylabel("Bad reason")
        ax.set_title("CPI image classification reason counts")

        fig.tight_layout()

        bad_reason_plot_path = flight_dir / "cpi_bad_reason_counts.png"
        fig.savefig(bad_reason_plot_path, dpi=300)
        plt.close(fig)
        saved_paths.append(bad_reason_plot_path)

    print()
    print_time_message("Saved diagnostic PNG files:")

    for path in saved_paths:
        print(f"  {path}", flush=True)


# ============================================================
# Process one flight
# ============================================================

def process_one_flight(flight_name):

    flight_start_time = time.time()

    flight_dir = flightdata_root / flight_name

    input_root = flight_dir / input_directory_name
    bad_output_root = flight_dir / bad_output_directory_name
    good_output_root = flight_dir / good_output_directory_name

    summary_csv = flight_dir / "cpi_good_bad_image_filtering_summary.csv"
    partial_summary_csv = flight_dir / "cpi_good_bad_image_filtering_summary_PARTIAL.csv"

    print()
    print("============================================================")
    print_time_message(f"Checking flight directory: {flight_name}")
    print("============================================================")

    if not flight_dir.exists():
        print_time_message(f"Flight directory does not exist. Skipping: {flight_dir}")
        return None

    if not input_root.exists():
        print_time_message(f"No {input_directory_name} directory found. Skipping this flight.")
        print_time_message(f"Missing input directory: {input_root}")
        return None

    if not input_root.is_dir():
        print_time_message(f"Input path exists but is not a directory. Skipping: {input_root}")
        return None

    results = []

    n_good = 0
    n_bad = 0
    n_blurry = 0
    n_low_edge_strength = 0
    n_low_edge_strength_p95 = 0
    n_low_edge_strength_p99 = 0
    n_empty = 0
    n_failed = 0

    bad_output_root.mkdir(parents=True, exist_ok=True)
    good_output_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Collect all PNG image paths first
    # --------------------------------------------------------

    print_time_message("Collecting PNG image paths...")

    image_paths = []

    for image_path in tqdm(
        sorted(input_root.rglob("*")),
        desc=f"Scanning {flight_name}",
        unit="path"
    ):
        if image_path.is_file() and image_path.suffix in image_extensions:
            image_paths.append(image_path)

    n_total = len(image_paths)

    print()
    print_time_message(f"Found {n_total} PNG images.")
    print_time_message(f"Input directory: {input_root}")
    print_time_message(f"Bad images output directory: {bad_output_root}")
    print_time_message(f"Good images output directory: {good_output_root}")
    print()

    if n_total == 0:
        print_time_message("No PNG images found for this flight. Skipping.")
        return None

    # --------------------------------------------------------
    # Parallel image analysis
    # --------------------------------------------------------

    if n_workers is None:
        workers_to_use = os.cpu_count()
    else:
        workers_to_use = n_workers

    print_time_message(f"Using {workers_to_use} worker processes.")
    print_time_message(f"Using chunksize = {chunksize}.")
    print_time_message("Building analysis argument list...")
    print()

    analysis_args = [
        (
            str(image_path),
            str(input_root),
            flight_name,
            blur_threshold,
            edge_strength_p99_threshold,
            edge_strength_p95_threshold,
            empty_contrast_threshold,
            empty_min_particle_area_px,
            empty_min_dark_fraction
        )
        for image_path in image_paths
    ]

    print_time_message("Starting parallel image analysis...")
    print()

    try:
        with ProcessPoolExecutor(max_workers=workers_to_use) as executor:

            for result in tqdm(
                executor.map(
                    analyze_image_from_tuple,
                    analysis_args,
                    chunksize=chunksize
                ),
                total=len(analysis_args),
                desc=f"Analyzing {flight_name}",
                unit="image"
            ):
                results.append(result)

    except KeyboardInterrupt:
        print()
        print_time_message("KeyboardInterrupt detected during image analysis.")
        print_time_message("Saving partial results for this flight...")

        if len(results) > 0:
            partial_df = pd.DataFrame(results)

            partial_df = partial_df.sort_values(
                by=["flight", "relative_path"],
                ascending=True
            ).reset_index(drop=True)

            partial_df.to_csv(partial_summary_csv, index=False)

            print_time_message(f"Partial CSV saved to: {partial_summary_csv}")
            print_time_message(f"Partial results saved for {len(results)} images.")
        else:
            print_time_message("No completed image results were available to save.")

        raise

    print()
    print_time_message("Finished image analysis.")
    print_time_message(f"Completed image results: {len(results)}")
    print()

    # --------------------------------------------------------
    # Copy good and bad images
    # --------------------------------------------------------

    print_time_message("Copying good and bad images...")

    try:
        for result in tqdm(
            results,
            total=len(results),
            desc=f"Copying {flight_name}",
            unit="image"
        ):

            result["copied_to_bad"] = None
            result["copied_to_good"] = None

            if result["status"] != "ok":
                n_failed += 1
                continue

            image_path = Path(result["input_path"])

            if result["is_bad"]:
                n_bad += 1

                copied_to_bad = copy_preserving_structure(
                    image_path,
                    input_root,
                    bad_output_root
                )

                result["copied_to_bad"] = str(copied_to_bad)

            else:
                n_good += 1

                copied_to_good = copy_preserving_structure(
                    image_path,
                    input_root,
                    good_output_root
                )

                result["copied_to_good"] = str(copied_to_good)

            if result["is_blurry"]:
                n_blurry += 1

            if result["is_low_edge_strength"]:
                n_low_edge_strength += 1

            if result["is_low_edge_strength_p95"]:
                n_low_edge_strength_p95 += 1

            if result["is_low_edge_strength_p99"]:
                n_low_edge_strength_p99 += 1

            if result["is_empty"]:
                n_empty += 1

    except KeyboardInterrupt:
        print()
        print_time_message("KeyboardInterrupt detected during copying.")
        print_time_message("Saving partial copied results for this flight...")

        if len(results) > 0:
            partial_df = pd.DataFrame(results)

            partial_df = partial_df.sort_values(
                by=["flight", "relative_path"],
                ascending=True
            ).reset_index(drop=True)

            partial_df.to_csv(partial_summary_csv, index=False)

            print_time_message(f"Partial CSV saved to: {partial_summary_csv}")
            print_time_message(f"Partial results saved for {len(results)} images.")

        raise

    print()
    print_time_message("Finished copying images.")

    # --------------------------------------------------------
    # Build summary DataFrame
    # --------------------------------------------------------

    print_time_message("Building summary DataFrame...")

    summary_df = pd.DataFrame(results)

    summary_df = summary_df.sort_values(
        by=["flight", "relative_path"],
        ascending=True
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save one per-image summary CSV for this flight
    # --------------------------------------------------------

    print_time_message("Saving per-flight summary CSV...")

    summary_df.to_csv(summary_csv, index=False)

    print_time_message(f"Per-flight summary CSV saved to: {summary_csv}")

    # --------------------------------------------------------
    # Save diagnostic PNG plots for this flight
    # --------------------------------------------------------

    print_time_message("Making diagnostic plots for this flight...")

    make_good_bad_histograms(
        summary_df,
        flight_dir,
        blur_threshold,
        edge_strength_p99_threshold,
        edge_strength_p95_threshold
    )

    # --------------------------------------------------------
    # Print flight summary
    # --------------------------------------------------------

    bad_reason_summary = (
        summary_df["bad_reason"]
        .value_counts(dropna=False)
        .rename_axis("bad_reason")
        .reset_index(name="count")
    )

    elapsed_seconds = time.time() - flight_start_time
    elapsed_minutes = elapsed_seconds / 60.0

    print()
    print_time_message(f"Finished checking CPI images for flight: {flight_name}")
    print(f"Total images checked: {n_total}")
    print(f"Good images copied: {n_good}")
    print(f"Bad images copied: {n_bad}")
    print(f"Blurry images detected by Laplacian: {n_blurry}")
    print(f"Low-edge-strength images detected: {n_low_edge_strength}")
    print(f"Low-edge-strength p95 images detected: {n_low_edge_strength_p95}")
    print(f"Low-edge-strength p99 images detected: {n_low_edge_strength_p99}")
    print(f"Empty images detected: {n_empty}")
    print(f"Images failed to read: {n_failed}")
    print()
    print(f"Blur threshold used: {blur_threshold}")
    print(f"Edge strength p95 threshold used: {edge_strength_p95_threshold}")
    print(f"Edge strength p99 threshold used: {edge_strength_p99_threshold}")
    print(f"Empty contrast threshold used: {empty_contrast_threshold}")
    print(f"Empty minimum particle area used: {empty_min_particle_area_px} px")
    print(f"Empty minimum dark fraction used: {empty_min_dark_fraction}")
    print()
    print(f"Input directory: {input_root}")
    print(f"Bad images directory: {bad_output_root}")
    print(f"Good images directory: {good_output_root}")
    print(f"Per-image summary CSV saved to: {summary_csv}")
    print(f"Elapsed time for this flight: {elapsed_minutes:.2f} minutes")
    print()
    print("Bad-reason summary:")
    print(bad_reason_summary.to_string(index=False))

    return summary_df


# ============================================================
# Main
# ============================================================

def main():

    script_start_time = time.time()

    print_time_message("Starting full-year CPI good/bad image filtering script.")
    print_time_message(f"FlightData root: {flightdata_root}")
    print_time_message(f"Number of requested flight directories: {len(flight_directories)}")
    print_time_message(f"Input directory name: {input_directory_name}")
    print_time_message(f"Bad output directory name: {bad_output_directory_name}")
    print_time_message(f"Good output directory name: {good_output_directory_name}")
    print_time_message(f"Combined summary CSV: {combined_summary_csv}")
    print()

    all_summary_dfs = []

    processed_flights = []
    skipped_flights = []

    try:
        for flight_name in flight_directories:

            summary_df = process_one_flight(flight_name)

            if summary_df is None:
                skipped_flights.append(flight_name)
            else:
                processed_flights.append(flight_name)
                all_summary_dfs.append(summary_df)

    except KeyboardInterrupt:
        print()
        print_time_message("KeyboardInterrupt detected. Stopping full-year processing.")
        print_time_message("Any completed flight-level CSV files remain saved.")
        print_time_message("Combined summary will be created only from fully completed flights so far.")
        print()

    # --------------------------------------------------------
    # Save combined summary across fully completed flights
    # --------------------------------------------------------

    if len(all_summary_dfs) > 0:

        print_time_message("Building combined summary DataFrame from completed flights...")

        combined_summary_df = pd.concat(
            all_summary_dfs,
            ignore_index=True
        )

        combined_summary_df = combined_summary_df.sort_values(
            by=["flight", "relative_path"],
            ascending=True
        ).reset_index(drop=True)

        print_time_message("Saving combined all-flight summary CSV...")

        combined_summary_df.to_csv(combined_summary_csv, index=False)

        elapsed_seconds = time.time() - script_start_time
        elapsed_minutes = elapsed_seconds / 60.0

        print()
        print("============================================================")
        print_time_message("ALL-FLIGHT SUMMARY")
        print("============================================================")
        print(f"Processed flights: {len(processed_flights)}")
        print(f"Skipped flights: {len(skipped_flights)}")
        print(f"Total images checked: {len(combined_summary_df)}")
        print(f"Total good images: {int((combined_summary_df['is_good'] == True).sum())}")
        print(f"Total bad images: {int((combined_summary_df['is_bad'] == True).sum())}")
        print(f"Total failed-to-read images: {int((combined_summary_df['status'] != 'ok').sum())}")
        print()
        print(f"Combined summary CSV saved to: {combined_summary_csv}")
        print(f"Total elapsed time: {elapsed_minutes:.2f} minutes")

        print()
        print("Processed flight directories:")
        for flight_name in processed_flights:
            print(f"  {flight_name}")

        if len(skipped_flights) > 0:
            print()
            print("Skipped flight directories:")
            for flight_name in skipped_flights:
                print(f"  {flight_name}")

        print()
        print("Combined bad-reason summary:")
        combined_bad_reason_summary = (
            combined_summary_df["bad_reason"]
            .value_counts(dropna=False)
            .rename_axis("bad_reason")
            .reset_index(name="count")
        )
        print(combined_bad_reason_summary.to_string(index=False))

    else:
        print()
        print_time_message("No flights were fully processed.")
        print_time_message("No combined summary CSV was created.")


if __name__ == "__main__":
    main()