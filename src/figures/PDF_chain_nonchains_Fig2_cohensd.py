#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#
# Name:
#   PDF_chain_nonchains_Fig2_cohensd.py
#
# Purpose:
#   Plot PDF distributions of chain and non-chain aggregate particle properties
#   for Figure 2, with Cohen's d displayed in each panel.
#
#   Included descriptors:
#       Dmax
#       Circularity
#       Area Ratio
#       Complexity
#       Curl
#       Solidity
#       Compactness
#       Fine Detail
#
#   Fractal dimension and number of monomers are NOT plotted.
#
# Syntax:
#   python3 PDF_chain_nonchains_Fig2_cohensd.py
#
"""

# IMPORTS
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# COHEN'S d
# ============================================================

def cohens_d(chain_values, nonchain_values):
    """
    Calculate Cohen's d using the pooled standard deviation.

    Sign convention:
        positive d -> mean is larger for chain aggregates
        negative d -> mean is smaller for chain aggregates
    """
    x = np.asarray(chain_values, dtype=float)
    y = np.asarray(nonchain_values, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    pooled_sd = np.sqrt(
        ((len(x) - 1) * var_x + (len(y) - 1) * var_y)
        / (len(x) + len(y) - 2)
    )

    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

date = '2025-06-26'
datestr = date.replace('-', '')

# Load total particle property file
infile = (
    '/path/to/local_workspace/Documents/phd/data/IMPACTS/CPI_Particle_Properties/Filtered_Final_20250626/'
    f'cpi.total-particle.properties.filtered.{datestr}.nasa'
)

data_df = pd.read_csv(
    infile,
    skiprows=43,
    sep=r'\s+',
    low_memory=False
)

# Columns needed for filtering, matching, and plotting.
# Fractal dimension is retained ONLY because it is part of the existing QC filter.
# It is NOT plotted.
cols = [
    'dmax',
    'circularity',
    'area_ratio',
    'complexity',
    'curl',
    'pct_touch',
    'solidity',
    'frac_dim',
    'compactness',
    'fine_detail',
    'img_num',
    'aspe_ratio'
]

data_df = data_df.drop(0).set_index('Time')
data_df = data_df[cols].copy().dropna().astype(float)

# Remove bad images
data_df_filtered = data_df[
    (data_df['pct_touch'] < 0.01) &
    (data_df['curl'] < 9999) &
    (data_df['frac_dim'] < 9999) &
    (data_df['compactness'] < 15) &
    (data_df['fine_detail'] < 50) &
    (data_df['complexity'] < 5)
].reset_index()


# ============================================================
# LOAD MANUALLY IDENTIFIED CHAIN PARTICLES
# ============================================================

infile2 = (
    '/path/to/local_workspace/Documents/phd/data/IMPACTS/'
    'CPI_Particle_Properties/F-T2chain_merged_20250627/'
    f'cpi.FT2chain.properties.filtered.{datestr}.nasa'
)

chain_df = pd.read_csv(
    infile2,
    skiprows=43,
    sep=r'\s+',
    low_memory=False
)

chain_df = chain_df.drop(0).set_index('Time')[cols].copy().dropna().astype(float)

# Remove bad images
chain_df_filtered = chain_df[
    (chain_df['pct_touch'] < 0.01) &
    (chain_df['curl'] < 9999) &
    (chain_df['frac_dim'] < 9999) &
    (chain_df['compactness'] < 15) &
    (chain_df['fine_detail'] < 40) &
    (chain_df['complexity'] < 5)
].reset_index()


# ============================================================
# CREATE NON-CHAIN DATASET
# ============================================================

nonchain_df_filtered = data_df_filtered.merge(
    chain_df_filtered,
    how='left',
    on=cols,
    indicator=True
)

nonchain_df_filtered = nonchain_df_filtered[
    nonchain_df_filtered['_merge'] == 'left_only'
].drop(columns=['_merge'])


# ============================================================
# FIGURE 2
# ============================================================

# ONLY these eight descriptors are plotted.
columns_to_plot = [
    'dmax',
    'circularity',
    'area_ratio',
    'complexity',
    'curl',
    'solidity',
    'compactness',
    'fine_detail'
]

plot_titles = [
    r'D$_{max}$',
    'Circularity',
    'Area Ratio',
    'Complexity',
    'Curl',
    'Solidity',
    'Compactness',
    'Fine Detail'
]

sns.set(style="whitegrid")

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(15, 12),
    dpi=300
)

axes = axes.flatten()

for i, col in enumerate(columns_to_plot):

    sns.kdeplot(
        chain_df_filtered[col],
        color='blue',
        fill=True,
        alpha=0.5,
        ax=axes[i],
        common_norm=True
    )

    sns.kdeplot(
        nonchain_df_filtered[col],
        color='orange',
        fill=True,
        alpha=0.5,
        ax=axes[i],
        common_norm=True
    )

    # Cohen's d: chain minus non-chain.
    d = cohens_d(
        chain_df_filtered[col],
        nonchain_df_filtered[col]
    )

    if np.isfinite(d):
        d_text = rf"Cohen's $d$ = {d:+.2f}"
    else:
        d_text = r"Cohen's $d$ = N/A"

    axes[i].text(
        0.97,
        0.95,
        d_text,
        transform=axes[i].transAxes,
        ha='right',
        va='top',
        fontsize=14,
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='white',
            edgecolor='0.6',
            alpha=0.85
        )
    )

    axes[i].set_title(plot_titles[i], fontsize=20)
    axes[i].tick_params(
        axis='both',
        which='major',
        labelsize=16
    )

    if i % 3 == 0:
        axes[i].set_ylabel('Density', fontsize=16)
    else:
        axes[i].set_ylabel('')

    if col == 'dmax':
        axes[i].set_xlabel('mm', fontsize=16)

    elif col == 'curl':
        axes[i].set_xlim(0, 15)
        axes[i].set_xlabel('')

    elif col == 'fine_detail':
        axes[i].set_xlim(0, 30)
        axes[i].set_xlabel('')

    else:
        axes[i].set_xlabel('')


# ============================================================
# LEGEND
# ============================================================

# Eight plotted descriptors leave the bottom-right panel available for legend.
legend_ax = axes[-1]
legend_ax.axis('off')

handles = [
    plt.Line2D(
        [0], [0],
        color='blue',
        lw=6,
        label='Chain Aggregates'
    ),
    plt.Line2D(
        [0], [0],
        color='orange',
        lw=6,
        label='Non-chain Particles'
    )
]

legend_ax.legend(
    handles=handles,
    loc='center',
    ncol=1,
    frameon=True,
    fontsize=20
)

plt.tight_layout()
plt.show()
