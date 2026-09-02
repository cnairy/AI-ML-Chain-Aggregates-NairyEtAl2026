# Assessment of Machine Learning Classification of Ice Crystal Chain Aggregates

Reproducibility code for the Nairy et al. *Journal of Atmospheric and Oceanic Technology* manuscript on classifying ice-crystal chain aggregates in aircraft-collected cloud-particle imagery.

## Repository contents

- `src/preprocessing/`: CPI descriptor extraction and image-quality filtering.
- `src/descriptor_models/`: Random Forest and XGBoost training, validation, Appendix-A diagnostics, and the 17 January 2022 export.
- `src/cnn/training/`: final 30-epoch, five-fold training scripts for ResNet18/34/50/101 and VGG16/19.
- `src/cnn/model_selection/`: validation-only epoch/architecture evaluation, external validation, and threshold-sensitivity analysis.
- `src/cnn/evaluation/`: final selected ResNet34 independent-test evaluation.
- `src/cnn/inference/`: campaign-wide ResNet34 inference.
- `src/figures/`: manuscript and supplemental figure/table scripts.
- `src/postprocessing/`: conversion of CNN predictions to the project time-series format.
- `outputs/manuscript_summary_tables/`: compact manuscript result tables assembled from script outputs.

The files in this first migration preserve the audited analysis versions and filenames. Machine-specific source paths were replaced by generic `/path/to/...` placeholders; update those values before running. A later cleanup can consolidate the near-duplicate architecture and year-specific scripts into configurable entry points after numerical equivalence is verified.

## Source data

The underlying public observations are the NASA GHRC DAAC **NCAR Particle Probes IMPACTS** data set: [doi:10.5067/IMPACTS/PROBES/DATA101](https://doi.org/10.5067/IMPACTS/PROBES/DATA101).

Large processed image collections, manually labeled splits, and model checkpoints are not stored in ordinary Git history. See [data/README.md](data/README.md) for the derived files and manifests needed for full reproduction.

## Environment

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate chain-aggregates-jtech
```

Some conversion scripts also depend on the project-specific `adpaa_python3` module, which is not installed by the environment file.

## Provenance and scope

[SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) maps every copied file to the exact path and commit in the private working repository. Historical duplicate scripts and scripts that selected epochs using the independent test set were deliberately excluded.
