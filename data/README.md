# Data and model artifacts

## Public source observations

Cite and obtain the original aircraft particle-probe observations from the NASA GHRC DAAC **NCAR Particle Probes IMPACTS** data set:

- DOI: [10.5067/IMPACTS/PROBES/DATA101](https://doi.org/10.5067/IMPACTS/PROBES/DATA101)

Private server paths used during analysis are working locations, not data citations.

## Compact derived files to preserve with a release

The following small outputs should be added here when their final values have been regenerated and checked against the manuscript:

- `RF_XGB_unseen_Jan17_2022_particles.csv`
- `RF_XGB_unseen_Jan17_2022_metrics.csv`
- Appendix-A diagnostic CSV files, including `dataset_QC_counts.csv`
- final CNN independent-test metrics and per-particle predictions
- training/validation split manifests and label manifests
- selected epoch and operating-threshold records
- checksums for every externally archived image collection and checkpoint

## Large or redistribution-sensitive artifacts

Do not add the full image corpus or large model checkpoints to ordinary Git history. Deposit them in a versioned, DOI-bearing archive (institutional repository or a service such as Zenodo) if redistribution is permitted. If individual CPI images cannot be redistributed, publish label/split manifests and document how to regenerate the images from the NASA source observations.

The repository should link to the final archive DOI once it exists.
