# Source manifest

All migrated analysis scripts came from the private working repository `cnairy/Chain-Aggregation-Nairy-PHD` at commit:

```text
2f221bfa58a06b874990a846c9da7991ab80e845
```

Machine-specific path prefixes were replaced with generic `/path/to/...` placeholders. Apart from that path sanitization and repository placement, the scripts retain their audited source content and filenames.

| Publication-repository path | Original source path |
|---|---|
| `src/descriptor_models/RF_XGB_JTECH_Appendix_A_audit_all_treatments.py` | `src/scripts_python/code_audit_staging/local_scripts/RF_XGB_JTECH_Appendix_A_audit_all_treatments.py` |
| `src/descriptor_models/RF_XGBoost_JTECH_export_Jan17.py` | `src/scripts_python/code_audit_staging/local_scripts/RF_XGBoost_JTECH_export_Jan17.py` |
| `src/figures/PDF_chain_nonchains_Fig2_cohensd.py` | `src/scripts_python/code_audit_staging/local_scripts/PDF_chain_nonchains_Fig2_cohensd.py` |
| `src/figures/JTECH_plot_confusion_matrices_only.py` | `src/scripts_python/code_audit_staging/local_scripts/JTECH_plot_confusion_matrices_only.py` |
| `src/figures/make_descriptor_table_JTECH_all.py` | `src/scripts_python/code_audit_staging/local_scripts/make_descriptor_table_JTECH_all.py` |
| `src/cnn/training/CNN-ResNet18_Training_20251021-noleakage-cv-30ep.py` | `src/scripts_python/CNN/CNN_training_val_eval_JTECH/ResNet18/CNN-ResNet18_Training_20251021-noleakage-cv-30ep.py` |
| `src/cnn/training/CNN-ResNet34_Training_20251021-noleakage-cv-30ep.py` | `src/scripts_python/CNN/CNN_training_val_eval_JTECH/ResNet34/CNN-ResNet34_Training_20251021-noleakage-cv-30ep.py` |
| `src/cnn/training/CNN-ResNet50_Training_20251104-noleakage-cv_30ep.py` | `src/scripts_python/CNN/CNN_training_val_eval_JTECH/ResNet50/CNN-ResNet50_Training_20251104-noleakage-cv_30ep.py` |
| `src/cnn/training/CNN-ResNet101_Training_20251105-noleakage-cv_30ep.py` | `src/scripts_python/CNN/CNN_training_val_eval_JTECH/ResNet101/CNN-ResNet101_Training_20251105-noleakage-cv_30ep.py` |
| `src/cnn/training/CNN-VGG16_Training_20251014-noleakage-cv.py` | `src/scripts_python/CNN/CNN_training_val_eval_JTECH/VGG16/CNN-VGG16_Training_20251014-noleakage-cv.py` |
| `src/cnn/training/CNN-VGG19_Training_20251106-noleakage-cv-30ep.py` | `src/scripts_python/CNN/CNN_training_val_eval_JTECH/VGG19/CNN-VGG19_Training_20251106-noleakage-cv-30ep.py` |
| `src/cnn/model_selection/eval_resnet18_val_constant_threshold.py` | `src/scripts_python/code_audit_staging/nas_scripts/eval_resnet18_val_constant_threshold.py` |
| `src/cnn/model_selection/eval_ResNet34_val_constant_thresh.py` | `src/scripts_python/code_audit_staging/nas_scripts/eval_ResNet34_val_constant_thresh.py` |
| `src/cnn/model_selection/eval_ResNet50_val_constant_thresh.py` | `src/scripts_python/code_audit_staging/nas_scripts/eval_ResNet50_val_constant_thresh.py` |
| `src/cnn/model_selection/eval_ResNet101_val_constant_thresh.py` | `src/scripts_python/code_audit_staging/nas_scripts/eval_ResNet101_val_constant_thresh.py` |
| `src/cnn/model_selection/eval_VGG16_val_constant_thresh.py` | `src/scripts_python/code_audit_staging/nas_scripts/eval_VGG16_val_constant_thresh.py` |
| `src/cnn/model_selection/eval_VGG19_val_constant_thresh.py` | `src/scripts_python/code_audit_staging/nas_scripts/eval_VGG19_val_constant_thresh.py` |
| `src/cnn/model_selection/external_validate_all_models-30ep_20260323.py` | `src/scripts_python/code_audit_staging/nas_scripts/external_validate_all_models-30ep_20260323.py` |
| `src/cnn/model_selection/pick_best_cnn_and_epoch.py` | `src/scripts_python/code_audit_staging/nas_scripts/pick_best_cnn_and_epoch.py` |
| `src/cnn/model_selection/ResNet34_threshold_sensitivity_analysis.py` | `src/scripts_python/code_audit_staging/nas_scripts/ResNet34_threshold_sensitivity_analysis.py` |
| `src/figures/stats_comparison_plots_30ep_20260324.py` | `src/scripts_python/code_audit_staging/nas_scripts/stats_comparison_plots_30ep_20260324.py` |
| `src/figures/make_validation_boxplots_and_resnet34_calibration_envelope.py` | `src/scripts_python/code_audit_staging/nas_scripts/make_validation_boxplots_and_resnet34_calibration_envelope.py` |
| `src/cnn/evaluation/ResNet34_evaluate_CNN_testset_20260325.py` | `src/scripts_python/code_audit_staging/nas_scripts/ResNet34_evaluate_CNN_testset_20260325.py` |
| `src/cnn/inference/CNN-ResNet34_Process-Unlabeled_20260609-noleakage_filtered_images.py` | `src/scripts_python/CNN/CNN-ResNet34_Process-Unlabeled_20260609-noleakage_filtered_images.py` |
| `src/figures/GRAD-CAM_Saliency_plot_JTECH_1x2_20260401.py` | `src/scripts_python/CNN/plotting/GRAD-CAM_Saliency_plot_JTECH_1x2_20260401.py` |
| `src/figures/JTECH_2x2_training-validation_figure.py` | `src/scripts_python/CNN/plotting/JTECH_2x2_training-validation_figure.py` |
| `src/figures/JTECH_supplemental_learning_curves_examples_20260818.py` | `src/scripts_python/CNN/plotting/JTECH_supplemental_learning_curves_examples_20260818.py` |
| `src/figures/make_external_val_boxplots_20260323.py` | `src/scripts_python/CNN/plotting/make_external_val_boxplots_20260323.py` |
| `src/figures/plot_calibration_envelopes_all_models.py` | `src/scripts_python/CNN/plotting/plot_calibration_envelopes_all_models.py` |
| `src/cnn/model_selection/rank_all_models_two_stage_f1_then_precision.py` | `src/scripts_python/CNN/plotting/rank_all_models_two_stage_f1_then_precision.py` |
| `src/preprocessing/cpi-img_process_Nairy.py` | `src/scripts_python/hawkeye-cpi_scripts/cpi-img_process_Nairy.py` |
| `src/preprocessing/test_removing_blurry_cpi-images_full-year_2020.py` | `src/scripts_python/cpi_filtering/test_removing_blurry_cpi-images_full-year_2020.py` |
| `src/preprocessing/test_removing_blurry_cpi-images_full-year_2022.py` | `src/scripts_python/cpi_filtering/test_removing_blurry_cpi-images_full-year_2022.py` |
| `src/preprocessing/test_removing_blurry_cpi-images_full-year_2023.py` | `src/scripts_python/cpi_filtering/test_removing_blurry_cpi-images_full-year_2023.py` |
| `src/postprocessing/convert_CNN-preds_2nasa.py` | `src/scripts_python/file_conversion_scripts/convert_CNN-preds_2nasa.py` |

## Deliberately excluded

- Older dated duplicates superseded by the files above.
- `eval_all_epochs_*_testset.py`: these compare epochs on the independent test set and are not part of the valid model-selection path.
- `pick_best_epoch_per_model.py` and `pick_best_epoch_per_model2.py`: these select epochs from independent-test outputs.
- `extract_val_operating_confusions.py`: it recovers values from raster figures with OCR instead of exporting them directly.
- The three separate no-resampling/ADASYN/SMOTE RF/XGBoost scripts, which are superseded by the consolidated Appendix-A workflow.
- Downstream aircraft/radar/environmental analyses belonging to other manuscripts.
