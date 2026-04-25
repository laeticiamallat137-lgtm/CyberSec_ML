# HGB Evaluation Report

## Objective
Compare the original HistGradientBoosting configuration vs an improved configuration
using the same refreshed pipeline artifacts and the same train/val/test split.

## What Changed
- Increased boosting iterations (`max_iter`: 300 -> 600).
- Increased tree capacity (`max_depth`: 8 -> 12, `max_leaf_nodes`: 63).
- Added regularization and smoother learning (`learning_rate`: 0.05 -> 0.03, `l2_regularization`: 1e-3).
- Added stronger stability controls (`min_samples_leaf`: 40, early stopping with `n_iter_no_change`: 20, `validation_fraction`: 0.1).
- Kept `class_weight='balanced'` and `random_state=42` for fair comparison.

## Historical HGB Snapshot (from `dashboard/metrics_summary.json`)

| Metric (Validation) | Historical |
|---|---:|
| Accuracy | 0.7571 |
| Macro F1 | 0.5816 |
| Weighted F1 | 0.7695 |
| Benign FPR | 0.3524 |

## Validation Metrics (Old vs New)

| Metric | Old Baseline | New Improved | Delta (New-Old) |
|---|---:|---:|---:|
| Accuracy | 0.7571 | 0.7650 | +0.0079 |
| Macro Precision | 0.5834 | 0.5853 | +0.0019 |
| Macro Recall | 0.6386 | 0.6439 | +0.0053 |
| Macro F1 | 0.5816 | 0.5875 | +0.0059 |
| Weighted Precision | 0.7959 | 0.7960 | +0.0001 |
| Weighted Recall | 0.7571 | 0.7650 | +0.0079 |
| Weighted F1 | 0.7695 | 0.7743 | +0.0048 |
| Benign FPR | 0.3524 | 0.3269 | -0.0255 |

## Test Metrics (Old vs New)

| Metric | Old Baseline | New Improved | Delta (New-Old) |
|---|---:|---:|---:|
| Accuracy | 0.7576 | 0.7650 | +0.0074 |
| Macro Precision | 0.5861 | 0.5873 | +0.0011 |
| Macro Recall | 0.6443 | 0.6464 | +0.0021 |
| Macro F1 | 0.5841 | 0.5892 | +0.0051 |
| Weighted Precision | 0.7968 | 0.7962 | -0.0006 |
| Weighted Recall | 0.7576 | 0.7650 | +0.0074 |
| Weighted F1 | 0.7701 | 0.7743 | +0.0042 |
| Benign FPR | 0.3556 | 0.3297 | -0.0259 |

## Training Time

| Model | Fit Time (seconds) |
|---|---:|
| Old Baseline HGB | 462.8 |
| New Improved HGB | 699.7 |



## Notes
- This report compares models trained on the same refreshed data pipeline output.
- Lower benign FPR is better (fewer benign flows incorrectly flagged as attack).
