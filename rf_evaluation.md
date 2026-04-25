# RF Evaluation Report

## Objective
Compare the original Random Forest baseline vs the improved Random Forest configuration
using the same refreshed pipeline artifacts and the same train/val/test split.

## What Changed
- Tuned Random Forest hyperparameters to improve multiclass intrusion detection performance.
- Trained and compared baseline and improved configurations under the same preprocessing/split pipeline.
- Selected the improved model as the best RF candidate based on validation and test performance.

## Validation Metrics (Baseline vs Best)

| Metric | Baseline | Best Improved | Delta (Best-Baseline) |
|---|---:|---:|---:|
| Accuracy | 0.7431 | 0.7779 | +0.0348 |
| Macro Precision | 0.5931 | 0.5965 | +0.0034 |
| Macro Recall | 0.6254 | 0.6273 | +0.0019 |
| Macro F1 | 0.5687 | 0.5964 | +0.0277 |
| Weighted Precision | 0.7961 | 0.7925 | -0.0036 |
| Weighted Recall | 0.7431 | 0.7779 | +0.0348 |
| Weighted F1 | 0.7506 | 0.7755 | +0.0249 |
| Benign FPR | 0.0127 | 0.0148 | +0.0021 |

## Test Metrics (Baseline vs Best)

| Metric | Baseline | Best Improved | Delta (Best-Baseline) |
|---|---:|---:|---:|
| Accuracy | 0.7445 | 0.7796 | +0.0351 |
| Macro Precision | 0.5956 | 0.5986 | +0.0030 |
| Macro Recall | 0.6250 | 0.6284 | +0.0034 |
| Macro F1 | 0.5707 | 0.5987 | +0.0280 |
| Weighted Precision | 0.7977 | 0.7940 | -0.0037 |
| Weighted Recall | 0.7445 | 0.7796 | +0.0351 |
| Weighted F1 | 0.7524 | 0.7774 | +0.0250 |
| Benign FPR | 0.0122 | 0.0144 | +0.0022 |

## Training Time

| Model | Fit Time (seconds) |
|---|---:|
| Baseline RF | 310.97 |
| Best Improved RF | 871.44 |



## Notes
- Macro and weighted F1 improved clearly on both validation and test.
- Benign FPR increased slightly in the improved model (+0.0021 validation, +0.0022 test), which should be discussed as a tradeoff against broader detection gains.
- This report uses the same split and preprocessing pipeline for fair comparison.
