# MLP Evaluation Report

## Objective
Compare the original MLP baseline against later MLP experiments using the same
security-focused metrics: accuracy, macro F1, weighted F1, benign recall, and
benign false positive rate (FPR).

## Experiment Progression
- The first MLP baseline performed poorly, especially on benign recall and benign FPR.
- `mlp_exp1_sqrt` was the first major improvement over the baseline.
- `mlp` was also improved based on `mlp_exp1_sqrt`, but did not outperform it overall.
- `mlp_exp3` was another improvement path based on `mlp_exp1_sqrt`, because the original MLP baseline was weak.
- `mlp_exp4` was improved based on `mlp_exp3` and became the strongest final MLP candidate by accuracy and weighted F1.

## Original Baseline Metrics

| Metric | Baseline |
|---|---:|
| Accuracy | 0.6741 |
| Macro F1 | 0.5167 |
| Weighted F1 | 0.6780 |
| Benign Recall | 0.4952 |
| Benign FPR | 0.5048 |

## MLP Experiment Metrics

| Model | Accuracy | Macro F1 | Weighted F1 | Benign Recall | Benign FPR |
|---|---:|---:|---:|---:|---:|
| `mlp_exp1_sqrt` | 0.7797 | 0.6054 | 0.7683 | 0.8559 | 0.1441 |
| `mlp` | 0.7785 | 0.5988 | 0.7672 | 0.8483 | 0.1517 |
| `mlp_exp3` | 0.7812 | 0.6083 | 0.7733 | 0.8306 | 0.1694 |
| `mlp_exp4` | 0.7828 | 0.6078 | 0.7750 | 0.8379 | 0.1621 |

## Improvement vs Original Baseline

| Model | Accuracy Delta | Macro F1 Delta | Weighted F1 Delta | Benign Recall Delta | Benign FPR Delta |
|---|---:|---:|---:|---:|---:|
| `mlp_exp1_sqrt` | +0.1056 | +0.0887 | +0.0903 | +0.3607 | -0.3607 |
| `mlp` | +0.1044 | +0.0821 | +0.0892 | +0.3531 | -0.3531 |
| `mlp_exp3` | +0.1071 | +0.0916 | +0.0953 | +0.3354 | -0.3354 |
| `mlp_exp4` | +0.1087 | +0.0911 | +0.0970 | +0.3427 | -0.3427 |

## Exp3 vs Exp4

| Metric | `mlp_exp3` | `mlp_exp4` | Delta (Exp4-Exp3) |
|---|---:|---:|---:|
| Accuracy | 0.7812 | 0.7828 | +0.0016 |
| Macro F1 | 0.6083 | 0.6078 | -0.0005 |
| Weighted F1 | 0.7733 | 0.7750 | +0.0017 |
| Benign Recall | 0.8306 | 0.8379 | +0.0073 |
| Benign FPR | 0.1694 | 0.1621 | -0.0073 |

## Notes
- Lower benign FPR is better because it means fewer benign flows are incorrectly flagged as attacks.
- `mlp_exp1_sqrt` achieved the best benign recall and lowest benign FPR among the improved MLP experiments.
- `mlp_exp3` achieved the best macro F1.
- `mlp_exp4` achieved the best accuracy and weighted F1, while also improving benign recall and benign FPR compared with `mlp_exp3`.
- The final MLP choice depends on the priority: `mlp_exp4` is strongest overall, while `mlp_exp1_sqrt` is strongest for minimizing benign false positives.
