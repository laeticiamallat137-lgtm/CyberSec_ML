from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from evaluation import security_metrics_dict

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DASHBOARD_METRICS = ROOT / "dashboard" / "metrics_summary.json"
REPORT_PATH = ROOT / "hgb_evaluation.md"

BASELINE_MODEL_PATH = MODELS_DIR / "gradient_boosting_baseline.joblib"
IMPROVED_MODEL_PATH = MODELS_DIR / "gradient_boosting.joblib"


def load_arrays():
    X_train = joblib.load(PROCESSED_DIR / "X_train.joblib")
    y_train = joblib.load(PROCESSED_DIR / "y_train.joblib")
    X_val = joblib.load(PROCESSED_DIR / "X_val.joblib")
    y_val = joblib.load(PROCESSED_DIR / "y_val.joblib")
    X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
    y_test = joblib.load(PROCESSED_DIR / "y_test.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    return X_train, y_train, X_val, y_val, X_test, y_test, le


def baseline_model():
    # Original project settings before improvement.
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=8,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=0,
    )


def improved_model():
    # Updated settings used in train.py.
    return HistGradientBoostingClassifier(
        max_iter=600,
        max_depth=12,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        learning_rate=0.03,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=42,
        verbose=0,
    )


def compute_metrics(y_true, y_pred, le):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    sec = security_metrics_dict(y_true, y_pred, le)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "benign_fpr": float(sec["benign_fpr"]),
    }


def fit_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, le):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    return model, {
        "fit_seconds": float(fit_sec),
        "val": compute_metrics(y_val, y_val_pred, le),
        "test": compute_metrics(y_test, y_test_pred, le),
    }


def try_historical_hgb_metrics():
    if not DASHBOARD_METRICS.is_file():
        return None
    with open(DASHBOARD_METRICS, encoding="utf-8") as f:
        data = json.load(f)
    for m in data.get("models", []):
        if m.get("id") == "hgb":
            return {
                "val_accuracy": m.get("val_accuracy"),
                "val_macro_f1": m.get("val_macro_f1"),
                "val_weighted_f1": m.get("val_weighted_f1"),
                "val_benign_fpr": m.get("val_benign_fpr"),
            }
    return None


def fmt(v):
    return f"{v:.4f}"


def delta(new_v, old_v):
    return f"{new_v - old_v:+.4f}"


def write_report(baseline, improved, historical):
    b_val = baseline["val"]
    n_val = improved["val"]
    b_test = baseline["test"]
    n_test = improved["test"]
    lines = [
        "# HGB Evaluation Report",
        "",
        "## Objective",
        "Compare the original HistGradientBoosting configuration vs an improved configuration",
        "using the same refreshed pipeline artifacts and the same train/val/test split.",
        "",
        "## What Changed",
        "- Increased boosting iterations (`max_iter`: 300 -> 600).",
        "- Increased tree capacity (`max_depth`: 8 -> 12, `max_leaf_nodes`: 63).",
        "- Added regularization and smoother learning (`learning_rate`: 0.05 -> 0.03, `l2_regularization`: 1e-3).",
        "- Added stronger stability controls (`min_samples_leaf`: 40, early stopping with `n_iter_no_change`: 20, `validation_fraction`: 0.1).",
        "- Kept `class_weight='balanced'` and `random_state=42` for fair comparison.",
        "",
    ]
    if historical is not None:
        lines.extend(
            [
                "## Historical HGB Snapshot (from `dashboard/metrics_summary.json`)",
                "",
                "| Metric (Validation) | Historical |",
                "|---|---:|",
                f"| Accuracy | {fmt(historical['val_accuracy'])} |",
                f"| Macro F1 | {fmt(historical['val_macro_f1'])} |",
                f"| Weighted F1 | {fmt(historical['val_weighted_f1'])} |",
                f"| Benign FPR | {fmt(historical['val_benign_fpr'])} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Validation Metrics (Old vs New)",
            "",
            "| Metric | Old Baseline | New Improved | Delta (New-Old) |",
            "|---|---:|---:|---:|",
            f"| Accuracy | {fmt(b_val['accuracy'])} | {fmt(n_val['accuracy'])} | {delta(n_val['accuracy'], b_val['accuracy'])} |",
            f"| Macro Precision | {fmt(b_val['macro_precision'])} | {fmt(n_val['macro_precision'])} | {delta(n_val['macro_precision'], b_val['macro_precision'])} |",
            f"| Macro Recall | {fmt(b_val['macro_recall'])} | {fmt(n_val['macro_recall'])} | {delta(n_val['macro_recall'], b_val['macro_recall'])} |",
            f"| Macro F1 | {fmt(b_val['macro_f1'])} | {fmt(n_val['macro_f1'])} | {delta(n_val['macro_f1'], b_val['macro_f1'])} |",
            f"| Weighted Precision | {fmt(b_val['weighted_precision'])} | {fmt(n_val['weighted_precision'])} | {delta(n_val['weighted_precision'], b_val['weighted_precision'])} |",
            f"| Weighted Recall | {fmt(b_val['weighted_recall'])} | {fmt(n_val['weighted_recall'])} | {delta(n_val['weighted_recall'], b_val['weighted_recall'])} |",
            f"| Weighted F1 | {fmt(b_val['weighted_f1'])} | {fmt(n_val['weighted_f1'])} | {delta(n_val['weighted_f1'], b_val['weighted_f1'])} |",
            f"| Benign FPR | {fmt(b_val['benign_fpr'])} | {fmt(n_val['benign_fpr'])} | {delta(n_val['benign_fpr'], b_val['benign_fpr'])} |",
            "",
            "## Test Metrics (Old vs New)",
            "",
            "| Metric | Old Baseline | New Improved | Delta (New-Old) |",
            "|---|---:|---:|---:|",
            f"| Accuracy | {fmt(b_test['accuracy'])} | {fmt(n_test['accuracy'])} | {delta(n_test['accuracy'], b_test['accuracy'])} |",
            f"| Macro Precision | {fmt(b_test['macro_precision'])} | {fmt(n_test['macro_precision'])} | {delta(n_test['macro_precision'], b_test['macro_precision'])} |",
            f"| Macro Recall | {fmt(b_test['macro_recall'])} | {fmt(n_test['macro_recall'])} | {delta(n_test['macro_recall'], b_test['macro_recall'])} |",
            f"| Macro F1 | {fmt(b_test['macro_f1'])} | {fmt(n_test['macro_f1'])} | {delta(n_test['macro_f1'], b_test['macro_f1'])} |",
            f"| Weighted Precision | {fmt(b_test['weighted_precision'])} | {fmt(n_test['weighted_precision'])} | {delta(n_test['weighted_precision'], b_test['weighted_precision'])} |",
            f"| Weighted Recall | {fmt(b_test['weighted_recall'])} | {fmt(n_test['weighted_recall'])} | {delta(n_test['weighted_recall'], b_test['weighted_recall'])} |",
            f"| Weighted F1 | {fmt(b_test['weighted_f1'])} | {fmt(n_test['weighted_f1'])} | {delta(n_test['weighted_f1'], b_test['weighted_f1'])} |",
            f"| Benign FPR | {fmt(b_test['benign_fpr'])} | {fmt(n_test['benign_fpr'])} | {delta(n_test['benign_fpr'], b_test['benign_fpr'])} |",
            "",
            "## Training Time",
            "",
            "| Model | Fit Time (seconds) |",
            "|---|---:|",
            f"| Old Baseline HGB | {baseline['fit_seconds']:.1f} |",
            f"| New Improved HGB | {improved['fit_seconds']:.1f} |",
            "",
            "## Saved Artifacts",
            f"- Old baseline model: `{BASELINE_MODEL_PATH.as_posix()}`",
            f"- New improved model (production default): `{IMPROVED_MODEL_PATH.as_posix()}`",
            "",
            "## Notes",
            "- This report compares models trained on the same refreshed data pipeline output.",
            "- Lower benign FPR is better (fewer benign flows incorrectly flagged as attack).",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, le = load_arrays()
    print("Training baseline HGB...")
    baseline_model_fitted, baseline = fit_and_eval(
        baseline_model(), X_train, y_train, X_val, y_val, X_test, y_test, le
    )
    joblib.dump(baseline_model_fitted, BASELINE_MODEL_PATH)

    print("Training improved HGB...")
    improved_model_fitted, improved = fit_and_eval(
        improved_model(), X_train, y_train, X_val, y_val, X_test, y_test, le
    )
    joblib.dump(improved_model_fitted, IMPROVED_MODEL_PATH)

    historical = try_historical_hgb_metrics()
    write_report(baseline, improved, historical)
    print(f"Report written to: {REPORT_PATH}")
    print(f"Saved baseline model to: {BASELINE_MODEL_PATH}")
    print(f"Saved improved model to: {IMPROVED_MODEL_PATH}")


if __name__ == "__main__":
    main()
