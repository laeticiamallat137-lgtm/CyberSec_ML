"""Shared metrics aligned with PDF section 5 (macro-F1, per-class recall, benign FPR)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def benign_class_index(classes) -> int | None:
    for i, c in enumerate(classes):
        if str(c).upper() == "BENIGN":
            return i
    return None


def benign_fpr(y_true: np.ndarray, y_pred: np.ndarray, benign_idx: int | None) -> float:
    if benign_idx is None:
        return float("nan")
    mask = y_true == benign_idx
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_pred[mask] != benign_idx))


def security_metrics_dict(
    y_true: np.ndarray, y_pred: np.ndarray, le, zero_division: int = 0
) -> dict:
    benign_idx = benign_class_index(le.classes_)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=zero_division)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        "benign_fpr": benign_fpr(y_true, y_pred, benign_idx),
    }


def print_metrics_block(name: str, y_true: np.ndarray, y_pred: np.ndarray, le) -> None:
    print(f"\n--- {name} (validation) ---")
    metrics = security_metrics_dict(y_true, y_pred, le)
    print(
        f"macro-F1: {metrics['macro_f1']:.4f} | "
        f"weighted-F1: {metrics['weighted_f1']:.4f} | "
        f"benign-FPR: {metrics['benign_fpr']:.4f}"
    )
    print("\nClassification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=list(le.classes_), digits=4, zero_division=0
        )
    )
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix shape: {cm.shape}")


def print_comparison_table(rows: list[tuple[str, dict]]) -> None:
    print("\n" + "=" * 60)
    print("Comparison (validation): macro-F1 | weighted-F1 | benign-FPR")
    print("=" * 60)
    for name, m in rows:
        print(
            f"  {name:6s}  {m['macro_f1']:.4f}        {m['weighted_f1']:.4f}        {m['benign_fpr']:.4f}"
        )
    print("=" * 60)
