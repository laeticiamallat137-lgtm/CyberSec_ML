import os
import json
import time
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
DASHBOARD_DIR = "dashboard"

X_TRAIN_FILE = os.path.join(PROCESSED_DIR, "X_train.joblib")
Y_TRAIN_FILE = os.path.join(PROCESSED_DIR, "y_train.joblib")
X_VAL_FILE = os.path.join(PROCESSED_DIR, "X_val.joblib")
Y_VAL_FILE = os.path.join(PROCESSED_DIR, "y_val.joblib")
X_TEST_FILE = os.path.join(PROCESSED_DIR, "X_test.joblib")
Y_TEST_FILE = os.path.join(PROCESSED_DIR, "y_test.joblib")

LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.joblib")
FEATURE_COLS_FILE = os.path.join(MODELS_DIR, "feature_cols.joblib")

BASELINE_MODEL_FILE = os.path.join(MODELS_DIR, "random_forest_baseline.joblib")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "random_forest_best.joblib")

REPORT_FILE = os.path.join(DASHBOARD_DIR, "random_forest_search_report.json")
SEARCH_RESULTS_FILE = os.path.join(DASHBOARD_DIR, "random_forest_search_results.csv")
FEATURE_IMPORTANCE_FILE = os.path.join(DASHBOARD_DIR, "random_forest_best_feature_importance.csv")


def load_data():
    X_train = joblib.load(X_TRAIN_FILE)
    y_train = joblib.load(Y_TRAIN_FILE)
    X_val = joblib.load(X_VAL_FILE)
    y_val = joblib.load(Y_VAL_FILE)
    X_test = joblib.load(X_TEST_FILE)
    y_test = joblib.load(Y_TEST_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    feature_cols = joblib.load(FEATURE_COLS_FILE)
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, feature_cols


def compute_benign_fpr(y_true, y_pred, benign_index, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fp = cm[:, benign_index].sum() - cm[benign_index, benign_index]
    tn = cm.sum() - (
        cm[benign_index, :].sum()
        + cm[:, benign_index].sum()
        - cm[benign_index, benign_index]
    )
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


def compute_metrics(y_true, y_pred, benign_index, n_classes):
    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    benign_fpr = compute_benign_fpr(y_true, y_pred, benign_index, n_classes)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "benign_fpr": float(benign_fpr),
    }


def metric_deltas(old_metrics, new_metrics):
    return {
        key: float(new_metrics[key] - old_metrics[key])
        for key in old_metrics.keys()
    }


def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    benign_index,
    n_classes,
):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_metrics = compute_metrics(y_val, y_val_pred, benign_index, n_classes)
    test_metrics = compute_metrics(y_test, y_test_pred, benign_index, n_classes)

    result = {
        "fit_time_seconds": round(fit_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }

    if hasattr(model, "oob_score_"):
        result["oob_score"] = float(model.oob_score_)

    return result


def score_for_selection(result):
    return (
        result["validation"]["macro_f1"],
        result["validation"]["weighted_f1"],
        -result["validation"]["benign_fpr"],
        result["validation"]["accuracy"],
    )


def save_feature_importance(model, feature_cols):
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importances.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print("\nTop 15 feature importances:")
    print(importances.head(15).to_string(index=False))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, le, feature_cols = load_data()

    print("Loaded data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    class_names = list(le.classes_)
    n_classes = len(class_names)

    benign_index = None
    for i, name in enumerate(class_names):
        if str(name).lower() == "benign":
            benign_index = i
            break

    if benign_index is None:
        raise ValueError("Could not find 'benign' class in label encoder.")

    print("\nTraining baseline Random Forest...")
    baseline_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    baseline_result = train_and_evaluate(
        baseline_model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        benign_index,
        n_classes,
    )
    joblib.dump(baseline_model, BASELINE_MODEL_FILE)

    candidate_configs = [
        {
            "name": "rf_candidate_1",
            "n_estimators": 180,
            "max_depth": 16,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "class_weight": "balanced_subsample",
            "max_samples": 0.8,
            "n_jobs": 1,
            "random_state": 42,
        },
        {
            "name": "rf_candidate_2",
            "n_estimators": 220,
            "max_depth": 18,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "class_weight": "balanced_subsample",
            "max_samples": 0.8,
            "n_jobs": 1,
            "random_state": 42,
        },
        {
            "name": "rf_candidate_3",
            "n_estimators": 260,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 3,
            "max_features": "log2",
            "bootstrap": True,
            "oob_score": True,
            "class_weight": "balanced_subsample",
            "max_samples": 0.75,
            "n_jobs": 1,
            "random_state": 42,
        },
        {
            "name": "rf_candidate_4",
            "n_estimators": 300,
            "max_depth": 22,
            "min_samples_split": 12,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "class_weight": "balanced_subsample",
            "max_samples": 0.7,
            "n_jobs": 1,
            "random_state": 42,
        },
    ]

    search_rows = []
    best_model = None
    best_result = None
    best_config = None
    best_score = None

    for i, cfg in enumerate(candidate_configs, start=1):
        print(f"\nTraining candidate {i}/{len(candidate_configs)}: {cfg['name']}")

        cfg_for_model = cfg.copy()
        model_name = cfg_for_model.pop("name")

        model = RandomForestClassifier(**cfg_for_model)
        result = train_and_evaluate(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            benign_index,
            n_classes,
        )

        row = {"name": model_name, **cfg_for_model}
        row.update({
            "fit_time_seconds": result["fit_time_seconds"],
            "val_accuracy": result["validation"]["accuracy"],
            "val_macro_f1": result["validation"]["macro_f1"],
            "val_weighted_f1": result["validation"]["weighted_f1"],
            "val_benign_fpr": result["validation"]["benign_fpr"],
            "test_accuracy": result["test"]["accuracy"],
            "test_macro_f1": result["test"]["macro_f1"],
            "test_weighted_f1": result["test"]["weighted_f1"],
            "test_benign_fpr": result["test"]["benign_fpr"],
            "oob_score": result.get("oob_score", None),
        })
        search_rows.append(row)

        current_score = score_for_selection(result)
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_model = model
            best_result = result
            best_config = {"name": model_name, **cfg_for_model}

        gc.collect()

    search_df = pd.DataFrame(search_rows).sort_values(
        by=["val_macro_f1", "val_weighted_f1", "val_accuracy"],
        ascending=[False, False, False]
    )
    search_df.to_csv(SEARCH_RESULTS_FILE, index=False)

    joblib.dump(best_model, BEST_MODEL_FILE)
    save_feature_importance(best_model, feature_cols)

    report = {
        "model": "RandomForest",
        "selection_rule": "Best validation macro_f1, then weighted_f1, then lower benign_fpr, then accuracy",
        "classes": class_names,
        "baseline_config": {
            "n_estimators": 120,
            "max_depth": 14,
            "class_weight": "balanced",
            "n_jobs": 1,
            "random_state": 42,
        },
        "baseline": baseline_result,
        "best_improved_config": best_config,
        "best_improved": best_result,
        "validation_delta_new_minus_old": metric_deltas(
            baseline_result["validation"], best_result["validation"]
        ),
        "test_delta_new_minus_old": metric_deltas(
            baseline_result["test"], best_result["test"]
        ),
        "saved_artifacts": {
            "baseline_model": BASELINE_MODEL_FILE,
            "best_model": BEST_MODEL_FILE,
            "search_results_csv": SEARCH_RESULTS_FILE,
            "feature_importance_csv": FEATURE_IMPORTANCE_FILE,
            "comparison_report": REPORT_FILE,
        },
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nBest candidate selected:")
    print(json.dumps(best_config, indent=2))

    print("\nValidation metrics: baseline vs best")
    for key in baseline_result["validation"].keys():
        old_v = baseline_result["validation"][key]
        new_v = best_result["validation"][key]
        delta = new_v - old_v
        print(f"{key}: {old_v:.4f} -> {new_v:.4f}  delta={delta:+.4f}")

    print("\nTest metrics: baseline vs best")
    for key in baseline_result["test"].keys():
        old_v = baseline_result["test"][key]
        new_v = best_result["test"][key]
        delta = new_v - old_v
        print(f"{key}: {old_v:.4f} -> {new_v:.4f}  delta={delta:+.4f}")

    print("\nTraining times")
    print(f"Baseline RF: {baseline_result['fit_time_seconds']:.2f}s")
    print(f"Best improved RF: {best_result['fit_time_seconds']:.2f}s")

    if "oob_score" in best_result:
        print(f"Best improved RF OOB score: {best_result['oob_score']:.4f}")

    print("\nSaved:")
    print(BASELINE_MODEL_FILE)
    print(BEST_MODEL_FILE)
    print(SEARCH_RESULTS_FILE)
    print(FEATURE_IMPORTANCE_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()