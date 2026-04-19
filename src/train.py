import argparse
import os
import time

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from evaluation import print_comparison_table, print_metrics_block, security_metrics_dict
from train_mlp import artifact_paths, train_and_save_mlp

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

X_TRAIN_FILE = os.path.join(PROCESSED_DIR, "X_train.joblib")
y_TRAIN_FILE = os.path.join(PROCESSED_DIR, "y_train.joblib")
X_VAL_FILE = os.path.join(PROCESSED_DIR, "X_val.joblib")
y_VAL_FILE = os.path.join(PROCESSED_DIR, "y_val.joblib")

LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.joblib")

MODEL_LR = os.path.join(MODELS_DIR, "logistic_regression.joblib")
MODEL_RF = os.path.join(MODELS_DIR, "random_forest.joblib")
MODEL_HGB = os.path.join(MODELS_DIR, "gradient_boosting.joblib")


def load_data():
    X_train = joblib.load(X_TRAIN_FILE)
    y_train = joblib.load(y_TRAIN_FILE)
    X_val = joblib.load(X_VAL_FILE)
    y_val = joblib.load(y_VAL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    return X_train, y_train, X_val, y_val, le


def train_logistic_regression(X_train, y_train, X_val, y_val, le, *, verbose=0):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
        verbose=verbose,
    )
    t0 = time.perf_counter()
    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)
    print(f"  LR fit time: {time.perf_counter() - t0:.1f}s")
    y_pred = model.predict(X_val)
    print_metrics_block("LogisticRegression", y_val, y_pred, le)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_LR)
    print(f"Saved: {MODEL_LR}")
    return model, security_metrics_dict(y_val, y_pred, le)


def train_random_forest(X_train, y_train, X_val, y_val, le, *, verbose=0):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=verbose,
    )
    t0 = time.perf_counter()
    print("\nTraining Random Forest (Option A)...")
    model.fit(X_train, y_train)
    print(f"  RF fit time: {time.perf_counter() - t0:.1f}s")
    y_pred = model.predict(X_val)
    print_metrics_block("RandomForest", y_val, y_pred, le)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_RF)
    print(f"Saved: {MODEL_RF}")
    return model, security_metrics_dict(y_val, y_pred, le)


def train_gradient_boosting(X_train, y_train, X_val, y_val, le, *, verbose=0):
    model = HistGradientBoostingClassifier(
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
        verbose=verbose,
    )
    t0 = time.perf_counter()
    print("\nTraining HistGradientBoosting (Option B)...")
    model.fit(X_train, y_train)
    print(f"  HGB fit time: {time.perf_counter() - t0:.1f}s")
    y_pred = model.predict(X_val)
    print_metrics_block("HistGradientBoosting", y_val, y_pred, le)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_HGB)
    print(f"Saved: {MODEL_HGB}")
    return model, security_metrics_dict(y_val, y_pred, le)


def train_mlp_pipeline(
    X_train,
    y_train,
    X_val,
    y_val,
    le,
    *,
    mlp_artifact: str = "mlp",
    mlp_class_weight: str = "balanced",
    mlp_selection: str = "val_loss",
    mlp_lr: float | None = None,
    mlp_min_benign_recall: float | None = None,
    mlp_max_benign_fpr: float | None = None,
):
    print("\nTraining MLP (PyTorch)...")
    print(
        f"  artifact={mlp_artifact!r}  class_weight={mlp_class_weight!r}  "
        f"selection={mlp_selection!r}"
    )
    lr = 1e-3 if mlp_lr is None else mlp_lr
    wrapper = train_and_save_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        list(le.classes_),
        lr=lr,
        artifact_basename=mlp_artifact,
        class_weight_mode=mlp_class_weight,
        selection_metric=mlp_selection,
        min_benign_recall=mlp_min_benign_recall,
        max_benign_fpr=mlp_max_benign_fpr,
    )
    y_pred = wrapper.predict(X_val)
    print_metrics_block("MLP", y_val, y_pred, le)
    wpath, mpath = artifact_paths(mlp_artifact)
    print(f"Saved: {wpath} + {mpath}")
    return wrapper, security_metrics_dict(y_val, y_pred, le)


def run_all(X_train, y_train, X_val, y_val, le, *, verbose=0, mlp_kw=None):
    rows = []
    _, m_lr = train_logistic_regression(
        X_train, y_train, X_val, y_val, le, verbose=verbose
    )
    rows.append(("lr", m_lr))
    _, m_rf = train_random_forest(
        X_train, y_train, X_val, y_val, le, verbose=verbose
    )
    rows.append(("rf", m_rf))
    _, m_hgb = train_gradient_boosting(
        X_train, y_train, X_val, y_val, le, verbose=verbose
    )
    rows.append(("hgb", m_hgb))
    mlp_kw = mlp_kw or {}
    _, m_mlp = train_mlp_pipeline(X_train, y_train, X_val, y_val, le, **mlp_kw)
    rows.append(("mlp", m_mlp))
    print_comparison_table(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Train intrusion classifiers (LR, RF, HGB, MLP)."
    )
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "hgb", "mlp", "all", "compare"],
        default="lr",
        help="compare is an alias for all",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 = sklearn prints progress during LR/RF/HGB fit (more console output)",
    )
    parser.add_argument(
        "--mlp-artifact",
        default="mlp",
        metavar="NAME",
        help=(
            "Basename for saved files: models/<NAME>.pt and models/<NAME>_meta.joblib. "
            "Use e.g. mlp_exp1_sqrt to avoid overwriting mlp_baseline.pt or mlp.pt."
        ),
    )
    parser.add_argument(
        "--mlp-class-weight",
        choices=["balanced", "balanced_sqrt", "none"],
        default="balanced",
        help=(
            "balanced = sklearn weights; balanced_sqrt = sqrt (softer, Experiment 1); "
            "none = uniform."
        ),
    )
    parser.add_argument(
        "--mlp-selection",
        choices=["val_loss", "macro_f1"],
        default="val_loss",
        help="Checkpoint selection: minimize val loss or maximize val macro-F1 (with optional benign gates).",
    )
    parser.add_argument(
        "--mlp-lr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Adam learning rate (default 1e-3). Try 5e-4 as a later experiment.",
    )
    parser.add_argument(
        "--mlp-min-benign-recall",
        type=float,
        default=None,
        metavar="FLOAT",
        help="With --mlp-selection macro_f1: only epochs with benign recall >= this can be best.",
    )
    parser.add_argument(
        "--mlp-max-benign-fpr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="With --mlp-selection macro_f1: only epochs with benign FPR <= this can be best.",
    )
    args = parser.parse_args()
    mode = "all" if args.model == "compare" else args.model
    sk_verbose = args.verbose

    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val, le = load_data()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    mlp_kw = dict(
        mlp_artifact=args.mlp_artifact,
        mlp_class_weight=args.mlp_class_weight,
        mlp_selection=args.mlp_selection,
        mlp_lr=args.mlp_lr,
        mlp_min_benign_recall=args.mlp_min_benign_recall,
        mlp_max_benign_fpr=args.mlp_max_benign_fpr,
    )

    if mode == "all":
        run_all(X_train, y_train, X_val, y_val, le, verbose=sk_verbose, mlp_kw=mlp_kw)
        return

    if mode == "lr":
        train_logistic_regression(
            X_train, y_train, X_val, y_val, le, verbose=sk_verbose
        )
    elif mode == "rf":
        train_random_forest(X_train, y_train, X_val, y_val, le, verbose=sk_verbose)
    elif mode == "hgb":
        train_gradient_boosting(
            X_train, y_train, X_val, y_val, le, verbose=sk_verbose
        )
    elif mode == "mlp":
        train_mlp_pipeline(X_train, y_train, X_val, y_val, le, **mlp_kw)


if __name__ == "__main__":
    main()
