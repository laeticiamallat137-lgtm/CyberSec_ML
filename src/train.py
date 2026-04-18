import argparse
import os
import time

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from evaluation import print_comparison_table, print_metrics_block, security_metrics_dict
from train_mlp import train_and_save_mlp

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


def train_mlp_pipeline(X_train, y_train, X_val, y_val, le):
    print("\nTraining MLP (PyTorch)...")
    wrapper = train_and_save_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        list(le.classes_),
    )
    y_pred = wrapper.predict(X_val)
    print_metrics_block("MLP", y_val, y_pred, le)
    print(f"Saved: models/mlp.pt + models/mlp_meta.joblib")
    return wrapper, security_metrics_dict(y_val, y_pred, le)


def run_all(X_train, y_train, X_val, y_val, le, *, verbose=0):
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
    _, m_mlp = train_mlp_pipeline(X_train, y_train, X_val, y_val, le)
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
    args = parser.parse_args()
    mode = "all" if args.model == "compare" else args.model
    sk_verbose = args.verbose

    print("Loading preprocessed data...")
    X_train, y_train, X_val, y_val, le = load_data()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    if mode == "all":
        run_all(X_train, y_train, X_val, y_val, le, verbose=sk_verbose)
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
        train_mlp_pipeline(X_train, y_train, X_val, y_val, le)


if __name__ == "__main__":
    main()
