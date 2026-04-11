import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from evaluation import security_metrics_dict
from train_mlp import load_mlp_wrapper

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

X_TEST_FILE = os.path.join(PROCESSED_DIR, "X_test.joblib")
y_TEST_FILE = os.path.join(PROCESSED_DIR, "y_test.joblib")
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.joblib")

MODEL_FILES = {
    "lr": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "rf": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "hgb": os.path.join(MODELS_DIR, "gradient_boosting.joblib"),
}


def load_sklearn_model(key: str):
    path = MODEL_FILES[key]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. Train with: python src/train.py --model {key}"
        )
    return joblib.load(path)


def load_model(key: str):
    if key == "mlp":
        if not os.path.isfile(os.path.join(MODELS_DIR, "mlp.pt")):
            raise FileNotFoundError(
                "MLP weights not found. Train with: python src/train.py --model mlp"
            )
        return load_mlp_wrapper()
    return load_sklearn_model(key)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved model on test set.")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "hgb", "mlp"],
        default="lr",
        help="Which trained model to evaluate",
    )
    args = parser.parse_args()

    print("Loading saved model and test data...")
    model = load_model(args.model)
    le = joblib.load(LABEL_ENCODER_FILE)
    X_test = joblib.load(X_TEST_FILE)
    y_test = joblib.load(y_TEST_FILE)

    print("Model:", args.model)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nRunning predictions...")
    y_pred = model.predict(X_test)

    print("\n--- Test set metrics (PDF section 5) ---")
    sm = security_metrics_dict(y_test, y_pred, le)
    print(f"macro-F1: {sm['macro_f1']:.4f}")
    print(f"weighted-F1: {sm['weighted_f1']:.4f}")
    print(f"benign-FPR: {sm['benign_fpr']:.4f}")

    print("\nOverall metrics:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Macro F1:", round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4))
    print(
        "Weighted F1:",
        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, target_names=list(le.classes_), digits=4, zero_division=0
        )
    )

    decoded_true = le.inverse_transform(y_test[:20])
    decoded_pred = le.inverse_transform(y_pred[:20])

    print("\nFirst 20 predictions:")
    for i, (true_label, pred_label) in enumerate(zip(decoded_true, decoded_pred), 1):
        print(f"Sample {i}: true = {true_label} | predicted = {pred_label}")

    sample_index = 0
    sample_proba = model.predict_proba(X_test[sample_index].reshape(1, -1))[0]
    sample_pred = model.predict(X_test[sample_index].reshape(1, -1))[0]

    print("\nSingle sample example:")
    print("True label:", le.inverse_transform([y_test[sample_index]])[0])
    print("Predicted label:", le.inverse_transform([sample_pred])[0])
    print("Confidence:", round(float(np.max(sample_proba)), 4))

    top5 = sorted(zip(le.classes_, sample_proba), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 probabilities for sample 0:")
    for cls, prob in top5:
        print(f"{cls}: {prob:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix shape:", cm.shape)


if __name__ == "__main__":
    main()
