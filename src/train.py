import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

X_TRAIN_FILE = os.path.join(PROCESSED_DIR, "X_train.joblib")
y_TRAIN_FILE = os.path.join(PROCESSED_DIR, "y_train.joblib")
X_VAL_FILE = os.path.join(PROCESSED_DIR, "X_val.joblib")
y_VAL_FILE = os.path.join(PROCESSED_DIR, "y_val.joblib")

LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.joblib")
MODEL_FILE = os.path.join(MODELS_DIR, "logistic_regression.joblib")


def main():
    print("Loading preprocessed data...")

    X_train = joblib.load(X_TRAIN_FILE)
    y_train = joblib.load(y_TRAIN_FILE)
    X_val = joblib.load(X_VAL_FILE)
    y_val = joblib.load(y_VAL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
        verbose=1
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    print("\nValidation report:")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()