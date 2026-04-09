import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

X_TEST_FILE = os.path.join(PROCESSED_DIR, "X_test.joblib")
y_TEST_FILE = os.path.join(PROCESSED_DIR, "y_test.joblib")

MODEL_FILE = os.path.join(MODELS_DIR, "logistic_regression.joblib")
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.joblib")


def main():
    print("Loading saved model and test data...")

    model = joblib.load(MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    X_test = joblib.load(X_TEST_FILE)
    y_test = joblib.load(y_TEST_FILE)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nRunning predictions...")
    y_pred = model.predict(X_test)

    print("\nOverall metrics:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 4))
    print("Weighted F1:", round(f1_score(y_test, y_pred, average="weighted"), 4))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

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