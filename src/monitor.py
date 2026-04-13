import numpy as np
import joblib
from scipy.stats import ks_2samp
from collections import deque
from datetime import datetime


# ── 1. Load reference data (training set) ──────────────────────────────────

def load_reference(processed_dir="data/processed"):
    X_train = joblib.load(f"{processed_dir}/X_train.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return X_train, feature_cols


# ── 2. Feature Drift Detection (KS Test) ───────────────────────────────────

def check_feature_drift(X_train_ref, X_new_window, feature_cols, threshold=0.05):
    """
    Compares distribution of each feature in training data vs a new window.
    Returns list of drifted features with their p-values.
    """
    drifted = []
    for i, col in enumerate(feature_cols):
        stat, p = ks_2samp(X_train_ref[:, i], X_new_window[:, i])
        if p < threshold:
            drifted.append({
                "feature": col,
                "p_value": round(p, 6),
                "ks_stat": round(float(stat), 6)
            })
    return drifted


# ── 3. Alert Rate Monitor ───────────────────────────────────────────────────

class AlertMonitor:
    def __init__(self, window_size=1000, alert_threshold=0.5):
        self.window = deque(maxlen=window_size)
        self.threshold = alert_threshold
        self.log = []

    def record(self, prediction: str):
        is_attack = prediction.lower() != "benign"
        self.window.append(int(is_attack))
        rate = sum(self.window) / len(self.window)
        if rate > self.threshold:
            entry = {
                "time": datetime.now().isoformat(),
                "alert_rate": round(rate, 4),
                "message": f"High alert rate: {rate:.1%}"
            }
            self.log.append(entry)
            print(f"[ALERT] {entry['message']}")
        return rate

    def summary(self):
        print(f"Window size: {len(self.window)}")
        if len(self.window) > 0:
            rate = sum(self.window) / len(self.window)
            print(f"Current alert rate: {rate:.1%}")
        print(f"Total alerts logged: {len(self.log)}")


# ── 4. Label Drift Detection ────────────────────────────────────────────────

def check_label_drift(y_train_ref, y_new_window, label_encoder):
    """
    Compares predicted class distribution between training and new window.
    Flags classes whose frequency shifted significantly.
    """
    classes = label_encoder.classes_
    drifted = []
    total_ref = len(y_train_ref)
    total_new = len(y_new_window)

    for i, cls in enumerate(classes):
        ref_rate = np.sum(y_train_ref == i) / total_ref
        new_rate = np.sum(y_new_window == i) / total_new
        delta = abs(new_rate - ref_rate)
        if delta > 0.05:  # flag if frequency shifted by more than 5%
            drifted.append({
                "class": cls,
                "train_rate": round(ref_rate, 4),
                "new_rate": round(new_rate, 4),
                "delta": round(delta, 4)
            })
    return drifted


# ── 5. Run Full Drift Report ────────────────────────────────────────────────

def run_drift_report(X_new_window, y_new_preds, threshold=0.05):
    """
    Full drift check — run this periodically on incoming inference batches.
    """
    X_train_ref, feature_cols = load_reference()
    le = joblib.load("models/label_encoder.joblib")
    y_train_ref = joblib.load("data/processed/y_train.joblib")

    print("\n====== DRIFT REPORT ======")
    print(f"New window size: {len(X_new_window)} samples")

    # Feature drift
    feature_drifts = check_feature_drift(X_train_ref, X_new_window, feature_cols, threshold)
    print(f"\n[Feature Drift] {len(feature_drifts)} / {len(feature_cols)} features drifted:")
    for d in feature_drifts:
        print(f"  {d['feature']:30s}  p={d['p_value']}  ks={d['ks_stat']}")

    # Label drift
    label_drifts = check_label_drift(y_train_ref, y_new_preds, le)
    print(f"\n[Label Drift] {len(label_drifts)} classes shifted significantly:")
    for d in label_drifts:
        print(f"  {d['class']:30s}  train={d['train_rate']:.1%}  new={d['new_rate']:.1%}  delta={d['delta']:.1%}")

    if not feature_drifts and not label_drifts:
        print("\nNo significant drift detected.")

    print("==========================\n")
    return {"feature_drift": feature_drifts, "label_drift": label_drifts}


# ── 6. Quick Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a new incoming batch using test data
    X_test = joblib.load("data/processed/X_test.joblib")
    y_test = joblib.load("data/processed/y_test.joblib")

    # Use first 500 rows as a simulated live window
    X_window = X_test[:500]
    y_window = y_test[:500]

    report = run_drift_report(X_window, y_window)

    # Test alert monitor
    print("--- Alert Monitor Test ---")
    monitor = AlertMonitor(window_size=100, alert_threshold=0.3)
    le = joblib.load("models/label_encoder.joblib")
    for pred_idx in y_window[:100]:
        label = le.inverse_transform([pred_idx])[0]
        monitor.record(label)
    monitor.summary()
