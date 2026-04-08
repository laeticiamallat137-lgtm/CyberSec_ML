import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.parquet")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.parquet")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.parquet")


def detect_label_column(df):
    for col in ["label", "Label", "attack", "Attack", "class", "Class"]:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")


def detect_timestamp_column(df):
    for col in ["timestamp", "Timestamp", "time", "Time", "date", "Date"]:
        if col in df.columns:
            return col
    return None


def find_non_feature_columns(df, label_col):
    non_feature_cols = [label_col]

    timestamp_col = detect_timestamp_column(df)
    if timestamp_col:
        non_feature_cols.append(timestamp_col)

    id_cols = [
        "src_ip", "dst_ip", "Src IP", "Dst IP",
        "flow_id", "Flow ID",
        "src_port", "dst_port", "Src Port", "Dst Port"
    ]

    for col in id_cols:
        if col in df.columns:
            non_feature_cols.append(col)

    return list(set(non_feature_cols))


def build_pipeline(train_df):
    label_col = detect_label_column(train_df)
    non_feature_cols = find_non_feature_columns(train_df, label_col)

    candidate_cols = [c for c in train_df.columns if c not in non_feature_cols]
    feature_cols = train_df[candidate_cols].select_dtypes(include=["number"]).columns.tolist()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    return X_train_scaled, y_train_encoded, scaler, le, feature_cols, label_col


def apply_pipeline(df, scaler, le, feature_cols, label_col):
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    X_scaled = scaler.transform(X)
    y_encoded = le.transform(y)

    return X_scaled, y_encoded


def save_artifacts(scaler, le, feature_cols):
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))


def save_data(X_train, y_train, X_val, y_val, X_test, y_test):
    joblib.dump(X_train, os.path.join(PROCESSED_DIR, "X_train.joblib"))
    joblib.dump(y_train, os.path.join(PROCESSED_DIR, "y_train.joblib"))

    joblib.dump(X_val, os.path.join(PROCESSED_DIR, "X_val.joblib"))
    joblib.dump(y_val, os.path.join(PROCESSED_DIR, "y_val.joblib"))

    joblib.dump(X_test, os.path.join(PROCESSED_DIR, "X_test.joblib"))
    joblib.dump(y_test, os.path.join(PROCESSED_DIR, "y_test.joblib"))


def main():
    print("Loading data...")

    train_df = pd.read_parquet(TRAIN_FILE)
    val_df = pd.read_parquet(VAL_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    X_train, y_train, scaler, le, feature_cols, label_col = build_pipeline(train_df)

    X_val, y_val = apply_pipeline(val_df, scaler, le, feature_cols, label_col)
    X_test, y_test = apply_pipeline(test_df, scaler, le, feature_cols, label_col)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("Classes:")
    for i, c in enumerate(le.classes_):
        print(i, "->", c)

    save_artifacts(scaler, le, feature_cols)
    save_data(X_train, y_train, X_val, y_val, X_test, y_test)

    print("Done.")


if __name__ == "__main__":
    main()