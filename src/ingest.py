import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
ATTACK_SAMPLE = 8000
BENIGN_SAMPLE = 50000


def find_csv_files(data_dir: str) -> list[str]:
    csv_files = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, name))
    return sorted(csv_files)

def load_and_sample(data_dir: str = RAW_DIR) -> pd.DataFrame:
    print("Looking in:", os.path.abspath(data_dir))
    print("Files found:", os.listdir(data_dir))
    frames = []
    csv_files = find_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under {os.path.abspath(data_dir)}. "
            "Place raw CSVs there or inside its subfolders."
        )

    print(f"CSV files found recursively: {len(csv_files)}")
    for csv_path in csv_files:
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Detect label column (adjust if yours is named differently)
        label_col = detect_label_column(df)
        
        for label, group in df.groupby(label_col):
            n = BENIGN_SAMPLE if "benign" in str(label).lower() else ATTACK_SAMPLE
            sampled = group.sample(min(len(group), n), random_state=42)
            frames.append(sampled)
        
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal rows after sampling: {combined.shape[0]}")
    return combined


def detect_label_column(df: pd.DataFrame) -> str:
    # CICIoT2023 commonly uses 'label' or 'Label' or 'attack'
    for candidate in ["label", "Label", "attack", "Attack", "class", "Class"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find label column. Columns are: {list(df.columns)}")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Validation Report ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("Missing values:")
        print(missing)
    else:
        print("No missing values found.")

    # Infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    inf_cols = numeric_df.columns[numeric_df.isin([np.inf, -np.inf]).any()]
    if len(inf_cols) > 0:
        print(f"\nColumns with inf values: {list(inf_cols)}")
    else:
        print("No infinite values found.")

    # Class distribution
    label_col = detect_label_column(df)
    print(f"\nClass distribution ({label_col}):")
    print(df[label_col].value_counts())

    # Fix issues
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    if before != after:
        print(f"\nDropped {before - after} rows with NaN/inf values.")

    print("\nValidation complete.")
    return df


def save_processed(df: pd.DataFrame, filename: str = "dataset.parquet"):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_parquet(path, index=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    df = load_and_sample()
    df = validate(df)
    save_processed(df)
    print("\nStep 1 complete! Your data is ready in data/processed/dataset.parquet")
