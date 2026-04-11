# CyberSec ML — IoT intrusion classification

Pipeline for the CICIoT2023-style tabular dataset: ingest → split → preprocess → **train multiple classifiers** → evaluate on held-out test.

## Dataset and references

- [CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- [Benchmark paper](https://www.mdpi.com/1424-8220/23/13/5941)

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline (from repo root)

1. `python src/ingest.py` — sample/validate raw CSVs → `data/processed/dataset.parquet` (requires `pyarrow`).
2. `python src/split_data` — `dataset.parquet` → `train.parquet`, `val.parquet`, `test.parquet`.
3. `python src/preprocess.py` — scaled arrays + `models/scaler.joblib`, `label_encoder.joblib`, etc.

## Training — four models (compare all)

Trains sklearn **Logistic Regression** (baseline), **Random Forest** (Option A), **HistGradientBoostingClassifier** (Option B), and a small **PyTorch MLP** (dropout + weighted cross-entropy + early stopping).

```bash
python src/train.py --model lr      # baseline only
python src/train.py --model rf      # Random Forest only
python src/train.py --model hgb     # gradient boosting only
python src/train.py --model mlp     # neural net only
python src/train.py --model all     # train all four; prints validation comparison table
python src/train.py --model compare # alias for all
python src/train.py --model all -v 1   # sklearn prints progress during LR/RF/HGB fits
```

**Artifacts:** `models/logistic_regression.joblib`, `random_forest.joblib`, `gradient_boosting.joblib`, `mlp.pt` + `mlp_meta.joblib`.

**Note:** Random Forest on ~1M+ rows can take a long time and significant RAM. HistGradientBoosting is usually faster at this scale.

## Metrics (course / PDF-style)

Reports include **macro-F1**, **weighted-F1**, **per-class recall** (`classification_report`), **confusion matrix shape**, and **benign FPR** (fraction of true BENIGN rows predicted as attack). Compare models on **validation** during training and on the **test** set with `test.py` — not accuracy alone.

## Evaluation on test set

```bash
python src/test.py --model lr
python src/test.py --model rf
python src/test.py --model hgb
python src/test.py --model mlp
```

## MLP details

- Architecture: 3 hidden layers (default 256 → 128 → 64), ReLU, dropout 0.3, linear output to `n_classes`.
- Loss: `CrossEntropyLoss` with **balanced class weights** from `sklearn.utils.class_weight.compute_class_weight`.
- Optimization: Adam; **early stopping** on validation loss (patience 7).
- Full training data by default (batched); CPU or CUDA if available.
