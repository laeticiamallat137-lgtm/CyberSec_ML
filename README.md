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

After each run, the confusion matrix is saved as **`results/confusion_matrix_<model>_test.png`** (heatmap for reports) and **`results/confusion_matrix_<model>_test.csv`** (exact counts). Rows = true class, columns = predicted. With many classes, the PNG omits cell numbers (heatmap only). Use `--no-print-cm` to skip printing the large text table in the terminal.

## Results dashboard (for presentation)

Interactive view of **metrics** (`dashboard/metrics_summary.json`) and **confusion matrix PNGs** in `results/` (after running `src/test.py`).

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens a browser (default `http://localhost:8501`). For your professor: run locally or deploy (e.g. Streamlit Community Cloud) with this repo.

- **Interactive Test** (sidebar): pick **model** + **number of test samples** (random or first N), run evaluation, see metrics, confusion matrix, and download a CSV of predictions.
- **Deployment** (sidebar): start FastAPI with `uvicorn`, then paste **raw** features and use **Predict class** (model dropdown sends `model` to `/predict`).

## Inference API (deployment simulation)

From the **repository root**, after `preprocess.py` and training:

```bash
# default model: HistGradientBoosting (IDS_MODEL=hgb)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Optional: `IDS_MODEL` sets the **default** checkpoint when a request does **not** include a model. You can switch models **per request** without restarting:

- `POST /predict` / `POST /predict/batch` — add `"model": "lr"` (or `"rf"`, `"hgb"`, `"mlp"`). Same field can be spelled `"model_key"`.

The **Deployment** Streamlit page has a **Model** dropdown that sends this for you.

- `GET /health` — status, **default** model, **cached** model keys, feature count  
- `GET /meta` — default model, **allowed** model ids, `n_features`, max batch size  
- `POST /predict` — body `{"features": [float, ...], "model": "hgb"}` — **raw** numeric features (same order as `feature_cols`), *not* z-scores; the API applies `StandardScaler` like training  
- `POST /predict/batch` — body `{"batch": [[...], ...], "model": "hgb"}` — each row is raw features (micro-batch; max size `IDS_MAX_BATCH`, default 2048)  

Interactive docs: `http://127.0.0.1:8000/docs`

Edit `dashboard/metrics_summary.json` if you retrain and validation numbers change.

The sidebar is intentionally **compact**: **Results**, **Interactive**, and **Deployment** (with icons). Dataset/help text lives in an **About** expander on the Results page. Requires **Streamlit ≥ 1.33** (`client.showSidebarNavigation` + `st.page_link`). If the collapsed sidebar does not hide labels on your version, adjust CSS in `nav_sidebar.py`.

## MLP details

- Architecture: 3 hidden layers (default 256 → 128 → 64), ReLU, dropout 0.3, linear output to `n_classes`.
- Loss: `CrossEntropyLoss` with **balanced class weights** from `sklearn.utils.class_weight.compute_class_weight`.
- Optimization: Adam; **early stopping** on validation loss (patience 7).
- Full training data by default (batched); CPU or CUDA if available.
