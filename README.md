# Streaming IoT Intrusion Detection with Drift Monitoring

Project for multiclass IoT intrusion detection using CICIoT2023-style traffic, with deployment simulation and monitoring.

## Project Summary

This project builds an end-to-end cybersecurity ML pipeline:

1. Ingest and validate IoT network-flow data
2. Split into train/validation/test
3. Apply reproducible preprocessing
4. Train 4 models (baseline + stronger models)
5. Evaluate with security-focused metrics
6. Serve predictions with FastAPI
7. Monitor alert rate and drift in the API pipeline

## Models Trained

- Logistic Regression (baseline)
- Random Forest
- HistGradientBoosting 
- MLP (PyTorch)

## Improvements We Applied

We improved 3 models (Random Forest, HistGradientBoosting, and MLP) through hyperparameter tuning and training controls.

Improvement details and metrics are documented in:
- [hgb_evaluation.md](./hgb_evaluation.md)
- [rf_evaluation.md](./rf_evaluation.md)
- [mlp_evaluation.md](./mlp_evaluation.md)

## Project Structure

```text
CyberSec_ML/
├── api/
│   └── main.py                 # FastAPI inference and monitoring endpoints
├── dashboard/
│   └── metrics_summary.json    # metrics used by the Streamlit dashboard
├── data/
│   ├── raw/                    # original dataset files, not required for inference
│   └── processed/              # processed train/val/test artifacts
├── models/                     # trained models and preprocessing artifacts
├── pages/                      # Streamlit multipage demo screens
├── src/                        # ingestion, preprocessing, training, evaluation scripts
├── streamlit_app.py            # Streamlit app entry point
├── nav_sidebar.py              # shared Streamlit navigation/sidebar
├── requirements.txt            # Python dependencies
├── Dockerfile                  # container setup for API + Streamlit
├── hgb_evaluation.md           # HistGradientBoosting evaluation report
├── rf_evaluation.md            # Random Forest evaluation report
└── mlp_evaluation.md           # MLP evaluation report
```

## Setup

```bash
pip install -r requirements.txt
```

## Run with Docker

Before building the image, make sure the required model artifacts are present in `models/`.
See [Download Models Into `models/`](#download-models-into-models) for the required filenames.

Build the Docker image:

```bash
docker build -t cybersec-ml .
```

Run the container:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 cybersec-ml
```

Open:
- Streamlit demo: http://127.0.0.1:8501
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Monitor: http://127.0.0.1:8000/monitor

## Pipeline Commands

```bash
python src/ingest.py
python src/split_data
python src/preprocess.py
python src/train.py --model all
python src/test.py --model hgb
```

## Run API

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Meta: http://127.0.0.1:8000/meta
- Monitor: http://127.0.0.1:8000/monitor

## Run Streamlit Demo

```bash
streamlit run streamlit_app.py
```

## API Endpoints

- `POST /predict` - single sample prediction
- `POST /predict/batch` - micro-batch prediction
- `GET /monitor` - alert-rate + drift monitoring snapshot

## Model Links (Local + External)


| Model / Artifact | Local Path | External Link |
|---|---|---|
| Logistic Regression | `models/logistic_regression.joblib` | `https://drive.google.com/drive/folders/1XHY-V8g2U_mCQkcsRcStcXcmBB3yHcG7?usp=sharing` |
| Random Forest (best) | `models/random_forest_best.joblib` | `https://drive.google.com/drive/folders/1wSTcJT2jGSXnkij6KJQ_nPl0Fq_hQ2Q1?usp=sharing` |
| HistGradientBoosting (improved) | `models/gradient_boosting.joblib` | `https://drive.google.com/drive/folders/1J5m6Qt34djhpMzlxIf77ncpfm4Bex8_P?usp=sharing` |
| MLP | `models/mlp_exp4_meta.joblib  models/mlp_exp4.pt` | `https://drive.google.com/drive/folders/13bAiXPEJCoO04auFLnp2jAbh4qqqGEcJ?usp=drive_link` |

## Download Models Into `models/` 

This repository can be shared with an empty `models/` folder.  
Before running inference, download the artifacts from the links above and place them in `models/` with the exact filenames.

Required for API startup:
- `models/gradient_boosting.joblib` (default model)
- `models/scaler.joblib`
- `models/label_encoder.joblib`
- `models/feature_cols.joblib`

Required only when selecting those models:
- `models/logistic_regression.joblib`
- `models/random_forest_best.joblib`
- `models/mlp_exp4.pt`
- `models/mlp_exp4_meta.joblib`

If a requested model file is missing, the API returns an error for that model request.

## Evaluation Metrics Used

- Macro-F1
- Weighted-F1
- Per-class recall
- Confusion matrix
- Benign False Positive Rate (Benign FPR)

These metrics were chosen because accuracy alone is not reliable for imbalanced intrusion-detection data.
