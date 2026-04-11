"""
IoT IDS inference API — JSON feature vectors → class + probabilities.

Each ``features`` list must be **raw numeric values** (before scaling), in the same
column order as ``feature_cols`` saved during ``preprocess.py``. The service applies
``StandardScaler`` to match training.

Run from repository root:
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

Default checkpoint: env ``IDS_MODEL=lr|rf|hgb|mlp`` (default: hgb). Each request can
override with JSON field ``\"model\"`` or ``\"model_key\"``.

Requires preprocess artifacts: models/scaler.joblib, label_encoder.joblib, feature_cols.joblib
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from train_mlp import load_mlp_wrapper

MODELS_DIR = ROOT / "models"
MODEL_FILES = {
    "lr": MODELS_DIR / "logistic_regression.joblib",
    "rf": MODELS_DIR / "random_forest.joblib",
    "hgb": MODELS_DIR / "gradient_boosting.joblib",
}

MAX_BATCH = int(os.environ.get("IDS_MAX_BATCH", "2048"))
ALLOWED_MODELS = frozenset(MODEL_FILES.keys()) | {"mlp"}


class AppState:
    scaler = None
    le = None
    feat_cols: list | None = None
    default_model_key: str = "hgb"
    n_features: int = 0


state = AppState()
_model_cache: dict[str, object] = {}


def _load_sklearn(key: str):
    path = MODEL_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def _load_model(key: str):
    """train_mlp uses cwd-relative `models/`; chdir so MLP loads from repo root."""
    if key == "mlp":
        pt = MODELS_DIR / "mlp.pt"
        meta = MODELS_DIR / "mlp_meta.joblib"
        if not pt.is_file() or not meta.is_file():
            raise FileNotFoundError(f"Missing MLP weights or meta under {MODELS_DIR}")
        old = os.getcwd()
        os.chdir(ROOT)
        try:
            return load_mlp_wrapper()
        finally:
            os.chdir(old)
    return _load_sklearn(key)


def resolve_model_key(optional: str | None) -> str:
    key = (optional or state.default_model_key).lower().strip()
    if key not in ALLOWED_MODELS:
        raise HTTPException(
            400,
            detail=f"Invalid model '{key}'. Allowed: {sorted(ALLOWED_MODELS)}",
        )
    return key


def get_model(key: str):
    if key not in _model_cache:
        try:
            _model_cache[key] = _load_model(key)
        except FileNotFoundError as e:
            raise HTTPException(503, detail=str(e)) from e
    return _model_cache[key]


@asynccontextmanager
async def lifespan(_: FastAPI):
    key = os.environ.get("IDS_MODEL", "hgb").lower()
    if key not in ALLOWED_MODELS:
        key = "hgb"
    scaler_path = MODELS_DIR / "scaler.joblib"
    le_path = MODELS_DIR / "label_encoder.joblib"
    cols_path = MODELS_DIR / "feature_cols.joblib"
    for p in (scaler_path, le_path, cols_path):
        if not p.is_file():
            raise RuntimeError(
                f"Missing preprocess artifact: {p}. Run: python src/preprocess.py"
            )
    state.default_model_key = key
    state.scaler = joblib.load(scaler_path)
    state.le = joblib.load(le_path)
    state.feat_cols = list(joblib.load(cols_path))
    state.n_features = len(state.feat_cols)
    get_model(key)
    yield


app = FastAPI(
    title="IoT intrusion detection — inference",
    description="Scaled feature vectors → attack class + softmax probabilities.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    features: list[float] = Field(
        ...,
        description="One sample: raw feature values (not z-scores), length = n_features, order = feature_cols",
    )
    model: str | None = Field(
        default=None,
        description="lr | rf | hgb | mlp — overrides server default if set",
        validation_alias=AliasChoices("model", "model_key"),
    )


class BatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    batch: list[list[float]] = Field(
        ...,
        description="Micro-batch: each inner list is one sample (same length as features)",
    )
    model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model", "model_key"),
    )


def _predict_scaled(X: np.ndarray, model_key: str) -> tuple[np.ndarray, np.ndarray]:
    """X is float array shape (n, n_features); use named columns to match scaler."""
    model = get_model(model_key)
    cols = list(state.feat_cols)
    frame = pd.DataFrame(X, columns=cols)
    Xs = state.scaler.transform(frame)
    proba = model.predict_proba(Xs)
    pred_idx = np.argmax(proba, axis=1)
    return pred_idx, proba


def _one_result(pred_idx: int, proba_row: np.ndarray, *, model_key: str) -> dict:
    classes = list(state.le.classes_)
    label = classes[int(pred_idx)]
    return {
        "prediction": label,
        "class_index": int(pred_idx),
        "confidence": float(np.max(proba_row)),
        "all_probabilities": {classes[i]: float(proba_row[i]) for i in range(len(classes))},
        "model": model_key,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": state.default_model_key,
        "cached_models": sorted(_model_cache.keys()),
        "n_features": state.n_features,
    }


@app.get("/meta")
def meta():
    return {
        "default_model": state.default_model_key,
        "allowed_models": sorted(ALLOWED_MODELS),
        "n_features": state.n_features,
        "max_batch": MAX_BATCH,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != state.n_features:
        raise HTTPException(
            400,
            detail=f"Expected {state.n_features} features, got {len(req.features)}",
        )
    mk = resolve_model_key(req.model)
    X = np.asarray(req.features, dtype=np.float64).reshape(1, -1)
    pred_idx, proba = _predict_scaled(X, mk)
    return _one_result(int(pred_idx[0]), proba[0], model_key=mk)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if not req.batch:
        raise HTTPException(400, detail="batch must be non-empty")
    if len(req.batch) > MAX_BATCH:
        raise HTTPException(
            400,
            detail=f"Batch size {len(req.batch)} exceeds max {MAX_BATCH}",
        )
    mk = resolve_model_key(req.model)
    n = state.n_features
    rows = []
    for i, row in enumerate(req.batch):
        if len(row) != n:
            raise HTTPException(
                400,
                detail=f"Row {i}: expected {n} features, got {len(row)}",
            )
        rows.append(row)
    X = np.asarray(rows, dtype=np.float64)
    pred_idx, proba = _predict_scaled(X, mk)
    results = [
        _one_result(int(pred_idx[i]), proba[i], model_key=mk) for i in range(len(rows))
    ]
    return {"count": len(results), "model": mk, "results": results}
