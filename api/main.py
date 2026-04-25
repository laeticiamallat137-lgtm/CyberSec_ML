"""
IoT IDS inference API with deployment-style monitoring.

Predict endpoints:
- POST /predict
- POST /predict/batch

Monitoring endpoints:
- GET /health
- GET /meta
- GET /monitor

The API applies the same scaler used in training, returns class probabilities,
tracks alert rate on predictions, and keeps rolling windows for feature/label drift.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime
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

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover
    ks_2samp = None

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_FILES = {
    "lr": MODELS_DIR / "logistic_regression.joblib",
    "rf": MODELS_DIR / "random_forest_best.joblib",
    "hgb": MODELS_DIR / "gradient_boosting.joblib",
}

MAX_BATCH = int(os.environ.get("IDS_MAX_BATCH", "2048"))
ALLOWED_MODELS = frozenset(MODEL_FILES.keys()) | {"mlp"}

ALERT_WINDOW = int(os.environ.get("IDS_ALERT_WINDOW", "1000"))
ALERT_THRESHOLD = float(os.environ.get("IDS_ALERT_THRESHOLD", "0.5"))

DRIFT_WINDOW = int(os.environ.get("IDS_DRIFT_WINDOW", "500"))
DRIFT_MIN_WINDOW = int(os.environ.get("IDS_DRIFT_MIN_WINDOW", "200"))
DRIFT_THRESHOLD = float(os.environ.get("IDS_DRIFT_THRESHOLD", "0.05"))
DRIFT_LABEL_DELTA = float(os.environ.get("IDS_DRIFT_LABEL_DELTA", "0.05"))
DRIFT_TOP_FEATURES = int(os.environ.get("IDS_DRIFT_TOP_FEATURES", "10"))
REF_SAMPLE_SIZE = int(os.environ.get("IDS_DRIFT_REF_SIZE", "5000"))


class AppState:
    scaler = None
    le = None
    feat_cols: list | None = None
    default_model_key: str = "hgb"
    n_features: int = 0
    y_train_ref = None
    X_ref_sample = None


state = AppState()
_model_cache: dict[str, object] = {}
_monitor_lock = threading.Lock()


class AlertRateMonitor:
    def __init__(self, window_size: int, threshold: float):
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.alert_events = deque(maxlen=200)

    def update(self, labels: list[str]) -> float:
        for label in labels:
            self.window.append(0 if str(label).upper() == "BENIGN" else 1)
        if not self.window:
            return 0.0
        rate = float(sum(self.window) / len(self.window))
        if rate > self.threshold:
            self.alert_events.append(
                {
                    "time": datetime.utcnow().isoformat() + "Z",
                    "alert_rate": round(rate, 4),
                    "threshold": self.threshold,
                    "message": f"High alert rate: {rate:.1%}",
                }
            )
        return rate

    def snapshot(self) -> dict:
        current_rate = float(sum(self.window) / len(self.window)) if self.window else 0.0
        return {
            "window_size": self.window.maxlen,
            "observed": len(self.window),
            "threshold": self.threshold,
            "current_rate": round(current_rate, 4),
            "total_alert_events": len(self.alert_events),
            "last_alert": self.alert_events[-1] if self.alert_events else None,
        }


class DriftMonitor:
    def __init__(
        self,
        *,
        feature_cols: list[str],
        classes: list[str],
        X_ref_sample: np.ndarray | None,
        y_train_ref: np.ndarray | None,
        window_size: int,
    ):
        self.feature_cols = feature_cols
        self.classes = classes
        self.X_ref_sample = X_ref_sample
        self.y_train_ref = y_train_ref
        self.window_X = deque(maxlen=window_size)
        self.window_pred = deque(maxlen=window_size)
        self.ref_label_rates = self._build_ref_label_rates(y_train_ref)

    def _build_ref_label_rates(self, y_train_ref: np.ndarray | None) -> dict[str, float]:
        if y_train_ref is None or len(y_train_ref) == 0:
            return {}
        counts = Counter(int(v) for v in y_train_ref.tolist())
        total = len(y_train_ref)
        rates = {}
        for i, cls in enumerate(self.classes):
            rates[cls] = float(counts.get(i, 0) / total)
        return rates

    def update(self, X_scaled: np.ndarray, pred_idx: np.ndarray) -> None:
        for row in X_scaled:
            self.window_X.append(np.asarray(row, dtype=np.float64))
        for idx in pred_idx:
            self.window_pred.append(int(idx))

    def quick_status(self) -> dict:
        return {
            "feature_window_size": len(self.window_X),
            "label_window_size": len(self.window_pred),
            "min_window_for_checks": DRIFT_MIN_WINDOW,
            "feature_ref_loaded": self.X_ref_sample is not None,
            "label_ref_loaded": bool(self.ref_label_rates),
            "feature_check_available": ks_2samp is not None,
        }

    def _feature_drift(self) -> dict:
        if self.X_ref_sample is None:
            return {"enabled": False, "reason": "reference_features_unavailable"}
        if ks_2samp is None:
            return {"enabled": False, "reason": "scipy_not_installed"}
        if len(self.window_X) < DRIFT_MIN_WINDOW:
            return {
                "enabled": True,
                "ready": False,
                "reason": f"need_at_least_{DRIFT_MIN_WINDOW}_samples",
                "window_size": len(self.window_X),
            }

        Xw = np.vstack(self.window_X)
        drifted = []
        for i, col in enumerate(self.feature_cols):
            stat, p = ks_2samp(self.X_ref_sample[:, i], Xw[:, i])
            if p < DRIFT_THRESHOLD:
                drifted.append(
                    {
                        "feature": col,
                        "p_value": round(float(p), 6),
                        "ks_stat": round(float(stat), 6),
                    }
                )
        drifted = sorted(drifted, key=lambda x: x["p_value"])
        return {
            "enabled": True,
            "ready": True,
            "threshold": DRIFT_THRESHOLD,
            "window_size": len(self.window_X),
            "drifted_feature_count": len(drifted),
            "top_drifted_features": drifted[:DRIFT_TOP_FEATURES],
        }

    def _label_drift(self) -> dict:
        if not self.ref_label_rates:
            return {"enabled": False, "reason": "reference_labels_unavailable"}
        if len(self.window_pred) < DRIFT_MIN_WINDOW:
            return {
                "enabled": True,
                "ready": False,
                "reason": f"need_at_least_{DRIFT_MIN_WINDOW}_samples",
                "window_size": len(self.window_pred),
            }
        counts = Counter(self.window_pred)
        total = len(self.window_pred)
        shifted = []
        for i, cls in enumerate(self.classes):
            train_rate = float(self.ref_label_rates.get(cls, 0.0))
            new_rate = float(counts.get(i, 0) / total)
            delta = abs(new_rate - train_rate)
            if delta > DRIFT_LABEL_DELTA:
                shifted.append(
                    {
                        "class": cls,
                        "train_rate": round(train_rate, 4),
                        "new_rate": round(new_rate, 4),
                        "delta": round(delta, 4),
                    }
                )
        shifted = sorted(shifted, key=lambda x: x["delta"], reverse=True)
        return {
            "enabled": True,
            "ready": True,
            "threshold": DRIFT_LABEL_DELTA,
            "window_size": len(self.window_pred),
            "shifted_class_count": len(shifted),
            "shifted_classes": shifted,
        }

    def snapshot(self) -> dict:
        return {
            "feature_drift": self._feature_drift(),
            "label_drift": self._label_drift(),
        }


_alert_monitor: AlertRateMonitor | None = None
_drift_monitor: DriftMonitor | None = None


def _load_sklearn(key: str):
    path = MODEL_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def _load_model(key: str):
    if key == "mlp":
        pt = MODELS_DIR / "mlp_exp4.pt"
        meta = MODELS_DIR / "mlp_exp4_meta.joblib"
        if not pt.is_file() or not meta.is_file():
            raise FileNotFoundError(f"Missing MLP weights or meta under {MODELS_DIR}")
        try:
            return load_mlp_wrapper(weights_path=str(pt), meta_path=str(meta))
        except TypeError:
            # Backward compatibility for wrappers that only support default paths.
            return load_mlp_wrapper()
    return _load_sklearn(key)


def _load_reference_X_sample(n_features: int) -> np.ndarray | None:
    x_train_path = PROCESSED_DIR / "X_train.joblib"
    if not x_train_path.is_file():
        return None
    try:
        X_train = np.asarray(joblib.load(x_train_path), dtype=np.float64)
    except Exception:
        return None
    if X_train.ndim != 2 or X_train.shape[1] != n_features:
        return None
    n = len(X_train)
    if n <= REF_SAMPLE_SIZE:
        return X_train
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=REF_SAMPLE_SIZE, replace=False)
    return X_train[idx]


def _load_reference_labels() -> np.ndarray | None:
    y_train_path = PROCESSED_DIR / "y_train.joblib"
    if not y_train_path.is_file():
        return None
    try:
        return np.asarray(joblib.load(y_train_path))
    except Exception:
        return None


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


def _monitor_update(X_scaled: np.ndarray, pred_idx: np.ndarray) -> dict:
    if _alert_monitor is None or _drift_monitor is None:
        return {"enabled": False}
    labels = [state.le.classes_[int(i)] for i in pred_idx.tolist()]
    with _monitor_lock:
        alert_rate = _alert_monitor.update(labels)
        _drift_monitor.update(X_scaled, pred_idx)
        return {
            "enabled": True,
            "alert_rate": round(alert_rate, 4),
            "drift_status": _drift_monitor.quick_status(),
        }


def _monitor_snapshot() -> dict:
    if _alert_monitor is None or _drift_monitor is None:
        return {"enabled": False}
    with _monitor_lock:
        return {
            "enabled": True,
            "alert": _alert_monitor.snapshot(),
            "drift": _drift_monitor.snapshot(),
        }


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _alert_monitor, _drift_monitor

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

    state.y_train_ref = _load_reference_labels()
    state.X_ref_sample = _load_reference_X_sample(state.n_features)

    _alert_monitor = AlertRateMonitor(window_size=ALERT_WINDOW, threshold=ALERT_THRESHOLD)
    _drift_monitor = DriftMonitor(
        feature_cols=list(state.feat_cols),
        classes=list(state.le.classes_),
        X_ref_sample=state.X_ref_sample,
        y_train_ref=state.y_train_ref,
        window_size=DRIFT_WINDOW,
    )

    get_model(key)
    yield


app = FastAPI(
    title="IoT intrusion detection - inference",
    description="Scaled feature vectors -> attack class + probabilities + monitoring.",
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
        description="lr | rf | hgb | mlp - overrides server default if set",
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


def _predict_scaled(
    X: np.ndarray, model_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = get_model(model_key)
    cols = list(state.feat_cols)
    frame = pd.DataFrame(X, columns=cols)
    Xs = state.scaler.transform(frame)
    proba = model.predict_proba(Xs)
    pred_idx = np.argmax(proba, axis=1)
    return pred_idx, proba, Xs


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
        "monitoring_enabled": _alert_monitor is not None and _drift_monitor is not None,
    }


@app.get("/meta")
def meta():
    return {
        "default_model": state.default_model_key,
        "allowed_models": sorted(ALLOWED_MODELS),
        "n_features": state.n_features,
        "max_batch": MAX_BATCH,
        "monitoring": {
            "alert_window": ALERT_WINDOW,
            "alert_threshold": ALERT_THRESHOLD,
            "drift_window": DRIFT_WINDOW,
            "drift_min_window": DRIFT_MIN_WINDOW,
            "drift_threshold": DRIFT_THRESHOLD,
            "label_drift_threshold": DRIFT_LABEL_DELTA,
            "reference_sample_size": REF_SAMPLE_SIZE,
        },
    }


@app.get("/monitor")
def monitor():
    return _monitor_snapshot()


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != state.n_features:
        raise HTTPException(
            400,
            detail=f"Expected {state.n_features} features, got {len(req.features)}",
        )
    mk = resolve_model_key(req.model)
    X = np.asarray(req.features, dtype=np.float64).reshape(1, -1)
    pred_idx, proba, Xs = _predict_scaled(X, mk)
    out = _one_result(int(pred_idx[0]), proba[0], model_key=mk)
    out["monitor"] = _monitor_update(Xs, pred_idx)
    return out


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
    pred_idx, proba, Xs = _predict_scaled(X, mk)
    results = [
        _one_result(int(pred_idx[i]), proba[i], model_key=mk) for i in range(len(rows))
    ]
    return {
        "count": len(results),
        "model": mk,
        "results": results,
        "monitor": _monitor_update(Xs, pred_idx),
    }
