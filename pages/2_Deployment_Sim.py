"""
Simulate deployment: send **your own feature row** (new data) to the API → predicted class.

The API must be running: uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from nav_sidebar import inject_compact_sidebar_css, render_minimal_sidebar_nav

MODELS = ROOT / "models"

DEFAULT_API = "http://127.0.0.1:8000"

MODEL_OPTIONS = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "HistGradientBoosting (default in API)": "hgb",
    "MLP (PyTorch)": "mlp",
    "MLP baseline (PyTorch)": "mlp_baseline",
}


@st.cache_data
def load_feature_cols():
    p = MODELS / "feature_cols.joblib"
    if not p.is_file():
        return None
    return list(joblib.load(p))


def parse_feature_input(text: str) -> list[float]:
    """Accept JSON array or comma/whitespace-separated numbers."""
    text = text.strip()
    if not text:
        raise ValueError("Empty input.")
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of numbers.")
        return [float(x) for x in data]
    parts = re.split(r"[\s,;]+", text.strip().strip(","))
    parts = [p for p in parts if p]
    return [float(p) for p in parts]


st.set_page_config(page_title="Deployment simulation", layout="wide")
inject_compact_sidebar_css()
render_minimal_sidebar_nav()

st.title("Simulate deployment")

if httpx is None:
    st.error("Install **httpx** (`pip install httpx`) to call the API from this page.")
    st.stop()

feat_cols = load_feature_cols()
if feat_cols is None:
    st.error("Missing `models/feature_cols.joblib`. Run `python src/preprocess.py`.")
    st.stop()

n_feat = len(feat_cols)
base = DEFAULT_API.rstrip("/")

model_label = st.selectbox(
    "Model used for predictions",
    options=list(MODEL_OPTIONS.keys()),
    index=2,
    help="Sent as JSON field `model` to `/predict` (lr, rf, hgb, mlp, mlp_baseline).",
)
model_key = MODEL_OPTIONS[model_label]

st.subheader("Predict from your own feature row")
st.markdown(
    f"Paste **{n_feat}** numbers: either a **JSON array** `[0.12, 3.4, ...]` or "
    "**comma-separated** values in the same order as your dataset’s numeric columns "
    "(see names in the expander below)."
)

with st.expander("Feature column order (training)"):
    st.text("\n".join(f"{i}: {c}" for i, c in enumerate(feat_cols)))

input_mode = st.radio("Input format", ("JSON array", "Comma / space separated"), horizontal=True)
default_ph = "[0.0, 0.0, ...]" if input_mode == "JSON array" else ", ".join(["0.0"] * min(5, n_feat)) + ", ..."
features_text = st.text_area(
    "Feature values",
    height=120,
    placeholder=default_ph,
    help=f"Exactly {n_feat} values.",
)

if st.button("Predict class", type="primary"):
    if not features_text.strip():
        st.warning("Enter feature values first.")
    else:
        try:
            values = parse_feature_input(features_text)
        except (ValueError, json.JSONDecodeError) as e:
            st.error(f"Could not parse input: {e}")
            st.stop()
        if len(values) != n_feat:
            st.error(f"Need exactly **{n_feat}** numbers; you gave {len(values)}.")
            st.stop()
        try:
            r = httpx.post(
                f"{base}/predict",
                json={"features": values, "model": model_key},
                timeout=120.0,
            )
            if r.is_success:
                out = r.json()
                st.success(f"**Predicted class:** `{out.get('prediction', '?')}`")
                st.metric("Confidence", f"{out.get('confidence', 0):.4f}")
                with st.expander("Full API response"):
                    st.json(out)
            else:
                st.error(f"HTTP {r.status_code}: {r.text}")
        except httpx.RequestError as e:
            st.error(
                f"Could not reach API at `{base}` (is uvicorn running?). Error: {e}"
            )
