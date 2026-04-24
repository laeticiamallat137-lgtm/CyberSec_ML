"""
Interactive evaluation: pick a model + sample size from held-out test data.
Run: streamlit run streamlit_app.py  → open "Interactive Test" in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nav_sidebar import inject_compact_sidebar_css, render_minimal_sidebar_nav  # noqa: E402

from evaluation import security_metrics_dict  # noqa: E402
from train_mlp import MLPClassifierWrapper, load_mlp_wrapper  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODEL_FILES = {
    "Logistic Regression": "lr",
    "Random Forest (Best)": "rf",
    "HistGradientBoosting": "hgb",
    "MLP (Exp4)": "mlp",
}
SKLEARN_PATHS = {
    "lr": MODELS / "logistic_regression.joblib",
    "rf": MODELS / "random_forest_best.joblib",
    "hgb": MODELS / "gradient_boosting.joblib",
}


@st.cache_data
def load_test_data():
    x_path = PROCESSED / "X_test.joblib"
    y_path = PROCESSED / "y_test.joblib"
    le_path = MODELS / "label_encoder.joblib"
    if not x_path.is_file() or not y_path.is_file():
        return None, None, None
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    le = joblib.load(le_path)
    return X, y, le


@st.cache_resource
def load_sklearn_model(key: str):
    path = SKLEARN_PATHS[key]
    if not path.is_file():
        return None
    return joblib.load(path)


@st.cache_resource
def load_mlp_model(which: str):
    if which == "mlp":
        pt, meta = MODELS / "mlp_exp4.pt", MODELS / "mlp_exp4_meta.joblib"
    else:
        return None
    if not pt.is_file() or not meta.is_file():
        return None
    return load_mlp_wrapper(weights_path=str(pt), meta_path=str(meta))


def get_model(key: str):
    if key == "mlp":
        return load_mlp_model(key)
    return load_sklearn_model(key)


def run_predict(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, MLPClassifierWrapper):
        out = []
        bs = 32768
        for i in range(0, len(X), bs):
            out.append(model.predict(X[i : i + bs]))
        return np.concatenate(out)
    return model.predict(X)


st.set_page_config(page_title="Interactive test", layout="wide")
inject_compact_sidebar_css()
render_minimal_sidebar_nav()

st.title("Interactive model test")

X_test, y_test, le = load_test_data()
if X_test is None:
    st.error(
        f"Could not load `{PROCESSED / 'X_test.joblib'}`. "
        "Run preprocessing first: `python src/preprocess.py`."
    )
    st.stop()

n_total = len(y_test)
st.caption(f"Held-out test set size: **{n_total:,}** samples.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    label_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    key = MODEL_FILES[label_name]
with col_b:
    n_samples = st.number_input(
        "Number of samples",
        min_value=1,
        max_value=n_total,
        value=min(5000, n_total),
        step=100,
        help="Random subset of the test set (reproducible with seed below).",
    )
with col_c:
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

sample_mode = st.radio(
    "Sampling",
    ("Random subset", "First N rows (no shuffle)"),
    horizontal=True,
)

run = st.button("Run evaluation", type="primary")

if not run:
    st.info("Set options above and click **Run evaluation**.")
    st.stop()

model = get_model(key)
if model is None:
    st.error(
        f"Model artifact not found for **{label_name}**. "
        f"Train with: `python src/train.py --model {key}`"
    )
    st.stop()

if sample_mode == "Random subset":
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n_total, size=int(n_samples), replace=False)
else:
    idx = np.arange(int(n_samples))

X_s = X_test[idx]
y_s = y_test[idx]

with st.spinner("Scoring samples…"):
    y_pred = run_predict(model, X_s)

labels_all = np.arange(len(le.classes_))

acc = accuracy_score(y_s, y_pred)
macro = f1_score(y_s, y_pred, average="macro", zero_division=0)
weighted = f1_score(y_s, y_pred, average="weighted", zero_division=0)
sm = security_metrics_dict(y_s, y_pred, le)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{acc:.4f}")
m2.metric("Macro-F1", f"{macro:.4f}")
m3.metric("Weighted-F1", f"{weighted:.4f}")
m4.metric("Benign FPR", f"{sm['benign_fpr']:.4f}")
m5.metric("Samples", f"{len(y_s):,}")

st.subheader("Classification report")
report = classification_report(
    y_s,
    y_pred,
    labels=labels_all,
    target_names=list(le.classes_),
    digits=4,
    zero_division=0,
)
st.code(report, language="text")

true_names = le.inverse_transform(y_s)
pred_names = le.inverse_transform(y_pred)
correct = y_s == y_pred

results_df = pd.DataFrame(
    {
        "run_row": np.arange(1, len(y_s) + 1),
        "test_set_index": idx,
        "true_label": true_names,
        "predicted_label": pred_names,
        "correct": correct,
    }
)

st.subheader("Every sample: true label vs prediction")
st.caption(
    "**correct** = model output matches the true class. "
    "**test_set_index** = row position in the full held-out `X_test` matrix."
)
st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)

n_ok = int(np.sum(correct))
n_wrong = int(len(correct) - n_ok)
sum_col1, sum_col2 = st.columns(2)
with sum_col1:
    st.metric("Correct guesses", n_ok)
with sum_col2:
    st.metric("Incorrect guesses", n_wrong)

st.subheader("Confusion matrix (this subset)")
cm = confusion_matrix(y_s, y_pred, labels=np.arange(len(le.classes_)))
labels = list(le.classes_)
n_cls = len(labels)
fig_w = max(10.0, n_cls * 0.38)
fig_h = max(8.0, n_cls * 0.34)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plot_kw = dict(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    colorbar=True,
    include_values=n_cls <= 20,
)
if n_cls <= 20:
    plot_kw["values_format"] = "d"
disp.plot(**plot_kw)
ax.tick_params(axis="both", labelsize=6 if n_cls > 20 else 8)
plt.title(f"{label_name} — n={len(y_s):,} samples", fontsize=11)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download full run table (CSV)",
    data=csv_bytes,
    file_name=f"interactive_pred_{key}_n{len(y_s)}.csv",
    mime="text/csv",
)
