"""
Professional results dashboard for the professor / demo.
Run from project root:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from nav_sidebar import inject_compact_sidebar_css, render_minimal_sidebar_nav

ROOT = Path(__file__).resolve().parent
METRICS_JSON = ROOT / "dashboard" / "metrics_summary.json"
RESULTS_DIR = ROOT / "results"


@st.cache_data
def load_metrics(_cache_key: float | None = None) -> dict:
    if not METRICS_JSON.is_file():
        return {}
    with open(METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(
        page_title="IoT Intrusion Detection — Results",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_compact_sidebar_css()
    render_minimal_sidebar_nav()

    cache_key = METRICS_JSON.stat().st_mtime if METRICS_JSON.is_file() else None
    data = load_metrics(cache_key)
    if not data:
        st.error(
            f"Missing `{METRICS_JSON.relative_to(ROOT)}`. "
            "Restore the dashboard folder or add metrics_summary.json."
        )
        st.stop()

    # --- Header ---
    st.title(data.get("project_title", "Project results"))
    st.caption(data.get("course_line", ""))

    ds = data.get("dataset", {})

    # --- Dataset strip ---
    st.subheader("Experimental setup")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Classes", ds.get("n_classes", "—"))
    with c2:
        st.metric("Features", ds.get("n_features", "—"))
    with c3:
        st.metric("Validation samples", f"{ds.get('n_val', 0):,}" if ds.get("n_val") else "—")
    with c4:
        st.metric("Test samples", f"{ds.get('n_test', 0):,}" if ds.get("n_test") else "—")

    note = data.get("split_note")
    if note:
        st.info(note)

    # --- Model comparison table ---
    st.subheader("Model comparison (validation set)")
    models = data.get("models", [])
    rows = []
    for m in models:
        rows.append(
            {
                "Model": m.get("name", ""),
                "Macro-F1 ↑": m.get("val_macro_f1"),
                "Weighted-F1 ↑": m.get("val_weighted_f1"),
                "Benign FPR ↓": m.get("val_benign_fpr"),
                "Accuracy": m.get("val_accuracy"),
            }
        )
    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.format(
            {
                "Macro-F1 ↑": "{:.4f}",
                "Weighted-F1 ↑": "{:.4f}",
                "Benign FPR ↓": "{:.4f}",
                "Accuracy": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    help_txt = data.get("metric_help", {})
    with st.expander("How to read these metrics (PDF / IDS context)"):
        for k, v in help_txt.items():
            st.markdown(f"**{k.replace('_', ' ')}:** {v}")

    st.markdown(
        "**Takeaway:** Random Forest and HistGradientBoosting outperform the linear baseline "
        "on macro-F1 and especially **benign FPR**; compare tables in your written report."
    )

    # --- Confusion matrices ---
    st.subheader("Confusion matrices (held-out test set)")

    cms = []
    for m in models:
        mid = m.get("id", "")
        cm_id = m.get("cm_id", mid)
        png = RESULTS_DIR / f"confusion_matrix_{cm_id}_test.png"
        cms.append((m.get("name", mid), png, cm_id))

    missing = [name for name, p, _ in cms if not p.is_file()]
    if missing:
        st.warning(
            "Some images are missing. Run evaluation for each model, e.g. "
            "`python src/test.py --model rf --no-print-cm`"
        )

    for row_start in range(0, len(cms), 2):
        row = cms[row_start : row_start + 2]
        cols = st.columns(2)
        for col, (title, png_path, _) in zip(cols, row):
            with col:
                if png_path.is_file():
                    st.image(str(png_path), caption=title, use_container_width=True)
                else:
                    st.info(f"No file: `{png_path.name}`")


if __name__ == "__main__":
    main()
