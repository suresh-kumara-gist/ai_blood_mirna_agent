import os
import sys
import traceback

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from data.synthetic_generator import generate_synthetic_mirna_dataset
from model.train import train
from agent.diagnostic_agent import AIBloodMiRNADiagnosticAgent


st.set_page_config(
    page_title="AI Blood microRNA Multi-Cancer Detection Agent",
    layout="wide",
)

st.title("🧬 AI Blood microRNA Multi-Cancer Detection Agent")
st.warning("Research prototype only. Not for diagnosis or clinical decision-making.")

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "mirna_transformer.pt")
META_PATH = os.path.join(ARTIFACT_DIR, "preprocess_meta.joblib")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def model_ready():
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)


def show_error(title):
    st.error(title)
    st.code(traceback.format_exc())


with st.sidebar:
    st.header("Setup")

    if st.button("Generate synthetic data"):
        try:
            X, y = generate_synthetic_mirna_dataset(n_samples=800, n_features=200)
            df = X.copy()
            df["label"] = y
            out_path = os.path.join(DATA_DIR, "synthetic_mirna_dataset.csv")
            df.to_csv(out_path, index=False)
            st.success("Synthetic dataset created.")
            st.code(out_path)
        except Exception:
            show_error("Synthetic data generation failed.")

    if st.button("Train demo model"):
        try:
            with st.spinner("Training demo model..."):
                train(epochs=8, out_dir=ARTIFACT_DIR, synthetic=True, augment=True)
            st.success("Model trained successfully.")
        except Exception:
            show_error("Training failed.")

    st.divider()
    st.caption("Artifacts")
    st.write("Model:", "✅" if os.path.exists(MODEL_PATH) else "❌")
    st.write("Metadata:", "✅" if os.path.exists(META_PATH) else "❌")
    st.code(ARTIFACT_DIR)


uploaded = st.file_uploader("Upload patient/sample microRNA CSV", type=["csv"])

if uploaded is None:
    st.info("Train a model from the sidebar, then upload a patient/sample CSV.")
    st.stop()

if not model_ready():
    st.error("No trained model found. Click 'Train demo model' in the sidebar first, or run docker compose run --rm train-demo.")
    st.stop()

try:
    uploaded.seek(0)
    preview = pd.read_csv(uploaded)
    st.subheader("Uploaded sample preview")
    st.dataframe(preview.head(), use_container_width=True)

    uploaded.seek(0)
    agent = AIBloodMiRNADiagnosticAgent(artifact_dir=ARTIFACT_DIR)
    result = agent.run(uploaded, sample_index=0, explain=True)
    raw = result["raw"]
except Exception:
    show_error("Prediction failed.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cancer probability score")

    class_names = list(raw.get("class_names", []))
    probabilities = np.asarray(raw.get("probabilities", []), dtype=float)

    if probabilities.ndim == 2 and len(class_names) == probabilities.shape[1]:
        probs = pd.DataFrame({"class": class_names, "probability": probabilities[0]})
        st.bar_chart(probs.set_index("class"))
        st.dataframe(probs, use_container_width=True)
    else:
        st.json({k: str(v) for k, v in raw.items() if k != "X_preprocessed"})

    predicted = raw.get("predicted_labels", ["unknown"])[0]
    uncertainty = float(raw.get("uncertainty", [0.0])[0])
    anomaly = float(raw.get("anomaly_score", [0.0])[0])

    st.metric("Predicted class", predicted)
    st.metric("Uncertainty", f"{uncertainty:.4f}")
    st.metric("Anomaly score", f"{anomaly:.4f}")

with col2:
    st.subheader("Explainability")

    if result.get("figure") is not None:
        st.pyplot(result["figure"])
    else:
        st.info("Explanation plot unavailable. Prediction still completed.")

    if result.get("top_features") is not None:
        st.dataframe(result["top_features"], use_container_width=True)
    else:
        st.info("Top feature table unavailable.")

st.subheader("AI-generated clinical-style report")
st.text_area("Report", result.get("report", "Report unavailable."), height=320)
