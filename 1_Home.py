import streamlit as st
import os
import time
import pandas as pd
import joblib
from utils import inject_custom_css

st.set_page_config(
    page_title="Home Hub", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

start_time = time.perf_counter()
model_path = os.path.join("src", "authentication_model.pkl")

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        model_status = "Online"
    except Exception:
        model_status = "Error"
else:
    model_status = "Offline"

end_time = time.perf_counter()
latency_ms = max(1, int((end_time - start_time) * 1000))

data_path = os.path.join("src", "cleaned_dataset.csv")
if os.path.exists(data_path):
    try:
        total_rows = len(pd.read_csv(data_path, usecols=[0]))
        logs_display = f"{total_rows:,}"
    except Exception:
        logs_display = "Error"
else:
    logs_display = "File Missing"

active_rules = 4

st.toast('System Online. AI Models Synchronized.', icon='⚡')

st.markdown('<h1 class="gradient-text">Risk Shield </h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Next-gen machine learning for real-time identity threat detection.</div>', unsafe_allow_html=True)

st.divider()

st.subheader("Live System Telemetry")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(label="Historical Logs Analyzed", value=logs_display, delta="src/cleaned_dataset.csv")
with c2:
    st.metric(label="Active Security Rules", value=active_rules, delta="Hardcoded Overrides", delta_color="off")
with c3:
    st.metric(label="Model Load Latency", value=f"{latency_ms} ms", delta="Real-time disk I/O", delta_color="inverse")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("ℹ️ System Capabilities")
    st.write(
        """
        - Real-time risk prediction
        - Model input aligned with trained authentication features
        - Rule-based overrides for obvious high-risk behavior
        - Admin analytics dashboard
        - SHAP explainability image support
        """
    )

with col2:
    st.subheader("🧠 Model Features")
    st.write("The underlying machine learning model is actively trained on the following telemetry vectors:")
    FEATURES = [
        "failed_attempts",
        "device_type",
        "browser",
        "location",
        "mfa_enabled",
        "login_method",
        "role",
    ]
    st.code(", ".join(FEATURES))

st.divider()

st.info("**Launch Sequence:** Select **Prediction** from the sidebar to analyze a login vector.")