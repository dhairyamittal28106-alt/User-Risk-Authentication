import streamlit as st
from utils import inject_custom_css
inject_custom_css()
st.set_page_config(
    page_title="Home Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.toast('System Online. AI Models Synchronized.', icon='⚡')

st.markdown('<h1 class="gradient-text">Auth Risk Engine</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Next-gen machine learning for real-time identity threat detection.</div>', unsafe_allow_html=True)

st.divider()

st.subheader("Live System Telemetry")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(label="Active ML Models", value="CatBoost", delta="Online & Syncing")
with c2:
    st.metric(label="Threat Overrides", value="Armed", delta="Rules Enforced", delta_color="off")
with c3:
    st.metric(label="API Latency", value="24ms", delta="-2ms", delta_color="inverse")

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

st.info("**Launch Sequence:** Select **Live Risk Prediction** from the sidebar to analyze a login vector.")