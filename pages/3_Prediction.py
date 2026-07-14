import os
import joblib
import pandas as pd
import streamlit as st
import time
import urllib.request
from utils import inject_custom_css
inject_custom_css()

FEATURES = [
    "failed_attempts",
    "device_type",
    "browser",
    "location",
    "mfa_enabled",
    "login_method",
    "role",
]

CATEGORICAL_COLUMNS = [
    "device_type",
    "browser",
    "location",
    "login_method",
    "role",
]

@st.cache_resource
def load_ml_assets():
    try:
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.join(root_path, "src")

        model_path = os.path.join(src_path, "authentication_model.pkl")
        encoder_path = os.path.join(src_path, "label_encoders.pkl")
        target_path = os.path.join(src_path, "target_encoder.pkl")

        missing_files = [path for path in [model_path, encoder_path, target_path] if not os.path.exists(path)]

        if missing_files:
            return None, None, None

        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        target_encoder = joblib.load(target_path)

        return model, encoders, target_encoder
    except Exception:
        return None, None, None

def encode_with_label_encoder(value, encoder):
    value = str(value)
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0

def prepare_model_input(raw_input, encoders):
    model_input = pd.DataFrame([{feature: raw_input[feature] for feature in FEATURES}])

    for column in CATEGORICAL_COLUMNS:
        model_input[column] = model_input[column].apply(
            lambda value: encode_with_label_encoder(value, encoders[column])
        )

    model_input["failed_attempts"] = model_input["failed_attempts"].astype(int)
    model_input["mfa_enabled"] = model_input["mfa_enabled"].astype(int)

    return model_input[FEATURES]

def decode_risk_score(raw_prediction, target_encoder):
    try:
        return int(target_encoder.inverse_transform([raw_prediction])[0])
    except Exception:
        return int(raw_prediction)

@st.cache_data(ttl=300)
def fetch_public_ip():
    try:
        with urllib.request.urlopen("https://api.ipify.org", timeout=3) as response:
            return response.read().decode("utf-8").strip()
    except Exception:
        return "192.168.1.1"

model, encoders, target_encoder = load_ml_assets()

st.title("🔮 Live Risk Prediction")
st.markdown("Enter the login details below to run a real-time risk assessment.")

default_ip_address = fetch_public_ip()

with st.form("prediction_form"):
    st.subheader("Login Attempt")

    col1, col2 = st.columns(2)

    with col1:
        username = st.text_input("Username", value=st.session_state.get("username", "admin_user"))
        ip_address = st.text_input("IP Address", value=default_ip_address)
        location = st.selectbox("Location", ["Local", "Out of State", "International", "Known Bad Subnet"])
        mfa_enabled = st.checkbox("MFA Enabled", value=True)

    with col2:
        device_type = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet", "Unknown/Headless"])
        browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "Tor/Custom"])
        login_method = st.selectbox("Login Method", ["Password", "SSO", "OAuth", "Passwordless", "API Token"])
        role = st.selectbox("Role", ["User", "Admin", "Developer", "Service Account", "Guest"])
        failed_attempts = st.slider("Recent Failed Attempts", 0, 10, 0)

    submitted = st.form_submit_button("Run Risk Analysis", use_container_width=True)

if submitted:
    if model is None or encoders is None or target_encoder is None:
        st.error("ML model assets could not be loaded.")
        st.stop()

    raw_input = {
        "failed_attempts": int(failed_attempts),
        "device_type": device_type,
        "browser": browser,
        "location": location,
        "mfa_enabled": int(mfa_enabled),
        "login_method": login_method,
        "role": role,
    }

    with st.status("Initializing Threat Engine...", expanded=True) as status:
        st.write("📡 Extracting login telemetry...")
        time.sleep(0.5)

        st.write("🛡️ Evaluating against baseline heuristics...")
        time.sleep(0.6)

        st.write("🧠 Querying Machine Learning model...")

        try:
            model_input = prepare_model_input(raw_input, encoders)
            raw_prediction = model.predict(model_input)[0]
            numeric_risk = decode_risk_score(raw_prediction, target_encoder)
            time.sleep(0.7)
            status.update(label="Threat Analysis Complete!", state="complete", expanded=False)
        except Exception:
            status.update(label="Analysis Failed", state="error", expanded=True)
            st.error("Prediction failed.")
            st.stop()

    override_reasons = []

    if failed_attempts >= 5:
        numeric_risk = max(numeric_risk, 9)
        override_reasons.append("Brute force override")

    if browser == "Tor/Custom":
        numeric_risk = max(numeric_risk, 8)
        override_reasons.append("Anonymous browser override")

    if role in ["Admin", "Service Account"] and not mfa_enabled:
        numeric_risk = max(numeric_risk, 8)
        override_reasons.append("Privileged account without MFA")

    if login_method == "API Token" and location in ["International", "Known Bad Subnet"]:
        numeric_risk = max(numeric_risk, 8)
        override_reasons.append("Sensitive login method from risky location")

    override_text = ""
    if override_reasons:
        override_text = " (" + "; ".join(override_reasons) + ")"

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("User", username)
    c2.metric("IP Address", ip_address)
    c3.metric("Risk Score", f"{numeric_risk}/10")

    if numeric_risk >= 8:
        st.error(f"🚨 HIGH RISK - Risk Score: {numeric_risk}/10{override_text}")
        st.info("Recommendation: Lock the account and require password reset.")
    elif numeric_risk >= 4:
        st.warning(f"⚠️ MEDIUM RISK - Risk Score: {numeric_risk}/10{override_text}")
        st.info("Recommendation: Require MFA verification.")
    else:
        st.success(f"✅ LOW RISK - Risk Score: {numeric_risk}/10")
        st.info("Recommendation: Allow login and continue monitoring.")

    with st.expander("Model Input Used"):
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
