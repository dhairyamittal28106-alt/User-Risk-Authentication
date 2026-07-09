import os
import traceback

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Auth Risk Engine", layout="wide")


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
        base_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(base_path, "src")

        model_path = os.path.join(src_path, "authentication_model.pkl")
        encoder_path = os.path.join(src_path, "label_encoders.pkl")
        target_path = os.path.join(src_path, "target_encoder.pkl")

        missing_files = [
            path
            for path in [model_path, encoder_path, target_path]
            if not os.path.exists(path)
        ]

        if missing_files:
            st.error("Missing ML asset file(s):")
            for path in missing_files:
                st.code(path)
            return None, None, None

        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        target_encoder = joblib.load(target_path)

        return model, encoders, target_encoder

    except Exception:
        st.error("Failed to load ML assets.")
        st.code(traceback.format_exc())
        return None, None, None


def encode_with_label_encoder(value, encoder):
    value = str(value)
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0


def prepare_model_input(raw_input, encoders):
    model_input = pd.DataFrame([{feature: raw_input[feature] for feature in FEATURES}])

    for column in CATEGORICAL_COLUMNS:
        if column not in encoders:
            raise KeyError(f"Missing label encoder for column: {column}")

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


model, encoders, target_encoder = load_ml_assets()


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Module:",
    [
        "Home & About",
        "Live Threat Prediction",
        "Analytics Dashboard",
    ],
)


if page == "Home & About":
    st.title("User Authentication Risk Assessment Platform")

    st.write(
        """
This platform uses machine learning to analyze login behavior and estimate
whether an authentication attempt is safe, suspicious, or malicious.

### Features
- Real-time risk prediction
- Model input aligned with trained authentication features
- Rule-based overrides for obvious high-risk behavior
- Admin analytics dashboard
- SHAP explainability image support
"""
    )

    st.subheader("Model Features")
    st.code(", ".join(FEATURES))


elif page == "Live Threat Prediction":
    st.title("Live Threat Prediction")
    st.markdown("Enter the login details below to run a real-time risk assessment.")

    with st.form("prediction_form"):
        st.subheader("Login Attempt")

        col1, col2 = st.columns(2)

        with col1:
            username = st.text_input("Username", "admin_user")
            ip_address = st.text_input("IP Address", "192.168.1.1")
            location = st.selectbox(
                "Location",
                [
                    "Local",
                    "Out of State",
                    "International",
                    "Known Bad Subnet",
                ],
            )
            mfa_enabled = st.checkbox("MFA Enabled", value=True)

        with col2:
            device_type = st.selectbox(
                "Device Type",
                [
                    "Desktop",
                    "Mobile",
                    "Tablet",
                    "Unknown/Headless",
                ],
            )
            browser = st.selectbox(
                "Browser",
                [
                    "Chrome",
                    "Safari",
                    "Firefox",
                    "Tor/Custom",
                ],
            )
            login_method = st.selectbox(
                "Login Method",
                [
                    "Password",
                    "SSO",
                    "OAuth",
                    "Passwordless",
                    "API Token",
                ],
            )
            role = st.selectbox(
                "Role",
                [
                    "User",
                    "Admin",
                    "Developer",
                    "Service Account",
                    "Guest",
                ],
            )
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

        with st.spinner("Analyzing threat vectors..."):
            try:
                model_input = prepare_model_input(raw_input, encoders)
                raw_prediction = model.predict(model_input)[0]
                numeric_risk = decode_risk_score(raw_prediction, target_encoder)
            except Exception:
                st.error("Prediction failed.")
                st.code(traceback.format_exc())
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

        st.success("Analysis complete.")
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("User", username)
        c2.metric("IP Address", ip_address)
        c3.metric("Risk Score", f"{numeric_risk}/10")

        if numeric_risk >= 8:
            st.error(f"HIGH RISK - Risk Score: {numeric_risk}/10{override_text}")
            st.info("Recommendation: Lock the account and require password reset.")
        elif numeric_risk >= 4:
            st.warning(f"MEDIUM RISK - Risk Score: {numeric_risk}/10{override_text}")
            st.info("Recommendation: Require MFA verification.")
        else:
            st.success(f"LOW RISK - Risk Score: {numeric_risk}/10")
            st.info("Recommendation: Allow login and continue monitoring.")

        with st.expander("Model Input Used"):
            st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)


elif page == "Analytics Dashboard":
    st.title("Admin Analytics Dashboard")

    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src",
        "auth_authz_failures_dataset_cleaned.csv",
    )

    try:
        df = pd.read_csv(dataset_path)
        total_logins = len(df)

        if "threat_level" in df.columns and pd.api.types.is_numeric_dtype(df["threat_level"]):
            threats = len(df[df["threat_level"] >= 8])
        elif "threat_level" in df.columns:
            threats = len(
                df[
                    df["threat_level"]
                    .astype(str)
                    .str.lower()
                    .isin(["high", "critical"])
                ]
            )
        else:
            threats = 0

    except Exception:
        df = None
        total_logins = 0
        threats = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Logins", f"{total_logins:,}")
    c2.metric("Threats Blocked", f"{threats:,}")
    c3.metric("Model Status", "Loaded" if model else "Not Loaded")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        [
            "Threat Breakdown",
            "Feature Distributions",
            "SHAP Explainability",
        ]
    )

    with tab1:
        if df is not None:
            if "threat_level" in df.columns:
                st.bar_chart(df["threat_level"].value_counts())
            else:
                st.warning("Column 'threat_level' was not found in the dataset.")
        else:
            st.warning("Dataset not found.")

    with tab2:
        if df is not None:
            selected_feature = st.selectbox(
                "Select Feature",
                [feature for feature in FEATURES if feature in df.columns],
            )
            st.bar_chart(df[selected_feature].value_counts())
        else:
            st.warning("Dataset not found.")

    with tab3:
        shap_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "src",
            "shap_summary.png",
        )

        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True)
        else:
            st.warning("SHAP summary image not found.")
