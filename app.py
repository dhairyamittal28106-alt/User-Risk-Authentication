import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Auth Risk Engine", layout="wide")

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("src/authentication_model.pkl")
        encoders = joblib.load("src/label_encoders.pkl")
        target_le = joblib.load("src/target_encoder.pkl")
        return model, encoders, target_le
    except Exception:
        return None, None, None

model, encoders, target_le = load_ml_assets()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Module:", ["🏠 Home & About", "🛡️ Live Threat Prediction", "📊 Analytics Dashboard"])

if page == "🏠 Home & About":
    st.title("🔐 User Authentication Risk Assessment Platform")
    st.write("This platform uses Machine Learning to analyze login patterns and detect threats.")

elif page == "🛡️ Live Threat Prediction":
    st.title("🛡️ Live Threat Prediction")
    st.markdown("Enter the connection details below to run a real-time risk assessment.")
    
    with st.form("prediction_form"):
        st.subheader("Target Profile")
        
        # Use columns inside the form for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", "admin_user")
            ip_address = st.text_input("IP Address", "192.168.1.1")
            location = st.selectbox("Location", ["Local", "Out of State", "International", "Known Bad Subnet"])
            mfa_enabled = st.checkbox("MFA Enabled", value=True)
            
        with col2:
            device_type = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet", "Unknown/Headless"])
            browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "Tor/Custom"])
            failed_attempts = st.slider("Recent Failed Attempts", 0, 10, 0)
            
        submitted = st.form_submit_button("Run Risk Analysis", use_container_width=True)

        if submitted:
            if model is None:
                st.error("ML model assets not found. Please train the model first.")
            else:
                # Add a simulated loading state for better UX
                with st.spinner("Analyzing threat vectors against ML model..."):
                    input_data = pd.DataFrame([{
                        'username': username, 'ip_address': ip_address, 'location': location,
                        'failed_attempts': int(failed_attempts), 'device_type': device_type,
                        'browser': browser, 'mfa_enabled': int(mfa_enabled)
                    }])

                    for col in ["username", "ip_address", "location", "device_type", "browser"]:
                        le = encoders[col]
                        input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

                    raw_pred = model.predict(input_data)[0]
                    # Safely grab the numeric prediction
                    numeric_risk = int(target_le.inverse_transform([raw_pred])[0])
                
                # --- HEURISTIC RULES ENGINE (OVERRIDES ML) ---
                # Force high risk for obvious threat vectors, regardless of what the ML thinks
                override_reason = ""
                if int(failed_attempts) >= 5:
                    numeric_risk = max(numeric_risk, 9)
                    override_reason = " (Brute Force Override)"
                elif browser == "Tor/Custom":
                    numeric_risk = max(numeric_risk, 8)
                    override_reason = " (Anonymizer Network Override)"
                # ---------------------------------------------
                
                st.toast('Analysis complete!', icon='✅')
                st.markdown("---")
                
                # Map the numeric score (0-10) to a clear text label
                if numeric_risk >= 8:
                    st.error(f"🚨 **THREAT DETECTED: HIGH RISK** (Score: {numeric_risk}/10){override_reason}")
                    st.info("Recommendation: Lock account and require password reset.")
                elif numeric_risk >= 4:
                    st.warning(f"⚠️ **SUSPICIOUS LOGIN: MEDIUM RISK** (Score: {numeric_risk}/10){override_reason}")
                    st.info("Recommendation: Prompt for additional MFA verification.")
                else:
                    st.success(f"✅ **SAFE LOGIN: LOW RISK** (Score: {numeric_risk}/10)")

elif page == "📊 Analytics Dashboard":
    st.title("📊 Admin Analytics Dashboard")
    st.write("Overview of system threats and model explainability.")
    
    try:
        df = pd.read_csv("src/auth_authz_failures_dataset_cleaned.csv")
        total_logins_count = len(df)
        
        if pd.api.types.is_numeric_dtype(df['threat_level']):
            threats_count = len(df[df['threat_level'] <= 5])
        else:
            threats_count = len(df[df['threat_level'].astype(str).str.lower().isin(['high', 'critical'])])
            
        total_logins_display = f"{total_logins_count:,}"
        threats_display = f"{threats_count:,}"
    except Exception as e:
        total_logins_display = "Error"
        threats_display = "Error"
        df = None
    
    # Hero metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Logins Monitored", total_logins_display)
    col2.metric("Threats Blocked", threats_display, delta="-2% this week", delta_color="inverse")
    col3.metric("Model Status", "Optimized")
    
    st.markdown("---")
    
    # Use Tabs to organize the layout cleanly
    tab1, tab2 = st.tabs(["📉 Threat Breakdown", "🧠 ML Explainability (SHAP)"])
    
    with tab1:
        st.subheader("Threats by Device Type")
        if df is not None:
            # Create a simple interactive bar chart using native Streamlit
            device_counts = df['device_type'].value_counts()
            st.bar_chart(device_counts)
        else:
            st.warning("Dataset not available to render charts.")
            
    with tab2:
        st.subheader("Feature Importance")
        st.write("This chart shows which features the AI relies on most to determine the threat level.")
        try:
            st.image("src/shap_summary.png", use_column_width=True)
        except FileNotFoundError:
            st.warning("SHAP plot not found. Please run the training script to generate it.")