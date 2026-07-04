import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

np.NaN = np.nan

st.set_page_config(
    page_title="Auth Risk Engine",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("src/authentication_model.pkl")
        encoders = joblib.load("src/label_encoders.pkl")
        return model, encoders
    except Exception:  
        return None, None

@st.cache_data
def load_dataset():
    file_name = "auth_authz_failures_dataset_cleaned.csv"
    possible_paths = [file_name, f"src/{file_name}", f"data/{file_name}", f"../{file_name}"]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

model, encoders = load_ml_assets()
dataset = load_dataset()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Module:", ["🏠 Home & About", "🛡️ Live Threat Prediction", "📊 Analytics Dashboard"])

st.sidebar.markdown("---")
st.sidebar.info("Team: Ayush, Grusha, Dhairya")

if page == "🏠 Home & About":
    st.title("🔐 User Authentication Risk Assessment Platform")
    st.markdown("---")
    
    st.header("About the Project")
    st.write("Welcome to the Auth Risk Engine. This platform uses Machine Learning to analyze login patterns, device footprints, and behavioral anomalies to detect and block malicious authentication attempts in real-time.")
    
    st.subheader("System Architecture")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Data Pipeline**\n\nCleans and processes raw authentication logs, handling missing values and scaling features.")
    with col2:
        st.success("**ML Engine**\n\nEvaluates login features against a trained ML model to predict threat levels.")
    with col3:
        st.warning("**Dashboard**\n\nProvides admins with real-time analytics, threat visualization, and manual prediction testing.")

elif page == "🛡️ Live Threat Prediction":
    st.title("Live Threat Prediction Engine")
    st.markdown("Enter login event details below to evaluate the risk score.")
    
    with st.form("prediction_form"):
        st.subheader("User & Network Details")
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", "admin_user")
            ip_address = st.text_input("IP Address", "192.168.1.1")
            location = st.selectbox("Location", ["Local", "Out of State", "International", "Known Bad Subnet"])
            failed_attempts = st.slider("Recent Failed Attempts", 0, 10, 0)
            
        with col2:
            device_type = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet", "Unknown/Headless"])
            browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "Tor/Custom"])
            mfa_enabled = st.checkbox("MFA Enabled on Account", value=True)
            
        submitted = st.form_submit_button("Run Risk Analysis")
        
        if submitted:
            if model is None:
                st.error("Cannot run prediction: ML model is missing or corrupted.")
            elif dataset is None:
                st.error("Cannot run prediction: Dataset is missing.")
            else:
                user_row = dataset[dataset['username'] == username]
                
                if user_row.empty:
                    st.warning("User not found in database. Using baseline analysis.")
                    input_data = {'username': username, 'ip_address': ip_address, 'location': location, 
                                 'failed_attempts': int(failed_attempts), 'device_type': device_type, 
                                 'browser': browser, 'mfa_enabled': int(mfa_enabled)}
                else:
                    input_data = user_row.iloc[0].to_dict()
                    input_data.update({'ip_address': ip_address, 'location': location, 
                                      'failed_attempts': int(failed_attempts), 'device_type': device_type, 
                                      'browser': browser, 'mfa_enabled': int(mfa_enabled)})
                
                input_df = pd.DataFrame([input_data])
                
                expected_cols = [
                    'user_id', 'username', 'ip_address', 'device_type', 'os_type', 'browser', 
                    'location', 'login_method', 'success', 'failure_reason', 'auth_type', 
                    'account_status', 'failed_attempts', 'mfa_enabled', 'token_expired', 
                    'session_duration', 'password_age_days', 'role', 'privilege_level', 
                    'suspicious_activity', 'threat_level', 'error_code', 'system_component'
                ]
                
                for col in expected_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0 if col in ['failed_attempts', 'mfa_enabled', 'success', 'token_expired', 'session_duration', 'password_age_days', 'suspicious_activity'] else "Unknown"
                
                input_df = input_df[expected_cols]
                
                try:
                    for col in input_df.select_dtypes(include=['object']).columns:
                        input_df[col] = input_df[col].astype("category")
                    
                    prediction = model.predict(input_df)[0]
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    if prediction in ["High", 1, True]:
                        st.error(f"🚨 **THREAT DETECTED!** Model Prediction: {prediction}")
                    else:
                        st.success(f"✅ **SAFE LOGIN.** Model Prediction: {prediction}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

elif page == "📊 Analytics Dashboard":
    st.title("Admin Analytics Dashboard")
    st.markdown("Overview of recent authentication events and system threats.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logins (24h)", "12,450", "+15%")
    col2.metric("Failed Attempts", "842", "-5%")
    col3.metric("High Threats Blocked", "115", "+22%")
    col4.metric("Active Sessions", "3,401", "Steady")
    
    st.markdown("---")
    st.subheader("Threats Detected Over the Last 7 Days")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=7)
    dummy_threat_data = pd.DataFrame({
        "Low Risk": np.random.randint(50, 150, size=7),
        "Medium Risk": np.random.randint(20, 80, size=7),
        "High Risk": np.random.randint(5, 30, size=7)
    }, index=dates)
    
    st.area_chart(dummy_threat_data)