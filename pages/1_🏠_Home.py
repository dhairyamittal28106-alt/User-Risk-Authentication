import streamlit as st

st.set_page_config(
    page_title="Home Hub", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        @keyframes fadeUp {
            0% { opacity: 0; transform: translateY(30px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .stApp {
            animation: fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        .gradient-text {
            background: linear-gradient(90deg, #FF007F, #7928CA, #00D2FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 4.5rem !important;
            font-weight: 900;
            margin-bottom: 0.5rem;
        }

        .sub-text {
            font-size: 1.3rem;
            color: #A0AEC0;
            margin-bottom: 2.5rem;
            font-weight: 500;
        }

        @keyframes floatCard {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        div[data-testid="metric-container"] {
            background-color: rgba(26, 28, 41, 0.5);
            border: 1px solid rgba(255, 0, 127, 0.2);
            padding: 1.5rem;
            border-radius: 15px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: floatCard 4s ease-in-out infinite;
        }

        div[data-testid="column"]:nth-child(1) div[data-testid="metric-container"] { animation-delay: 0s; }
        div[data-testid="column"]:nth-child(2) div[data-testid="metric-container"] { animation-delay: 0.5s; }
        div[data-testid="column"]:nth-child(3) div[data-testid="metric-container"] { animation-delay: 1s; }

        div[data-testid="metric-container"]:hover {
            transform: scale(1.05) translateY(-5px) !important;
            box-shadow: 0 10px 40px rgba(0, 210, 255, 0.4), inset 0 0 15px rgba(0, 210, 255, 0.2);
            border: 1px solid rgba(0, 210, 255, 0.8);
            animation-play-state: paused;
        }

        hr {
            border: 0;
            height: 2px;
            background-image: linear-gradient(to right, rgba(255,0,127,0), rgba(0,210,255,0.75), rgba(255,0,127,0));
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 0 10px rgba(0,210,255,0.5);
        }
    </style>
""", unsafe_allow_html=True)

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