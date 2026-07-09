import os
import pandas as pd
import streamlit as st
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

st.title("📊 Security Operations Dashboard")

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(root_path, "src", "cleaned_dataset.csv")
shap_path = os.path.join(root_path, "src", "shap_summary.png")

try:
    df = pd.read_csv(dataset_path)
    total_logins = len(df)

    if "blocked" in df.columns:
        if pd.api.types.is_numeric_dtype(df["blocked"]) or pd.api.types.is_bool_dtype(df["blocked"]):
            threats = len(df[df["blocked"] == 1])
        else:
            threats = len(
                df[
                    df["blocked"]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1", "yes", "blocked"])
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
c3.metric("Dataset Status", "Loaded" if df is not None else "Not Found")

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
        if "blocked" in df.columns:
            st.bar_chart(df["blocked"].value_counts())
        else:
            st.warning("Column 'blocked' was not found in the dataset.")
    else:
        st.warning(f"Dataset not found at: {dataset_path}")

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
    if os.path.exists(shap_path):
        st.image(shap_path, use_container_width=True)
    else:
        st.warning("SHAP summary image not found.")