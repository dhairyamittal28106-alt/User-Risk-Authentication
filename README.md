# User Risk Authentication

Streamlit prototype estimating login risk from failed attempts, device, browser, location, MFA, login method, and role. Combines a trained model with rule overrides and includes SHAP assets.

    python -m venv .venv
    pip install -r requirements.txt
    streamlit run 1_Home.py

Activate the environment before installing. Use Prediction and Dashboard in the sidebar. Model/data are under src/. Validate predictions independently before security use.
