# User Risk Authentication

A Streamlit prototype for estimating authentication risk from login telemetry. It combines a trained model with rule-based overrides and includes a prediction flow, dashboard, and SHAP explainability assets.

The model features include failed attempts, device type, browser, location, MFA status, login method, and user role.

## Run locally

Use Python 3.10 or newer:

    python -m venv .venv
    # Activate the environment, then:
    pip install -r requirements.txt
    streamlit run 1_Home.py

Choose **Prediction** or **Dashboard** in the Streamlit sidebar. The app expects the model and dataset files under `src/`; they are included in the repository.

## Main files

- `1_Home.py`: system status and home page
- `pages/2_login.py`: login flow
- `pages/3_Prediction.py`: prediction interface
- `pages/4_Dashboard.py`: analytics
- `src/`: model, data preparation, and explainability artifacts

Model output is an estimate and should not be used as a complete security decision without independent evaluation.
