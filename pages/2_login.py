import streamlit as st

from src.auth_db import authenticate_user, register_user
from utils import inject_custom_css


st.set_page_config(
    page_title="Secure Login",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

if st.session_state.get("logged_in", False):
    st.switch_page("pages/3_Prediction.py")

st.title("Risk Shield")
st.subheader("Secure User Login")
st.markdown(
    """
Welcome to the **Risk Shield - AI Powered User Authentication Risk Assessment Platform**.
"""
)

if st.button("Continue without login", use_container_width=True):
    st.switch_page("pages/3_Prediction.py")

login_tab, register_tab = st.tabs(["Login", "Sign Up"])

with login_tab:
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        remember = st.checkbox("Remember Me")
        login = st.form_submit_button("Login", use_container_width=True)

    if login:
        try:
            is_valid, result = authenticate_user(username, password)
        except Exception as exc:
            st.error(f"Could not connect to MongoDB: {exc}")
        else:
            if not is_valid:
                st.error(result)
            else:
                st.session_state["logged_in"] = True
                st.session_state["username"] = result
                st.session_state["remember_me"] = remember
                st.success("Login successful.")
                st.switch_page("pages/3_Prediction.py")

with register_tab:
    with st.form("register_form"):
        new_username = st.text_input("New username", placeholder="Choose a username")
        new_password = st.text_input("New password", type="password", placeholder="Minimum 6 characters")
        confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
        register = st.form_submit_button("Sign Up", use_container_width=True)

    if register:
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                created, message = register_user(new_username, new_password)
            except Exception as exc:
                st.error(f"Could not connect to MongoDB: {exc}")
            else:
                if not created:
                    st.error(message)
                else:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = new_username.strip().lower()
                    st.session_state["remember_me"] = True
                    st.success(message)
                    st.switch_page("pages/3_Prediction.py")
