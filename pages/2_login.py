import streamlit as st

st.set_page_config(
    page_title="Secure Login",
    page_icon="🔐",
    layout="wide"
)

# Redirect if already logged in
if st.session_state.get("logged_in", False):
    st.switch_page("pages/3_Prediction.py")

st.title("🛡️ Risk Shield")

st.subheader("Secure User Login")

st.markdown("""
Welcome to the **Risk Shield - AI Powered User Authentication Risk Assessment Platform**.

Please enter your credentials to continue.
""")

with st.form("login_form"):

    username = st.text_input(
        "👤 Username",
        placeholder="Enter your username"
    )

    password = st.text_input(
        "🔑 Password",
        type="password",
        placeholder="Enter your password"
    )

    remember = st.checkbox("Remember Me")

    login = st.form_submit_button(
        "🔓 Login",
        use_container_width=True
    )

if login:

    if username.strip() == "" or password.strip() == "":
        st.error("Please enter both username and password.")

    else:
        # Demo Login
        st.session_state["logged_in"] = True
        st.session_state["username"] = username

        st.success("Login Successful!")

        st.info("Redirecting to Risk Assessment...")

        st.switch_page("pages/3_Prediction.py")