import streamlit as st

# Must be the first streamlit command called on a page
st.set_page_config(
    page_title="Auth Risk Engine",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔐 User Authentication Risk Assessment Platform")
st.markdown("---")

st.info("👋 Welcome! Use the sidebar navigation menu to jump between the system Home, Project About section, Live Threat Predictions, and the Admin Analytics Dashboard.")

st.sidebar.success("Select a module above to begin.")