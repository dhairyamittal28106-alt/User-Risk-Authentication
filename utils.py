import os
import streamlit as st

def inject_custom_css():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    css_file = os.path.join(root_dir, "style.css")

    try:
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom CSS file not found.")