import streamlit as st

st.set_page_config(
    page_title="AutoMLOps",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoMLOps Platform")

st.markdown("""
Welcome to the AutoMLOps Platform.

Use the sidebar to:

- Upload Dataset
- Select Target
- Profile Dataset
- Preprocess Dataset
- Train Models
- Make Predictions
""")