import streamlit as st
from utils.api import upload_dataset

st.set_page_config(page_title="Upload Dataset", page_icon="📁")

st.title("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("Upload"):

        with st.spinner("Uploading dataset..."):

            response = upload_dataset(uploaded_file)

        if response.status_code == 200:

            st.success("Dataset uploaded successfully!")

            data = response.json()

            st.json(data)

        else:

            st.error("Upload failed")

            st.write(response.text)