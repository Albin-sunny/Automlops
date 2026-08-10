import streamlit as st
import pandas as pd

from utils.api import (
    get_datasets,
    predict
)

st.set_page_config(
    page_title="Prediction",
    page_icon="🔮"
)

st.title("🔮 Prediction")

response = get_datasets()

if response.status_code != 200:
    st.error("Unable to load datasets.")
    st.stop()

datasets = response.json()

if len(datasets) == 0:
    st.info("No trained datasets available.")
    st.stop()

dataset_names = [
    f"{d['filename']} ({d['_id']})"
    for d in datasets
]

selected = st.selectbox(
    "Choose Dataset",
    dataset_names
)

dataset = datasets[
    dataset_names.index(selected)
]

uploaded_file = st.file_uploader(
    "Upload CSV for Prediction",
    type=["csv"]
)

if uploaded_file is not None:

    preview = pd.read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(preview.head())

    uploaded_file.seek(0)

    if st.button("Predict"):

        with st.spinner("Generating predictions..."):

            response = predict(
                dataset["_id"],
                uploaded_file
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed.")

            st.json(result)

        else:

            st.error(response.text)