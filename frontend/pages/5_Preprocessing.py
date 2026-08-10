import streamlit as st
from utils.api import get_datasets, preprocess_dataset

st.set_page_config(
    page_title="Preprocessing",
    page_icon="⚙️"
)

st.title("⚙️ Dataset Preprocessing")

response = get_datasets()

if response.status_code != 200:
    st.error("Failed to load datasets.")
    st.stop()

datasets = response.json()

if len(datasets) == 0:
    st.info("No datasets available.")
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

if st.button("Run Preprocessing"):

    response = preprocess_dataset(
        dataset["_id"]
    )

    if response.status_code == 200:

        result = response.json()

        st.success("Preprocessing completed.")

        st.subheader("Summary")

        st.write(
            f"Original Rows: {result['original_rows']}"
        )

        st.write(
            f"Processed Rows: {result['processed_rows']}"
        )

        st.write(
            f"Duplicates Removed: {result['duplicates_removed']}"
        )

        st.write(
            f"Missing Value Method: {result['missing_value_method']}"
        )

        st.write(
            f"Encoding Method: {result['encoding_method']}"
        )

        st.write(
            f"Scaling Method: {result['scaling_method']}"
        )

        st.write(
            f"Outlier Strategy: {result['outlier_strategy']}"
        )

        st.write(
            f"Outlier Threshold: {result['outlier_threshold']}"
        )

        st.subheader("Outliers Removed")
        st.json(result["outliers_removed"])

        st.subheader("Outliers Capped")
        st.json(result["outliers_capped"])

    else:
        st.error(response.text)