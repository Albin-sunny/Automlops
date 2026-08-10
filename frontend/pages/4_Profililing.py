import streamlit as st
from utils.api import get_datasets, profile_dataset

st.set_page_config(page_title="Profiling", page_icon="📊")

st.title("📊 Dataset Profiling")

response = get_datasets()

if response.status_code != 200:
    st.error("Failed to load datasets.")
    st.stop()

datasets = response.json()

if not datasets:
    st.info("No datasets found.")
    st.stop()

dataset_names = [
    f"{d['filename']} ({d['_id']})"
    for d in datasets
]

selected = st.selectbox(
    "Choose Dataset",
    dataset_names
)

dataset = datasets[dataset_names.index(selected)]

if st.button("Generate Profile"):

    response = profile_dataset(dataset["_id"])

    if response.status_code == 200:

        profile = response.json()

        st.success("Profiling completed.")

        st.subheader("Dataset Information")
        st.json(profile)

    else:
        st.error(response.text)