import streamlit as st
from utils.api import get_datasets, select_target

st.set_page_config(page_title="Select Target", page_icon="🎯")

st.title("🎯 Select Target Column")

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

dataset = datasets[dataset_names.index(selected)]

target = st.selectbox(
    "Select Target Column",
    dataset["column_names"]
)

if st.button("Save Target Column"):

    response = select_target(
        dataset["_id"],
        target
    )

    if response.status_code == 200:
        st.success("Target column saved successfully.")
        st.json(response.json())
    else:
        st.error(response.text)