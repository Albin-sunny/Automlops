import streamlit as st
from utils.api import get_datasets, delete_dataset

st.set_page_config(page_title="Dataset Manager", page_icon="📂")

st.title("📂 Dataset Manager")

response = get_datasets()

if response.status_code != 200:
    st.error("Failed to fetch datasets.")
    st.stop()

datasets = response.json()

if len(datasets) == 0:
    st.info("No datasets uploaded yet.")
    st.stop()

for dataset in datasets:

    with st.container():

        st.subheader(dataset["filename"])

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Dataset ID:** {dataset['_id']}")
            st.write(f"**Rows:** {dataset['rows']}")
            st.write(f"**Columns:** {dataset['columns']}")
            st.write(f"**File Size:** {dataset['file_size']} bytes")

        with col2:
            st.write("**Column Names**")
            st.write(", ".join(dataset["column_names"]))

        if st.button(
            "🗑 Delete Dataset",
            key=dataset["_id"]
        ):
            delete_response = delete_dataset(dataset["_id"])

            if delete_response.status_code == 200:
                st.success("Dataset deleted successfully.")
                st.rerun()
            else:
                st.error(delete_response.text)

        st.divider()