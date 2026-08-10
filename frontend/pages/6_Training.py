import streamlit as st
from utils.api import get_datasets, train_model

st.set_page_config(
    page_title="Training",
    page_icon="🧠"
)

st.title("🧠 Model Training")

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

if st.button("Start Training"):

    with st.spinner("Training models..."):

        response = train_model(
            dataset["_id"]
        )

    if response.status_code == 200:

        result = response.json()

        st.success("Training completed successfully.")

        st.subheader("Best Model")

        st.write(
            f"**Task Type:** {result['task_type']}"
        )

        st.write(
            f"**Best Model:** {result['best_model']}"
        )

        st.write(
            f"**Training Time:** {result['training_time_seconds']:.2f} seconds"
        )

        st.subheader("Best Model Metrics")
        st.json(result["best_model_metrics"])

        st.subheader("All Model Scores")
        st.json(result["model_scores"])

    else:
        st.error(response.text)