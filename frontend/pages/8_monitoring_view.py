# import streamlit as st
# import pandas as pd

# from utils.api import (
#     get_latest_monitoring,
#     get_monitoring_history,
# )


# st.set_page_config(
#     page_title="Model Monitoring",
#     page_icon="📊",
#     layout="wide",
# )


# st.title("📊 Model Monitoring")
# st.caption("Dataset schema, missing-data and statistical drift monitoring")


# # ---------------------------------------------------------
# # LOAD LATEST RESULT
# # ---------------------------------------------------------

# try:
#     latest = get_latest_monitoring()

# except Exception as e:
#     st.error(f"Unable to load monitoring data: {e}")
#     st.stop()


# # ---------------------------------------------------------
# # STATUS
# # ---------------------------------------------------------

# status = latest.get(
#     "monitoring_status",
#     "UNKNOWN"
# )

# summary = latest.get(
#     "summary",
#     {}
# )


# if status == "HEALTHY":
#     st.success("🟢 Monitoring Status: HEALTHY")
# elif status == "WARNING":
#     st.warning("🟠 Monitoring Status: WARNING")
# else:
#     st.error(f"🔴 Monitoring Status: {status}")


# st.caption(
#     f"Last checked: {latest.get('timestamp', 'Unknown')}"
# )


# # ---------------------------------------------------------
# # SUMMARY CARDS
# # ---------------------------------------------------------

# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.metric(
#         "Schema",
#         summary.get("schema", "N/A")
#     )

# with col2:
#     st.metric(
#         "Missing Data",
#         summary.get("missing_data", "N/A")
#     )

# with col3:
#     st.metric(
#         "Data Drift",
#         summary.get("data_drift", "N/A")
#     )

# with col4:
#     st.metric(
#         "Drifted Features",
#         summary.get("drifted_features", 0)
#     )


# st.divider()


# # ---------------------------------------------------------
# # DATASET INFORMATION
# # ---------------------------------------------------------

# reference = latest.get(
#     "reference",
#     {}
# )

# current = latest.get(
#     "current",
#     {}
# )


# st.subheader("Dataset Comparison")

# col1, col2 = st.columns(2)


# with col1:

#     st.markdown("### Reference Dataset")

#     st.metric(
#         "Rows",
#         reference.get("rows", 0)
#     )

#     st.metric(
#         "Columns",
#         reference.get("columns", 0)
#     )


# with col2:

#     st.markdown("### Current Dataset")

#     st.metric(
#         "Rows",
#         current.get("rows", 0)
#     )

#     st.metric(
#         "Columns",
#         current.get("columns", 0)
#     )


# # ---------------------------------------------------------
# # DRIFT DETAILS
# # ---------------------------------------------------------

# st.divider()

# st.subheader("🔍 Data Drift Analysis")


# drift = latest.get(
#     "drift",
#     {}
# )

# drift_scores = drift.get(
#     "drift_scores",
#     {}
# )


# if drift_scores:

#     drift_rows = []

#     for feature, values in drift_scores.items():

#         drift_rows.append({
#             "Feature": feature,
#             "KS Statistic": values.get("ks_statistic"),
#             "P-Value": values.get("p_value"),
#             "Drift Detected": (
#                 "⚠️ Yes"
#                 if values.get("drift_detected")
#                 else "✅ No"
#             )
#         })

#     drift_df = pd.DataFrame(drift_rows)

#     st.dataframe(
#         drift_df,
#         use_container_width=True,
#         hide_index=True
#     )

# else:

#     st.info(
#         "No numeric features available for drift analysis."
#     )


# # ---------------------------------------------------------
# # SCHEMA ANALYSIS
# # ---------------------------------------------------------

# st.subheader("Schema Analysis")

# schema = drift.get(
#     "schema",
#     {}
# )


# missing_columns = schema.get(
#     "missing_columns",
#     []
# )

# extra_columns = schema.get(
#     "extra_columns",
#     []
# )

# dtype_changes = schema.get(
#     "dtype_changes",
#     {}
# )


# col1, col2, col3 = st.columns(3)


# with col1:

#     st.metric(
#         "Missing Columns",
#         len(missing_columns)
#     )

#     if missing_columns:
#         st.write(missing_columns)


# with col2:

#     st.metric(
#         "Extra Columns",
#         len(extra_columns)
#     )

#     if extra_columns:
#         st.write(extra_columns)


# with col3:

#     st.metric(
#         "Data Type Changes",
#         len(dtype_changes)
#     )

#     if dtype_changes:
#         st.json(dtype_changes)


# # ---------------------------------------------------------
# # MISSING VALUES
# # ---------------------------------------------------------

# st.divider()

# st.subheader("Missing Value Analysis")


# missing_values = drift.get(
#     "missing_values",
#     {}
# )


# if missing_values:

#     missing_rows = []

#     for feature, values in missing_values.items():

#         missing_rows.append({
#             "Feature": feature,
#             "Reference Count": values.get(
#                 "reference_count"
#             ),
#             "Current Count": values.get(
#                 "current_count"
#             ),
#             "Reference Rate": values.get(
#                 "reference_rate"
#             ),
#             "Current Rate": values.get(
#                 "current_rate"
#             ),
#             "Difference": values.get(
#                 "difference"
#             ),
#         })

#     missing_df = pd.DataFrame(
#         missing_rows
#     )

#     st.dataframe(
#         missing_df,
#         use_container_width=True,
#         hide_index=True
#     )

# else:

#     st.info("No missing-value information available.")


# # ---------------------------------------------------------
# # MONITORING HISTORY
# # ---------------------------------------------------------

# st.divider()

# st.subheader("📜 Monitoring History")


# try:

#     history_response = get_monitoring_history(
#         limit=20
#     )

#     history = history_response.get(
#         "results",
#         []
#     )

# except Exception as e:

#     st.error(
#         f"Unable to load monitoring history: {e}"
#     )

#     history = []


# if history:

#     history_rows = []

#     for result in history:

#         result_summary = result.get(
#             "summary",
#             {}
#         )

#         history_rows.append({
#             "Timestamp": result.get(
#                 "timestamp"
#             ),
#             "Status": result.get(
#                 "monitoring_status"
#             ),
#             "Schema": result_summary.get(
#                 "schema"
#             ),
#             "Missing Data": result_summary.get(
#                 "missing_data"
#             ),
#             "Data Drift": result_summary.get(
#                 "data_drift"
#             ),
#             "Drifted Features": result_summary.get(
#                 "drifted_features",
#                 0
#             ),
#         })

#     history_df = pd.DataFrame(
#         history_rows
#     )

#     st.dataframe(
#         history_df,
#         use_container_width=True,
#         hide_index=True
#     )

# else:

#     st.info(
#         "No monitoring history found."
#     )


# # ---------------------------------------------------------
# # REFRESH
# # ---------------------------------------------------------

# st.divider()

# if st.button(
#     "🔄 Refresh Monitoring"
# ):

#     st.rerun()



import os
import requests
import pandas as pd
import streamlit as st

from utils.api import (
    get_latest_monitoring,
    get_monitoring_history,
)

# Backend URL config
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Monitoring")
st.caption("Dataset schema, missing-data and statistical drift monitoring")


# ---------------------------------------------------------
# RUN NEW MONITORING CHECK
# ---------------------------------------------------------

st.subheader("🚀 Run New Monitoring Check")

col_ref, col_curr = st.columns(2)

with col_ref:
    reference_file = st.file_uploader(
        "Upload Reference Dataset",
        type=["csv"],
        key="monitoring_reference"
    )

with col_curr:
    current_file = st.file_uploader(
        "Upload Current Dataset",
        type=["csv"],
        key="monitoring_current"
    )

if st.button("🚀 Run Monitoring"):
    if reference_file is None or current_file is None:
        st.warning("Please upload both reference and current CSV files.")
    else:
        try:
            with st.spinner("Running monitoring analysis..."):
                response = requests.post(
                    f"{BACKEND_URL}/monitoring/compare",
                    files={
                        "reference_file": (
                            reference_file.name,
                            reference_file.getvalue(),
                            "text/csv",
                        ),
                        "current_file": (
                            current_file.name,
                            current_file.getvalue(),
                            "text/csv",
                        ),
                    },
                    timeout=120,
                )

            response.raise_for_status()
            result = response.json()

            st.success("Monitoring completed successfully.")
            st.session_state["monitoring_result"] = result
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Monitoring request failed: {e}")

st.divider()


# ---------------------------------------------------------
# LOAD LATEST RESULT
# ---------------------------------------------------------

if "monitoring_result" in st.session_state:
    latest = st.session_state["monitoring_result"]
else:
    try:
        latest = get_latest_monitoring()
    except Exception as e:
        st.error(f"Unable to load monitoring data: {e}")
        st.stop()


# ---------------------------------------------------------
# STATUS & TIMESTAMP
# ---------------------------------------------------------

status = latest.get("monitoring_status", "UNKNOWN")
summary = latest.get("summary", {})

if status == "HEALTHY":
    st.success("🟢 Monitoring Status: HEALTHY")
elif status == "WARNING":
    st.warning("🟠 Monitoring Status: WARNING")
else:
    st.error(f"🔴 Monitoring Status: {status}")

st.caption(f"Last checked: {latest.get('timestamp', 'Unknown')}")


# ---------------------------------------------------------
# SUMMARY CARDS
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Schema", summary.get("schema", "N/A"))

with col2:
    st.metric("Missing Data", summary.get("missing_data", "N/A"))

with col3:
    st.metric("Data Drift", summary.get("data_drift", "N/A"))

with col4:
    st.metric("Drifted Features", summary.get("drifted_features", 0))

st.divider()


# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

reference = latest.get("reference", {})
current = latest.get("current", {})

st.subheader("Dataset Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Reference Dataset")
    st.metric("Rows", reference.get("rows", 0))
    st.metric("Columns", reference.get("columns", 0))

with col2:
    st.markdown("### Current Dataset")
    st.metric("Rows", current.get("rows", 0))
    st.metric("Columns", current.get("columns", 0))


# ---------------------------------------------------------
# DRIFT DETAILS
# ---------------------------------------------------------

st.divider()
st.subheader("🔍 Data Drift Analysis")

drift = latest.get("drift", {})
drift_scores = drift.get("drift_scores", {})

if drift_scores:
    drift_rows = []
    for feature, values in drift_scores.items():
        drift_rows.append({
            "Feature": feature,
            "KS Statistic": values.get("ks_statistic"),
            "P-Value": values.get("p_value"),
            "Drift Detected": (
                "⚠️ Yes" if values.get("drift_detected") else "✅ No"
            ),
        })

    drift_df = pd.DataFrame(drift_rows)
    st.dataframe(drift_df, use_container_width=True, hide_index=True)
else:
    st.info("No numeric features available for drift analysis.")


# ---------------------------------------------------------
# SCHEMA ANALYSIS
# ---------------------------------------------------------

st.subheader("Schema Analysis")

schema = drift.get("schema", {})
missing_columns = schema.get("missing_columns", [])
extra_columns = schema.get("extra_columns", [])
dtype_changes = schema.get("dtype_changes", {})

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Missing Columns", len(missing_columns))
    if missing_columns:
        st.write(missing_columns)

with col2:
    st.metric("Extra Columns", len(extra_columns))
    if extra_columns:
        st.write(extra_columns)

with col3:
    st.metric("Data Type Changes", len(dtype_changes))
    if dtype_changes:
        st.json(dtype_changes)


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

st.divider()
st.subheader("Missing Value Analysis")

missing_values = drift.get("missing_values", {})

if missing_values:
    missing_rows = []
    for feature, values in missing_values.items():
        missing_rows.append({
            "Feature": feature,
            "Reference Count": values.get("reference_count"),
            "Current Count": values.get("current_count"),
            "Reference Rate": values.get("reference_rate"),
            "Current Rate": values.get("current_rate"),
            "Difference": values.get("difference"),
        })

    missing_df = pd.DataFrame(missing_rows)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
else:
    st.info("No missing-value information available.")


# ---------------------------------------------------------
# MONITORING HISTORY
# ---------------------------------------------------------

st.divider()
st.subheader("📜 Monitoring History")

try:
    history_response = get_monitoring_history(limit=20)
    history = (
        history_response.get("results", [])
        if isinstance(history_response, dict)
        else history_response
    )
except Exception as e:
    st.error(f"Unable to load monitoring history: {e}")
    history = []

if history:
    history_rows = []
    for result in history:
        result_summary = result.get("summary", {})
        history_rows.append({
            "Timestamp": result.get("timestamp"),
            "Status": result.get("monitoring_status"),
            "Schema": result_summary.get("schema"),
            "Missing Data": result_summary.get("missing_data"),
            "Data Drift": result_summary.get("data_drift"),
            "Drifted Features": result_summary.get("drifted_features", 0),
        })

    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No monitoring history found.")


# ---------------------------------------------------------
# REFRESH
# ---------------------------------------------------------

st.divider()

if st.button("🔄 Refresh Monitoring"):
    if "monitoring_result" in st.session_state:
        del st.session_state["monitoring_result"]
    st.rerun()