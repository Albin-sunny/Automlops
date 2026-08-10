import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.stats import ks_2samp

# Async database handler import
from app.monitoring.monitoring_history import save_monitoring_result

# Configuration
REPORT_FOLDER = Path("app/monitoring/reports")
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)


def _safe_float(val) -> float | None:
    """Converts pandas numeric outputs into clean Python floats or None for valid JSON serialization."""
    return None if pd.isna(val) else float(val)


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """Computes summary statistics for numeric features in a DataFrame."""
    numeric_columns = df.select_dtypes(include=["number"]).columns
    statistics = {}

    for column in numeric_columns:
        col_data = df[column]
        statistics[column] = {
            "mean": _safe_float(col_data.mean()),
            "median": _safe_float(col_data.median()),
            "std": _safe_float(col_data.std()),
            "min": _safe_float(col_data.min()),
            "max": _safe_float(col_data.max()),
        }

    return statistics


def compare_schema(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Identifies missing columns, extra columns, and data type shifts between datasets."""
    ref_cols, curr_cols = set(reference_df.columns), set(current_df.columns)
    common_columns = ref_cols.intersection(curr_cols)

    dtype_changes = {
        col: {
            "reference": str(reference_df[col].dtype),
            "current": str(current_df[col].dtype),
        }
        for col in common_columns
        if str(reference_df[col].dtype) != str(current_df[col].dtype)
    }

    return {
        "missing_columns": sorted(ref_cols - curr_cols),
        "extra_columns": sorted(curr_cols - ref_cols),
        "dtype_changes": dtype_changes,
    }


def compare_missing_values(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Calculates missing-value counts, rates, and deltas across common columns."""
    results = {}
    common_columns = sorted(set(reference_df.columns).intersection(current_df.columns))

    ref_len = len(reference_df) if len(reference_df) > 0 else 1
    curr_len = len(current_df) if len(current_df) > 0 else 1

    for column in common_columns:
        ref_count = int(reference_df[column].isna().sum())
        curr_count = int(current_df[column].isna().sum())

        ref_rate = ref_count / ref_len
        curr_rate = curr_count / curr_len

        results[column] = {
            "reference_count": ref_count,
            "current_count": curr_count,
            "reference_rate": round(ref_rate, 4),
            "current_rate": round(curr_rate, 4),
            "difference": round(curr_rate - ref_rate, 4),
        }

    return results


def compare_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Performs a two-sample Kolmogorov-Smirnov test on numeric features to flag statistical drift."""
    drift_scores = {}
    common_columns = sorted(set(reference_df.columns).intersection(current_df.columns))

    for column in common_columns:
        if not (
            pd.api.types.is_numeric_dtype(reference_df[column])
            and pd.api.types.is_numeric_dtype(current_df[column])
        ):
            continue

        ref_vals = reference_df[column].dropna()
        curr_vals = current_df[column].dropna()

        if len(ref_vals) < 2 or len(curr_vals) < 2:
            continue

        statistic, p_value = ks_2samp(ref_vals, curr_vals)
        drift_detected = bool(p_value < alpha)

        drift_scores[column] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": drift_detected,
        }

    return drift_scores


def _sync_generate_report(reference_path: str, current_path: str, save_file: bool = True) -> dict:
    """Synchronous core runner that processes CSV files and generates report dictionary."""
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    # Clean whitespace from column headers
    reference_data.columns = reference_data.columns.str.strip()
    current_data.columns = current_data.columns.str.strip()

    schema_result = compare_schema(reference_data, current_data)
    missing_result = compare_missing_values(reference_data, current_data)
    drift_result = compare_data_drift(reference_data, current_data)

    schema_ok = (
        len(schema_result["missing_columns"]) == 0
        and len(schema_result["extra_columns"]) == 0
        and len(schema_result["dtype_changes"]) == 0
    )

    missing_drift_detected = any(
        abs(val["difference"]) >= 0.10 for val in missing_result.values()
    )

    data_drift_detected = any(
        val["drift_detected"] for val in drift_result.values()
    )

    overall_drift_detected = (
        not schema_ok or missing_drift_detected or data_drift_detected
    )

    monitoring_status = "WARNING" if overall_drift_detected else "HEALTHY"

    report = {
        "timestamp": datetime.now().isoformat(),
        "monitoring_status": monitoring_status,
        "summary": {
            "schema": "PASS" if schema_ok else "FAIL",
            "missing_data": "WARNING" if missing_drift_detected else "PASS",
            "data_drift": "WARNING" if data_drift_detected else "PASS",
            "drifted_features": sum(
                val["drift_detected"] for val in drift_result.values()
            ),
        },
        "reference": {
            "rows": len(reference_data),
            "columns": len(reference_data.columns),
            "statistics": get_dataset_statistics(reference_data),
        },
        "current": {
            "rows": len(current_data),
            "columns": len(current_data.columns),
            "statistics": get_dataset_statistics(current_data),
        },
        "drift": {
            "schema": schema_result,
            "missing_values": missing_result,
            "drift_scores": drift_result,
            "overall_drift_detected": overall_drift_detected,
        },
    }

    # Optional local file backup
    if save_file:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORT_FOLDER / f"drift_report_{timestamp_str}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

    return report


async def generate_monitoring_report(
    reference_path: str, current_path: str, save_db: bool = True, save_file: bool = True
) -> dict:
    """Non-blocking async entrypoint: calculates drift and persists result to MongoDB."""
    # 1. Run CPU-bound report generation in thread pool
    report = await asyncio.to_thread(_sync_generate_report, reference_path, current_path, save_file)

    # 2. Asynchronously persist report to MongoDB
    if save_db:
        await save_monitoring_result(report)

    return report