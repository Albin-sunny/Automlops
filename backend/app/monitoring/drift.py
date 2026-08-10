import pandas as pd
from scipy.stats import ks_2samp

def compare_schema(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    ref_cols, curr_cols = set(reference_df.columns), set(current_df.columns)
    common_columns = ref_cols & curr_cols

    dtype_changes = {
        col: {"reference": str(reference_df[col].dtype), "current": str(current_df[col].dtype)}
        for col in common_columns
        if str(reference_df[col].dtype) != str(current_df[col].dtype)
    }

    return {
        "missing_columns": sorted(ref_cols - curr_cols),
        "extra_columns": sorted(curr_cols - ref_cols),
        "dtype_changes": dtype_changes,
    }

def compare_missing_values(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    common_columns = list(set(reference_df.columns) & set(current_df.columns))
    if not common_columns:
        return {}

    # Vectorized computation across common columns
    ref_counts = reference_df[common_columns].isna().sum()
    curr_counts = current_df[common_columns].isna().sum()

    ref_len = len(reference_df) if len(reference_df) > 0 else 1
    curr_len = len(current_df) if len(current_df) > 0 else 1

    ref_rates = (ref_counts / ref_len).round(4)
    curr_rates = (curr_counts / curr_len).round(4)
    diffs = (curr_rates - ref_rates).round(4)

    return {
        col: {
            "reference_count": int(ref_counts[col]),
            "current_count": int(curr_counts[col]),
            "reference_rate": float(ref_rates[col]),
            "current_rate": float(curr_rates[col]),
            "difference": float(diffs[col]),
        }
        for col in common_columns
    }

def calculate_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    common_columns = set(reference_df.columns) & set(current_df.columns)
    drift_scores = {}

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

        drift_scores[column] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": bool(p_value < alpha),
        }

    return drift_scores

def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    drift_scores = calculate_drift(reference_df, current_df)
    
    return {
        "schema": compare_schema(reference_df, current_df),
        "missing_values": compare_missing_values(reference_df, current_df),
        "drift_scores": drift_scores,
        "overall_drift_detected": any(res["drift_detected"] for res in drift_scores.values()),
    }