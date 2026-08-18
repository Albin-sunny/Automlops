import pandas as pd
from datetime import datetime
from app.repositories.profiling_repository import save_profile


async def generate_profile(file_path: str, dataset_id: str):

    df = pd.read_csv(file_path)

    rows = len(df)
    columns = len(df.columns)

    missing_values = {
        col: int(value)
        for col, value in df.isnull().sum().items()
        if value > 0
    }

    duplicate_rows = int(df.duplicated().sum())

    data_types = {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    }

    numeric_summary = (
        df.describe()
        .select_dtypes(include=["number"])
        .to_dict()
    )

    quality_score = calculate_quality_score(
        df,
        missing_values,
        duplicate_rows
    )

    profile = {
        "dataset_id": dataset_id,
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "data_types": data_types,
        "numeric_summary": numeric_summary,
        "quality_score": quality_score,
        "created_at": datetime.utcnow()
    }

    profile_id = await save_profile(profile)

    profile["profile_id"] = profile_id

    return profile


def calculate_quality_score(df, missing_values, duplicates):

    score = 100

    missing_penalty = sum(missing_values.values()) * 2
    duplicate_penalty = duplicates * 1

    score -= missing_penalty
    score -= duplicate_penalty

    return max(score, 0)



