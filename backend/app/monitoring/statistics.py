import pandas as pd


def get_dataset_statistics(df: pd.DataFrame):

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    statistics = {}

    for column in numeric_columns:

        statistics[column] = {
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max())
        }

    return statistics