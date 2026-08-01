import os
from datetime import datetime, UTC
import joblib

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.repositories.preprocessing_repository import save_preprocessing


PROCESSED_FOLDER = "datasets/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ARTIFACT_FOLDER = "app/models/preprocessing"

os.makedirs(
    ARTIFACT_FOLDER,
    exist_ok=True
)


async def preprocess_dataset(
    file_path: str,
    dataset_id: str,
    target_column: str,
    outlier_strategy: str = "cap",
    outlier_threshold: float = 1.5
):
    if outlier_strategy not in ["remove", "cap"]:
        raise ValueError(
            "outlier_strategy must be 'remove' or 'cap'"
        )

    # Read dataset
    df = pd.read_csv(file_path)

    original_rows = len(df)

    y=df[target_column].copy()
    df=df.drop(columns=[target_column])

  

    # Remove duplicate rows
    duplicates_removed = df.duplicated().sum()
    df = df.drop_duplicates()
    y = y.loc[df.index]   

    # Handle missing values
    for column in df.columns:

        if df[column].dtype == "object":
            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(mode[0])

        else:
            df[column] = df[column].fillna(df[column].mean())



    # Handle outliers using IQR

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns


    outliers_removed = {}
    outliers_capped = {}


    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - outlier_threshold * IQR
        upper_bound = Q3 + outlier_threshold * IQR


        if outlier_strategy == "remove":

            before = len(df)

            df = df[
                (df[column] >= lower_bound) &
                (df[column] <= upper_bound)
            ]

            y=y.loc[df.index]

            after = len(df)

            outliers_removed[column] = before - after


        elif outlier_strategy == "cap":

            before = df[column].copy()

            df[column] = df[column].clip(
                lower_bound,
                upper_bound
            )

            outliers_capped[column] = int(
                (before != df[column]).sum()
            )   

    # Encode categorical columns
    encoders = {}

    for column in df.select_dtypes(include=["object"]).columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    # Scale numerical columns
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    encoder_path = os.path.join(
        ARTIFACT_FOLDER,
        f"{dataset_id}_encoder.pkl"
    )

    scaler_path = os.path.join(
        ARTIFACT_FOLDER,
        f"{dataset_id}_scaler.pkl"
    )


    joblib.dump(
        encoders,
        encoder_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Add target column

    df[target_column] = y

    # Save processed dataset
    processed_filename = f"processed_{dataset_id}.csv"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    df.to_csv(processed_path, index=False)

    # Metadata
    result = {
        "dataset_id": dataset_id,
        "processed_file_path": processed_path,
        "original_rows": original_rows,
        "processed_rows": len(df),
        "missing_values_handled": True,
        "duplicates_removed": int(duplicates_removed),

        "missing_value_method": "Mean (numeric), Mode (categorical)",
        "encoding_method": "LabelEncoder",
        "scaling_method": "StandardScaler",
        "outlier_strategy": outlier_strategy,
        "outlier_threshold": outlier_threshold,
        "outliers_removed": outliers_removed,
        "outliers_capped": outliers_capped,
        "encoder_path": encoder_path,   
        "scaler_path": scaler_path,   

        "created_at": datetime.now(UTC)
    }

    preprocessing_id = await save_preprocessing(result)

    result.pop("_id", None)

    result["preprocessing_id"] = preprocessing_id

    return result