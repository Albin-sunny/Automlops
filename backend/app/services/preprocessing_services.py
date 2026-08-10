
import os
from datetime import datetime, UTC

import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.repositories.preprocessing_repository import save_preprocessing


PROCESSED_FOLDER = "datasets/processed"
ARTIFACT_FOLDER = "app/models/preprocessing"


os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(ARTIFACT_FOLDER, exist_ok=True)


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


    # Load dataset

    df = pd.read_csv(file_path)

    original_rows = len(df)


    # Remove duplicates

    duplicates_removed = df.duplicated().sum()

    df = df.drop_duplicates().reset_index(drop=True)



    # Separate target

    if target_column not in df.columns:
        raise ValueError(
            f"{target_column} not found in dataset"
        )


    y = df[target_column].copy()

    X = df.drop(
        columns=[target_column]
    ).copy()



    # Handle missing target

    if y.isnull().sum() > 0:

        if y.dtype == "object":
            y = y.fillna(
                y.mode()[0]
            )

        else:
            y = y.fillna(
                y.mean()
            )



    # Handle missing features

    for column in X.columns:

        if X[column].dtype == "object":

            mode = X[column].mode()

            if not mode.empty:
                X[column] = X[column].fillna(
                    mode[0]
                )

        else:

            X[column] = X[column].fillna(
                X[column].mean()
            )



    # Outlier handling

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    outliers_removed = {}
    outliers_capped = {}


    for column in numeric_columns:

        Q1 = X[column].quantile(0.25)

        Q3 = X[column].quantile(0.75)

        IQR = Q3 - Q1


        lower = Q1 - outlier_threshold * IQR

        upper = Q3 + outlier_threshold * IQR



        if outlier_strategy == "remove":

            mask = (
                (X[column] >= lower)
                &
                (X[column] <= upper)
            )


            before = len(X)


            X = X[mask].copy()

            y = y[mask].copy()


            after = len(X)


            outliers_removed[column] = (
                before - after
            )



        else:

            before = X[column].copy()


            X[column] = X[column].clip(
                lower,
                upper
            )


            outliers_capped[column] = int(
                (before != X[column]).sum()
            )



    # Encode categorical columns

    encoders = {}


    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns


    for column in categorical_columns:

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(
            X[column]
        )

        encoders[column] = encoder



    # Scale numerical features

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    scaler = StandardScaler()


    if len(numeric_columns) > 0:

        X[numeric_columns] = scaler.fit_transform(
            X[numeric_columns]
        )



    # Final alignment

    X = X.reset_index(drop=True)

    y = y.reset_index(drop=True)



    # Final NaN check

    if y.isnull().sum() > 0:

        raise ValueError(
            "Target still contains NaN"
        )



    # Save preprocessing artifacts


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



    # Add target back

    X[target_column] = y.values



    # Save processed dataset


    processed_path = os.path.join(
        PROCESSED_FOLDER,
        f"processed_{dataset_id}.csv"
    )


    X.to_csv(
        processed_path,
        index=False
    )



    result = {

        "dataset_id": dataset_id,

        "processed_file_path": processed_path,

        "original_rows": original_rows,

        "processed_rows": len(X),

        "missing_values_handled": True,

        "duplicates_removed": int(
            duplicates_removed
        ),

        "missing_value_method":
            "Mean numeric, Mode categorical",

        "encoding_method":
            "LabelEncoder",

        "scaling_method":
            "StandardScaler",

        "outlier_strategy":
            outlier_strategy,

        "outlier_threshold":
            outlier_threshold,

        "outliers_removed":
            outliers_removed,

        "outliers_capped":
            outliers_capped,

        "encoder_path":
            encoder_path,

        "scaler_path":
            scaler_path,

        "created_at":
            datetime.now(UTC)

    }
    preprocessing_id = await save_preprocessing(result)
    result.pop("_id", None)
    result["preprocessing_id"] = preprocessing_id
    return result