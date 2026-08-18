
# import os
# from datetime import datetime, UTC

# import joblib
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from app.repositories.preprocessing_repository import save_preprocessing


# PROCESSED_FOLDER = "datasets/processed"
# ARTIFACT_FOLDER = "app/models/preprocessing"

# os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# os.makedirs(ARTIFACT_FOLDER, exist_ok=True)


# async def preprocess_dataset(
#     file_path: str,
#     dataset_id: str,
#     target_column: str,
#     outlier_strategy: str = "cap",
#     outlier_threshold: float = 1.5
# ):
#     if outlier_strategy not in ["remove", "cap"]:
#         raise ValueError("outlier_strategy must be 'remove' or 'cap'")

#     # Load dataset & strip column names
#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()
#     target_column = target_column.strip()

#     original_rows = len(df)

#     # Remove duplicates
#     duplicates_removed = int(df.duplicated().sum())
#     df = df.drop_duplicates().reset_index(drop=True)

#     # Separate target
#     if target_column not in df.columns:
#         raise ValueError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

#     y = df[target_column].copy()
#     X = df.drop(columns=[target_column]).copy()

#     # Handle missing target
#     if y.isnull().sum() > 0:
#         if y.dtype == "object":
#             y = y.fillna(y.mode()[0])
#         else:
#             y = y.fillna(y.mean())

#     # ---------------------------------------------------------
#     # 1. Handle Date Columns
#     # ---------------------------------------------------------
#     date_columns = []
#     for column in list(X.columns):
#         if "date" in column.lower() or "time" in column.lower():
#             try:
#                 parsed_dates = pd.to_datetime(X[column], errors="coerce")
                
#                 # Verify if it was successfully parsed as datetime
#                 if parsed_dates.notnull().sum() > 0:
#                     # Forward-fill / back-fill missing date values
#                     if parsed_dates.isnull().any():
#                         parsed_dates = parsed_dates.ffill().bfill()

#                     # Extract numerical features
#                     X[f"{column}_year"] = parsed_dates.dt.year
#                     X[f"{column}_month"] = parsed_dates.dt.month
#                     X[f"{column}_day"] = parsed_dates.dt.day
#                     X[f"{column}_dayofweek"] = parsed_dates.dt.dayofweek

#                     # Drop the original string/object date column
#                     X = X.drop(columns=[column])
#                     date_columns.append(column)
#             except Exception:
#                 pass

#     # ---------------------------------------------------------
#     # 2. Handle Missing Values in Features
#     # ---------------------------------------------------------
#     for column in X.columns:
#         if X[column].dtype == "object" or X[column].dtype.name == "category":
#             mode_series = X[column].mode()
#             fill_val = mode_series[0] if not mode_series.empty else "missing"
#             X[column] = X[column].fillna(fill_val)
#         else:
#             mean_val = X[column].mean()
#             fill_val = mean_val if pd.notnull(mean_val) else 0.0
#             X[column] = X[column].fillna(fill_val)

#     # ---------------------------------------------------------
#     # 3. Outlier Handling
#     # ---------------------------------------------------------
#     numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     outliers_removed = {}
#     outliers_capped = {}

#     for column in numeric_columns:
#         Q1 = X[column].quantile(0.25)
#         Q3 = X[column].quantile(0.75)
#         IQR = Q3 - Q1

#         lower = Q1 - outlier_threshold * IQR
#         upper = Q3 + outlier_threshold * IQR

#         if outlier_strategy == "remove":
#             mask = (X[column] >= lower) & (X[column] <= upper)
#             before = len(X)
#             X = X[mask].copy()
#             y = y[mask].copy()
#             after = len(X)
#             outliers_removed[column] = int(before - after)
#         else:
#             before = X[column].copy()
#             X[column] = X[column].clip(lower, upper)
#             outliers_capped[column] = int((before != X[column]).sum())

#     # ---------------------------------------------------------
#     # 4. Encode Categorical Columns
#     # ---------------------------------------------------------
#     encoders = {}
#     categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

#     for column in categorical_columns:
#         encoder = LabelEncoder()
#         X[column] = X[column].astype(str)
#         X[column] = encoder.fit_transform(X[column])
#         encoders[column] = encoder

#     # ---------------------------------------------------------
#     # 5. Scale Numerical Features
#     # ---------------------------------------------------------
#     numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     scaler = StandardScaler()

#     if len(numeric_columns) > 0:
#         X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

#     # Final alignment & checks
#     X = X.reset_index(drop=True)
#     y = y.reset_index(drop=True)

#     if y.isnull().sum() > 0:
#         raise ValueError("Target still contains NaN after preprocessing")

#     # ---------------------------------------------------------
#     # 6. Save Artifacts & Processed CSV
#     # ---------------------------------------------------------
#     artifact_payload = {
#         "encoders": encoders,
#         "scaler": scaler,
#         "numeric_columns": numeric_columns,
#         "categorical_columns": categorical_columns,
#         "date_columns": date_columns,
#         "feature_order": list(X.columns)
#     }

#     pipeline_artifact_path = os.path.join(ARTIFACT_FOLDER, f"{dataset_id}_preprocessor.pkl")
#     joblib.dump(artifact_payload, pipeline_artifact_path)

#     # Legacy individual artifact saves for backward compatibility
#     encoder_path = os.path.join(ARTIFACT_FOLDER, f"{dataset_id}_encoder.pkl")
#     scaler_path = os.path.join(ARTIFACT_FOLDER, f"{dataset_id}_scaler.pkl")
#     joblib.dump(encoders, encoder_path)
#     joblib.dump(scaler, scaler_path)

#     # Attach target back and save
#     processed_df = X.copy()
#     processed_df[target_column] = y.values

#     processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{dataset_id}.csv")
#     processed_df.to_csv(processed_path, index=False)

#     result = {
#         "dataset_id": dataset_id,
#         "processed_file_path": processed_path,
#         "original_rows": original_rows,
#         "processed_rows": len(processed_df),
#         "missing_values_handled": True,
#         "duplicates_removed": duplicates_removed,
#         "date_columns_extracted": date_columns,
#         "missing_value_method": "Mean numeric, Mode categorical",
#         "encoding_method": "LabelEncoder",
#         "scaling_method": "StandardScaler",
#         "outlier_strategy": outlier_strategy,
#         "outlier_threshold": outlier_threshold,
#         "outliers_removed": outliers_removed,
#         "outliers_capped": outliers_capped,
#         "pipeline_artifact_path": pipeline_artifact_path,
#         "encoder_path": encoder_path,
#         "scaler_path": scaler_path,
#         "created_at": datetime.now(UTC)
#     }

#     preprocessing_id = await save_preprocessing(result)
#     result.pop("_id", None)
#     result["preprocessing_id"] = preprocessing_id
#     return result

# import os
# from datetime import datetime, UTC

# import joblib
# import pandas as pd

# from app.repositories.preprocessing_repository import save_preprocessing


# PROCESSED_FOLDER = "datasets/processed"
# ARTIFACT_FOLDER = "app/models/preprocessing"

# os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# os.makedirs(ARTIFACT_FOLDER, exist_ok=True)


# async def preprocess_dataset(
#     file_path: str,
#     dataset_id: str,
#     target_column: str,
#     outlier_strategy: str = "cap",
#     outlier_threshold: float = 1.5
# ):
#     """
#     Performs dataset-level preprocessing without fitting ML transformations.

#     Important:
#     - Target missing values are DROPPED, never imputed.
#     - Date columns are expanded into numerical features.
#     - Feature missing values are left for the training Pipeline.
#     - Encoding is handled by OneHotEncoder inside train_models().
#     - Scaling is handled by StandardScaler inside train_models().
#     - Outlier treatment is NOT performed here because doing IQR calculations
#       before train/test splitting causes data leakage.

#     This service creates a clean structural dataset.
#     """

#     if outlier_strategy not in ["remove", "cap"]:
#         raise ValueError(
#             "outlier_strategy must be 'remove' or 'cap'"
#         )

#     if outlier_threshold <= 0:
#         raise ValueError(
#             "outlier_threshold must be greater than 0"
#         )

#     # ---------------------------------------------------------
#     # 1. LOAD DATASET
#     # ---------------------------------------------------------

#     df = pd.read_csv(file_path)

#     if df.empty:
#         raise ValueError("Dataset is empty.")

#     df.columns = df.columns.str.strip()
#     target_column = target_column.strip()

#     if target_column not in df.columns:
#         raise ValueError(
#             f"Target column '{target_column}' not found. "
#             f"Available columns: {list(df.columns)}"
#         )

#     original_rows = len(df)

#     # ---------------------------------------------------------
#     # 2. REMOVE DUPLICATES
#     # ---------------------------------------------------------

#     duplicates_removed = int(df.duplicated().sum())

#     df = (
#         df
#         .drop_duplicates()
#         .reset_index(drop=True)
#     )

#     # ---------------------------------------------------------
#     # 3. REMOVE ROWS WITH MISSING TARGET
#     # ---------------------------------------------------------

#     target_missing_before = int(
#         df[target_column].isna().sum()
#     )

#     # IMPORTANT:
#     # Never mean-impute the target.
#     df = (
#         df
#         .dropna(subset=[target_column])
#         .reset_index(drop=True)
#     )

#     if df.empty:
#         raise ValueError(
#             "No valid rows remain after removing missing target values."
#         )

#     # ---------------------------------------------------------
#     # 4. SEPARATE TARGET
#     # ---------------------------------------------------------

#     y = df[target_column].copy()
#     X = df.drop(columns=[target_column]).copy()

#     # ---------------------------------------------------------
#     # 5. DATE FEATURE EXTRACTION
#     # ---------------------------------------------------------

#     date_columns = []

#     for column in list(X.columns):

#         column_name = column.lower()

#         # Only attempt date parsing for columns that look like
#         # dates/timestamps.
#         if "date" not in column_name and "time" not in column_name:
#             continue

#         try:

#             parsed_dates = pd.to_datetime(
#                 X[column],
#                 errors="coerce"
#             )

#             valid_dates = int(
#                 parsed_dates.notna().sum()
#             )

#             # Require at least one successfully parsed date.
#             if valid_dates == 0:
#                 continue

#             X[f"{column}_year"] = parsed_dates.dt.year
#             X[f"{column}_month"] = parsed_dates.dt.month
#             X[f"{column}_day"] = parsed_dates.dt.day
#             X[f"{column}_dayofweek"] = (
#                 parsed_dates.dt.dayofweek
#             )

#             # Drop original date column.
#             X = X.drop(columns=[column])

#             date_columns.append(column)

#         except (ValueError, TypeError):
#             continue

#     # ---------------------------------------------------------
#     # 6. DO NOT IMPUTE FEATURES HERE
#     # ---------------------------------------------------------

#     # Missing values are intentionally preserved.
#     #
#     # train_models() handles:
#     #
#     # Numeric:
#     #     SimpleImputer(strategy="median")
#     #
#     # Categorical:
#     #     SimpleImputer(strategy="most_frequent")
#     #
#     # This prevents preprocessing leakage.

#     missing_summary = {
#         column: int(X[column].isna().sum())
#         for column in X.columns
#         if X[column].isna().sum() > 0
#     }

#     # ---------------------------------------------------------
#     # 7. DO NOT ENCODE HERE
#     # ---------------------------------------------------------

#     # LabelEncoder has been removed.
#     #
#     # Feature categorical encoding is performed inside
#     # train_models() using:
#     #
#     # OneHotEncoder(handle_unknown="ignore")
#     #
#     # This is safer for inference because unseen categories
#     # won't crash the prediction pipeline.

#     categorical_columns = (
#         X
#         .select_dtypes(
#             include=["object", "category", "bool"]
#         )
#         .columns
#         .tolist()
#     )

#     # ---------------------------------------------------------
#     # 8. DO NOT SCALE HERE
#     # ---------------------------------------------------------

#     # StandardScaler has been removed.
#     #
#     # Scaling belongs inside the sklearn Pipeline so it is fitted
#     # ONLY on X_train.

#     numeric_columns = (
#         X
#         .select_dtypes(include=["number"])
#         .columns
#         .tolist()
#     )

#     # ---------------------------------------------------------
#     # 9. OUTLIER HANDLING
#     # ---------------------------------------------------------

#     # IMPORTANT:
#     #
#     # We intentionally DO NOT calculate IQR here.
#     #
#     # If we calculate:
#     #
#     # Q1, Q3, IQR
#     #
#     # using the complete dataset before train_test_split(),
#     # information from the test set leaks into training.
#     #
#     # Outlier handling should be moved into the training
#     # pipeline and fitted only on X_train.
#     #
#     # For now, record that it is deferred.

#     outliers_removed = {}
#     outliers_capped = {}

#     outlier_handling = (
#         "Deferred to training pipeline to prevent data leakage."
#     )

#     # ---------------------------------------------------------
#     # 10. FINAL DATASET
#     # ---------------------------------------------------------

#     X = X.reset_index(drop=True)
#     y = y.reset_index(drop=True)

#     processed_df = X.copy()
#     processed_df[target_column] = y

#     if processed_df.empty:
#         raise ValueError(
#             "Processed dataset is empty."
#         )

#     # ---------------------------------------------------------
#     # 11. SAVE PROCESSED DATASET
#     # ---------------------------------------------------------

#     processed_path = os.path.join(
#         PROCESSED_FOLDER,
#         f"processed_{dataset_id}.csv"
#     )

#     processed_df.to_csv(
#         processed_path,
#         index=False
#     )

#     # ---------------------------------------------------------
#     # 12. SAVE METADATA
#     # ---------------------------------------------------------

#     artifact_payload = {
#         "dataset_id": dataset_id,
#         "target_column": target_column,
#         "date_columns": date_columns,
#         "numeric_columns": numeric_columns,
#         "categorical_columns": categorical_columns,
#         "feature_order": list(X.columns),
#         "encoding_method": (
#             "OneHotEncoder inside training pipeline"
#         ),
#         "scaling_method": (
#             "StandardScaler inside training pipeline"
#         ),
#         "missing_value_method": (
#             "SimpleImputer inside training pipeline"
#         ),
#         "outlier_strategy": outlier_strategy,
#         "outlier_threshold": outlier_threshold,
#         "outlier_handling": outlier_handling,
#     }

#     pipeline_artifact_path = os.path.join(
#         ARTIFACT_FOLDER,
#         f"{dataset_id}_preprocessor.pkl"
#     )

#     joblib.dump(
#         artifact_payload,
#         pipeline_artifact_path
#     )

#     # ---------------------------------------------------------
#     # 13. RESULT
#     # ---------------------------------------------------------

#     result = {
#         "dataset_id": dataset_id,
#         "processed_file_path": processed_path,

#         "original_rows": original_rows,
#         "processed_rows": len(processed_df),

#         "duplicates_removed": duplicates_removed,

#         "target_missing_before": target_missing_before,
#         "target_missing_handling": "rows_removed",

#         "missing_values": missing_summary,

#         "date_columns_extracted": date_columns,

#         "numeric_columns": numeric_columns,
#         "categorical_columns": categorical_columns,

#         "missing_value_method": (
#             "Handled inside training pipeline"
#         ),

#         "encoding_method": (
#             "OneHotEncoder inside training pipeline"
#         ),

#         "scaling_method": (
#             "StandardScaler inside training pipeline"
#         ),

#         "outlier_strategy": outlier_strategy,
#         "outlier_threshold": outlier_threshold,

#         "outlier_handling": outlier_handling,

#         "outliers_removed": outliers_removed,
#         "outliers_capped": outliers_capped,

#         "pipeline_artifact_path": pipeline_artifact_path,

#         "created_at": datetime.now(UTC),
#     }

#     # ---------------------------------------------------------
#     # 14. SAVE TO MONGODB
#     # ---------------------------------------------------------

#     preprocessing_id = await save_preprocessing(
#         result
#     )

#     result.pop("_id", None)

#     result["preprocessing_id"] = preprocessing_id

#     return result

import os
from datetime import datetime, UTC

import joblib
import pandas as pd

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
    """
    Performs dataset-level preprocessing without fitting ML transformations.

    Important:
    - Target missing values are DROPPED, never imputed.
    - Date columns are expanded into numerical features.
    - Feature missing values are left for the training Pipeline.
    - Encoding is handled by OneHotEncoder inside train_models().
    - Scaling is handled by StandardScaler inside train_models().
    - Outlier treatment is deferred to avoid data leakage before splitting.
    """

    if outlier_strategy not in ["remove", "cap"]:
        raise ValueError("outlier_strategy must be 'remove' or 'cap'")

    if outlier_threshold <= 0:
        raise ValueError("outlier_threshold must be greater than 0")

    # ---------------------------------------------------------
    # 1. LOAD DATASET
    # ---------------------------------------------------------
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    df.columns = df.columns.str.strip()
    target_column = target_column.strip()

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    original_rows = len(df)

    # ---------------------------------------------------------
    # 2. REMOVE DUPLICATES
    # ---------------------------------------------------------
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. REMOVE ROWS WITH MISSING TARGET
    # ---------------------------------------------------------
    target_missing_before = int(df[target_column].isna().sum())
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after removing missing target values.")

    # ---------------------------------------------------------
    # 4. SEPARATE TARGET
    # ---------------------------------------------------------
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    # ---------------------------------------------------------
    # 5. DATE FEATURE EXTRACTION
    # ---------------------------------------------------------
    date_columns = []

    for column in list(X.columns):
        column_name = column.lower()
        if "date" not in column_name and "time" not in column_name:
            continue

        try:
            parsed_dates = pd.to_datetime(X[column], errors="coerce")
            valid_dates = int(parsed_dates.notna().sum())

            if valid_dates == 0:
                continue

            # Fill missing dates if partial NaNs exist
            if parsed_dates.isnull().any():
                parsed_dates = parsed_dates.ffill().bfill()

            X[f"{column}_year"] = parsed_dates.dt.year
            X[f"{column}_month"] = parsed_dates.dt.month
            X[f"{column}_day"] = parsed_dates.dt.day
            X[f"{column}_dayofweek"] = parsed_dates.dt.dayofweek

            X = X.drop(columns=[column])
            date_columns.append(column)

        except (ValueError, TypeError):
            continue

    # ---------------------------------------------------------
    # 6. FEATURE SUMMARIES (Preserve NaNs for Pipeline)
    # ---------------------------------------------------------
    missing_summary = {
        column: int(X[column].isna().sum())
        for column in X.columns
        if X[column].isna().sum() > 0
    }

    categorical_columns = (
        X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    )
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

    outlier_handling = "Deferred to training pipeline to prevent data leakage."

    # ---------------------------------------------------------
    # 7. ASSEMBLE AND SAVE PROCESSED DATASET
    # ---------------------------------------------------------
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    processed_df = X.copy()
    processed_df[target_column] = y

    if processed_df.empty:
        raise ValueError("Processed dataset is empty.")

    processed_path = os.path.join(PROCESSED_FOLDER, f"processed_{dataset_id}.csv")
    processed_df.to_csv(processed_path, index=False)

    # ---------------------------------------------------------
    # 8. SAVE ARTIFACT METADATA
    # ---------------------------------------------------------
    artifact_payload = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_order": list(X.columns),
        "encoding_method": "OneHotEncoder inside training pipeline",
        "scaling_method": "StandardScaler inside training pipeline",
        "missing_value_method": "SimpleImputer inside training pipeline",
        "outlier_strategy": outlier_strategy,
        "outlier_threshold": outlier_threshold,
        "outlier_handling": outlier_handling,
    }

    pipeline_artifact_path = os.path.join(
        ARTIFACT_FOLDER, f"{dataset_id}_preprocessor.pkl"
    )
    joblib.dump(artifact_payload, pipeline_artifact_path)

    # ---------------------------------------------------------
    # 9. RESULT PAYLOAD
    # ---------------------------------------------------------
    result = {
        "dataset_id": dataset_id,
        "processed_file_path": processed_path,
        "original_rows": original_rows,
        "processed_rows": len(processed_df),
        "duplicates_removed": duplicates_removed,
        "target_missing_before": target_missing_before,
        "target_missing_handling": "rows_removed",
        "missing_values": missing_summary,
        "date_columns_extracted": date_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_value_method": "Handled inside training pipeline",
        "encoding_method": "OneHotEncoder inside training pipeline",
        "scaling_method": "StandardScaler inside training pipeline",
        "outlier_strategy": outlier_strategy,
        "outlier_threshold": outlier_threshold,
        "outlier_handling": outlier_handling,
        "outliers_removed": {},
        "outliers_capped": {},
        "pipeline_artifact_path": pipeline_artifact_path,
        "created_at": datetime.now(UTC),
    }

    preprocessing_id = await save_preprocessing(result)
    result.pop("_id", None)
    result["preprocessing_id"] = preprocessing_id

    return result