
# # import io
# # import os
# # import joblib
# # import numpy as np
# # import pandas as pd
# # from functools import lru_cache
# # from fastapi import APIRouter, UploadFile, File, HTTPException, status
# # from datetime import datetime, timezone

# # from app.repositories.prediction_repository import save_prediction
# # from app.repositories.training_repository import get_training_by_dataset

# # router = APIRouter()

# # MODEL_FOLDER = "app/models/trained_models"


# # def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
# #     """Detects and expands explicit date/time columns into numerical components."""
# #     df = df.copy()
# #     for col in list(df.columns):
# #         if df[col].dtype == "object" and ("date" in col.lower() or "time" in col.lower()):
# #             try:
# #                 parsed_dates = pd.to_datetime(df[col], errors="coerce")
# #                 if parsed_dates.notnull().sum() > 0:
# #                     if parsed_dates.isnull().any():
# #                         parsed_dates = parsed_dates.ffill().bfill()

# #                     df[f"{col}_year"] = parsed_dates.dt.year
# #                     df[f"{col}_month"] = parsed_dates.dt.month
# #                     df[f"{col}_day"] = parsed_dates.dt.day
# #                     df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
# #                     df = df.drop(columns=[col])
# #             except (ValueError, TypeError):
# #                 continue
# #     return df


# # @lru_cache(maxsize=32)
# # def load_pipeline(dataset_id: str):
# #     model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")

# #     if not os.path.exists(model_path):
# #         raise HTTPException(
# #             status_code=status.HTTP_404_NOT_FOUND,
# #             detail=f"Model artifact not found for dataset ID: {dataset_id}",
# #         )

# #     try:
# #         return joblib.load(model_path)
# #     except Exception as e:
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"Failed to load model pipeline: {str(e)}",
# #         )


# # @router.post("/prediction/{dataset_id}")
# # async def predict(dataset_id: str, file: UploadFile = File(...)):
# #     if not file.filename.lower().endswith(".csv"):
# #         raise HTTPException(
# #             status_code=status.HTTP_400_BAD_REQUEST,
# #             detail="Invalid file format. Only CSV files are accepted."
# #         )

# #     # 1. Fetch pre-loaded pipeline
# #     pipeline = load_pipeline(dataset_id)

# #     # 2. Read file safely
# #     try:
# #         contents = await file.read()
# #         df_raw = pd.read_csv(io.BytesIO(contents))
# #         df_raw.columns = df_raw.columns.str.strip()
# #     except Exception as e:
# #         raise HTTPException(
# #             status_code=status.HTTP_400_BAD_REQUEST,
# #             detail=f"Malformed CSV file: {str(e)}"
# #         )

# #     if df_raw.empty:
# #         raise HTTPException(
# #             status_code=status.HTTP_400_BAD_REQUEST,
# #             detail="Uploaded CSV file is empty."
# #         )

# #     # 3. Preprocess and Predict
# #     try:
# #         df_processed = extract_date_features(df_raw)

# #         if hasattr(pipeline, "feature_names_in_"):
# #             expected_columns = list(pipeline.feature_names_in_)
# #             missing = set(expected_columns) - set(df_processed.columns)
# #             if missing:
# #                 # Updated status code to avoid deprecation warning
# #                 raise HTTPException(
# #                     status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
# #                     detail=f"Missing required feature columns: {list(missing)}"
# #                 )
# #             df_processed = df_processed[expected_columns]

# #         predictions = pipeline.predict(df_processed)
        
# #         # Replace non-compliant float values in predictions
# #         cleaned_predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
# #         prediction_list = cleaned_predictions.tolist()

# #         # Fetch metadata
# #         training = await get_training_by_dataset(dataset_id)
# #         model_name = training.get("best_model", "Unknown") if training else "Unknown"

# #         prediction_document = {
# #             "dataset_id": dataset_id,
# #             "model_name": model_name,
# #             "file_filename": file.filename,
# #             "rows_predicted": len(prediction_list),
# #             "predictions": prediction_list,
# #             "created_at": datetime.now(timezone.utc)
# #         }

# #         prediction_id = await save_prediction(prediction_document)

# #         # 4. Attach prediction & sanitize NaN/nulls to None (valid JSON null)
# #         df_raw["Prediction"] = prediction_list
# #         df_sanitized = df_raw.replace({np.nan: None, np.inf: None, -np.inf: None})

# #         return {
# #             "prediction_id": str(prediction_id) if prediction_id else None,
# #             "dataset_id": dataset_id,
# #             "model_name": model_name,
# #             "rows_predicted": len(prediction_list),
# #             "predictions": prediction_list,
# #             "table": df_sanitized.to_dict(orient="records")
# #         }

# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail=f"Inference processing failure: {str(e)}"
# #         )
# import io
# import os
# import joblib
# import numpy as np
# import pandas as pd
# from functools import lru_cache
# from datetime import datetime, timezone

# from fastapi import APIRouter, UploadFile, File, HTTPException, status

# from app.repositories.prediction_repository import save_prediction
# from app.repositories.training_repository import get_training_by_dataset

# router = APIRouter()

# MODEL_FOLDER = "app/models/trained_models"


# @lru_cache(maxsize=32)
# def load_pipeline(dataset_id: str):
#     model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")

#     if not os.path.exists(model_path):
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Model artifact not found for dataset ID: {dataset_id}",
#         )

#     try:
#         return joblib.load(model_path)
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to load model pipeline: {str(e)}",
#         )


# @router.post("/prediction/{dataset_id}")
# async def predict(
#     dataset_id: str,
#     file: UploadFile = File(...)
# ):
#     if not file.filename.lower().endswith(".csv"):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Only CSV files are accepted."
#         )

#     pipeline = load_pipeline(dataset_id)

#     # 1. Parse CSV safely
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))
#         df.columns = df.columns.str.strip()
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Malformed CSV file: {str(e)}"
#         )

#     if df.empty:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Uploaded CSV file is empty."
#         )

#     # 2. Fetch metadata
#     training = await get_training_by_dataset(dataset_id)
#     model_name = training.get("best_model", "Unknown") if training else "Unknown"
#     target_column = training.get("target_column") if training else None

#     # Drop target column if user included it in the prediction CSV
#     if target_column and target_column in df.columns:
#         features_df = df.drop(columns=[target_column])
#     else:
#         features_df = df.copy()

#     # 3. Inference
#     try:
#         raw_predictions = pipeline.predict(features_df)

#         cleaned_predictions = np.nan_to_num(
#             raw_predictions,
#             nan=0.0,
#             posinf=0.0,
#             neginf=0.0
#         )
#         prediction_list = cleaned_predictions.tolist()

#         # Check for prediction probabilities (Classification)
#         probabilities = None
#         if hasattr(pipeline, "predict_proba"):
#             try:
#                 prob_array = pipeline.predict_proba(features_df)
#                 probabilities = prob_array.tolist()
#             except Exception:
#                 probabilities = None

#         # 4. Save metadata document
#         prediction_document = {
#             "dataset_id": dataset_id,
#             "model_name": model_name,
#             "file_filename": file.filename,
#             "rows_predicted": len(prediction_list),
#             "predictions": prediction_list,
#             "created_at": datetime.now(timezone.utc)
#         }

#         prediction_id = await save_prediction(prediction_document)

#         # 5. Format JSON-safe response table
#         result_df = df.copy()
#         result_df["Prediction"] = prediction_list

#         result_df = result_df.replace({
#             np.nan: None,
#             np.inf: None,
#             -np.inf: None
#         })

#         response_payload = {
#             "prediction_id": str(prediction_id) if prediction_id else None,
#             "dataset_id": dataset_id,
#             "model_name": model_name,
#             "rows_predicted": len(prediction_list),
#             "predictions": prediction_list,
#             "table": result_df.to_dict(orient="records")
#         }

#         if probabilities is not None:
#             response_payload["probabilities"] = probabilities

#         return response_payload

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Inference processing failure: {str(e)}"
#         )
import io
import os
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.repositories.prediction_repository import save_prediction
from app.repositories.training_repository import get_training_by_dataset

router = APIRouter()

MODEL_FOLDER = "app/models/trained_models"


def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands date columns to Date_year, Date_month, Date_day, Date_dayofweek
    matching what the ColumnTransformer was fitted on.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in list(df.columns):
        col_lower = str(col).lower()
        if "date" in col_lower or "time" in col_lower:
            try:
                parsed_dates = pd.to_datetime(df[col], errors="coerce")
                if parsed_dates.notnull().sum() > 0:
                    if parsed_dates.isnull().any():
                        parsed_dates = parsed_dates.ffill().bfill()

                    df[f"{col}_year"] = parsed_dates.dt.year
                    df[f"{col}_month"] = parsed_dates.dt.month
                    df[f"{col}_day"] = parsed_dates.dt.day
                    df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
                    df = df.drop(columns=[col])
            except Exception:
                continue
    return df


@lru_cache(maxsize=32)
def load_pipeline(dataset_id: str):
    model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifact not found for dataset ID: {dataset_id}",
        )

    try:
        return joblib.load(model_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model pipeline: {str(e)}",
        )


@router.post("/prediction/{dataset_id}")
async def predict(
    dataset_id: str,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted."
        )

    # 1. Load pipeline (clear cache to avoid stale models)
    load_pipeline.cache_clear()
    pipeline = load_pipeline(dataset_id)

    # 2. Read CSV safely
    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
        df_raw.columns = df_raw.columns.str.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed CSV file: {str(e)}"
        )

    if df_raw.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty."
        )

    # 3. Transform dates & prepare inference dataframe
    try:
        features_df = prepare_features_for_inference(df_raw)

        # Drop ground-truth target column if present in the uploaded test file
        training = await get_training_by_dataset(dataset_id)
        if training and "target_column" in training:
            target_col = training["target_column"]
            if target_col in features_df.columns:
                features_df = features_df.drop(columns=[target_col])

        # 4. Predict
        raw_predictions = pipeline.predict(features_df)

        cleaned_predictions = np.nan_to_num(
            raw_predictions,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        prediction_list = cleaned_predictions.tolist()

        # Check for classification probabilities
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(features_df).tolist()
            except Exception:
                probabilities = None

        # 5. Save prediction document
        model_name = training.get("best_model", "Unknown") if training else "Unknown"
        prediction_document = {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "file_filename": file.filename,
            "rows_predicted": len(prediction_list),
            "predictions": prediction_list,
            "created_at": datetime.now(timezone.utc)
        }

        prediction_id = await save_prediction(prediction_document)

        # 6. Format JSON response
        result_df = df_raw.copy()
        result_df["Prediction"] = prediction_list
        result_df = result_df.replace({np.nan: None, np.inf: None, -np.inf: None})

        response = {
            "prediction_id": str(prediction_id) if prediction_id else None,
            "dataset_id": dataset_id,
            "model_name": model_name,
            "rows_predicted": len(prediction_list),
            "predictions": prediction_list,
            "table": result_df.to_dict(orient="records")
        }

        if probabilities is not None:
            response["probabilities"] = probabilities

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failure: {str(e)}"
        )