# # # import os
# # # import joblib
# # # import pandas as pd



# # # async def predict(dataset_id: str, data: dict):

# # #     MODEL_FOLDER = "app/models/trained_models"
# # #     PREPROCESSING_FOLDER = "app/models/preprocessing"

# # #     model_path = os.path.join(
# # #         MODEL_FOLDER,
# # #         f"{dataset_id}_best_model.pkl"
# # #     )

# # #     encoder_path = os.path.join(
# # #         PREPROCESSING_FOLDER,
# # #         f"{dataset_id}_encoder.pkl"
# # #     )

# # #     scaler_path = os.path.join(
# # #         PREPROCESSING_FOLDER,
# # #         f"{dataset_id}_scaler.pkl"
# # #     )

# # #     model = joblib.load(model_path)

# # #     encoders = joblib.load(encoder_path)

# # #     scaler = joblib.load(scaler_path)

# # #     df = pd.DataFrame([data])

# # #     for column, encoder in encoders.items():

# # #         if column in df.columns:

# # #             df[column] = df[column].apply(
# # #             lambda x: encoder.transform([x])[0]
# # #                 if x in encoder.classes_
# # #                 else -1
# # #             )

# # #     numeric_columns = df.select_dtypes(
# # #         include=["int64", "float64"]
# # #     ).columns

# # #     df[numeric_columns] = scaler.transform(
# # #         df[numeric_columns]
# # #     )        

# # #     prediction = model.predict(df)

# # #     return {
# # #         "prediction": prediction[0]
# # #     }


# # import os
# # import joblib
# # import pandas as pd
# # from fastapi import UploadFile


# # async def predict(dataset_id: str, file: UploadFile):

# #     MODEL_FOLDER = "app/models/trained_models"
# #     PREPROCESSING_FOLDER = "app/models/preprocessing"


# #     model_path = os.path.join(
# #         MODEL_FOLDER,
# #         f"{dataset_id}_best_model.pkl"
# #     )

# #     encoder_path = os.path.join(
# #         PREPROCESSING_FOLDER,
# #         f"{dataset_id}_encoder.pkl"
# #     )

# #     scaler_path = os.path.join(
# #         PREPROCESSING_FOLDER,
# #         f"{dataset_id}_scaler.pkl"
# #     )


# #     if not os.path.exists(model_path):
# #         raise FileNotFoundError(
# #             "Model artifact not found"
# #         )

# #     if not os.path.exists(encoder_path):
# #         raise FileNotFoundError(
# #             "Encoder artifact not found"
# #         )

# #     if not os.path.exists(scaler_path):
# #         raise FileNotFoundError(
# #             "Scaler artifact not found"
# #         )


# #     model = joblib.load(model_path)

# #     encoders = joblib.load(encoder_path)

# #     scaler = joblib.load(scaler_path)


# #     df = pd.DataFrame([file.file])


# #     # Validate features

# #     if hasattr(model, "feature_names_in_"):

# #         expected_columns = list(
# #             model.feature_names_in_
# #         )

# #         missing = set(expected_columns) - set(df.columns)

# #         if missing:
# #             raise ValueError(
# #                 f"Missing features: {missing}"
# #             )

# #         df = df[expected_columns]


# #     # Encode categorical columns

# #     for column, encoder in encoders.items():

# #         if column in df.columns:

# #             df[column] = df[column].apply(
# #                 lambda x:
# #                 encoder.transform([x])[0]
# #                 if x in encoder.classes_
# #                 else -1
# #             )


# #     numeric_columns = df.select_dtypes(
# #         include=["int64","float64"]
# #     ).columns


# #     if len(numeric_columns) > 0:

# #         df[numeric_columns] = scaler.transform(
# #             df[numeric_columns]
# #         )


# #     prediction = model.predict(df)

# #     df["Prediction"] = prediction

# #     return {
# #         "dataset_id": dataset_id,
# #         "predictions": df.to_dict(orient="records")
# #     }
# import os
# import io
# import joblib
# import pandas as pd
# from fastapi import UploadFile

# async def predict(dataset_id: str, file: UploadFile):

#     MODEL_FOLDER = "app/models/trained_models"
#     PREPROCESSING_FOLDER = "app/models/preprocessing"

#     model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")
#     encoder_path = os.path.join(PREPROCESSING_FOLDER, f"{dataset_id}_encoder.pkl")
#     scaler_path = os.path.join(PREPROCESSING_FOLDER, f"{dataset_id}_scaler.pkl")

#     if not os.path.exists(model_path):
#         raise FileNotFoundError("Model artifact not found")

#     model = joblib.load(model_path)
#     encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}
#     scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

#     # Read uploaded CSV file
#     contents = await file.read()
#     df = pd.read_csv(io.BytesIO(contents))

#     # 1. Strip whitespace from column names to prevent string mismatch errors
#     df.columns = df.columns.str.strip()

#     # 2. Extract expected columns from trained model dynamically
#     if hasattr(model, "feature_names_in_"):
#         expected_columns = list(model.feature_names_in_)
        
#         # Check if any required feature is missing in uploaded file
#         missing = set(expected_columns) - set(df.columns)
#         if missing:
#             raise ValueError(f"Uploaded CSV is missing expected features for this dataset: {missing}")

#         # Keep and order only the columns used during training
#         df = df[expected_columns]

#     # 3. Apply categorical encoders dynamically
#     for column, encoder in encoders.items():
#         if column in df.columns:
#             df[column] = df[column].astype(str).apply(
#                 lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
#             )

#     # 4. Apply scaler dynamically
#     if scaler is not None:
#         numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
#         if len(numeric_columns) > 0:
#             df[numeric_columns] = scaler.transform(df[numeric_columns])

#     # 5. Run prediction
#     predictions = model.predict(df)
#     df["Prediction"] = predictions

#     return {
#         "dataset_id": dataset_id,
#         "predictions": df.to_dict(orient="records")
#     }

import io
import os
import joblib
import pandas as pd
from functools import lru_cache
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from datetime import datetime, UTC
from app.repositories.prediction_repository import save_prediction
from app.repositories.training_repository import get_training_by_dataset

router = APIRouter()

MODEL_FOLDER = "app/models/trained_models"
PREPROCESSING_FOLDER = "app/models/preprocessing"


# Caches loaded artifacts in memory so disk I/O happens only on the first request
@lru_cache(maxsize=32)
def load_model_artifacts(dataset_id: str):
    model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")
    encoder_path = os.path.join(PREPROCESSING_FOLDER, f"{dataset_id}_encoder.pkl")
    scaler_path = os.path.join(PREPROCESSING_FOLDER, f"{dataset_id}_scaler.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifact not found for dataset ID: {dataset_id}",
        )

    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path) if os.path.exists(encoder_path) else {}
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        return model, encoders, scaler
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model artifacts: {str(e)}",
        )


@router.post("/prediction/{dataset_id}")
async def predict(dataset_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV files are accepted."
        )

    # 1. Fetch pre-loaded model, encoders, and scaler from cache
    model, encoders, scaler = load_model_artifacts(dataset_id)

    # 2. Read uploaded file safely
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed CSV file: {str(e)}"
        )

    # 3. Preprocess and Predict
    try:
        # Validate required feature columns
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)
            missing = set(expected_columns) - set(df.columns)
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Missing required feature columns: {missing}"
                )
            df = df[expected_columns]

        # Apply encoders
        for column, encoder in encoders.items():
            if column in df.columns:
                df[column] = df[column].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )

        # Apply scaler
        if scaler is not None:
            numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
            if len(numeric_columns) > 0:
                df[numeric_columns] = scaler.transform(df[numeric_columns])

        # Generate predictions
        predictions = model.predict(df)
        prediction_list = predictions.tolist()

        training = await get_training_by_dataset(dataset_id)

        model_name = training["best_model"] if training else "Unknown"

        prediction_document = {
            "dataset_id": dataset_id,
            "model_name": model_name,
            "file_filename": file.filename,
            "rows_predicted": len(prediction_list),
            "predictions": prediction_list,
            "created_at": datetime.now(UTC)
        }

        prediction_id = await save_prediction(
                prediction_document
        )


        df["Prediction"] = predictions

        

        return {
            "prediction_id": prediction_id,
            "dataset_id": dataset_id,
            "rows_predicted": len(prediction_list),
            "predictions": prediction_list,
            "table": df.to_dict(orient="records")
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failure: {str(e)}"
        )