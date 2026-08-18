
# import os
# import time
# import joblib
# import mlflow
# import mlflow.sklearn
# import mlflow.xgboost
# import pandas as pd
# import numpy as np

# from datetime import datetime, timezone
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import (
#     cross_val_score,
#     train_test_split,
#     KFold,
#     StratifiedKFold
# )
# from sklearn.metrics import (
#     accuracy_score,
#     r2_score,
#     mean_absolute_error,
#     mean_squared_error
# )

# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     RandomForestRegressor
# )
# from sklearn.neighbors import KNeighborsClassifier

# from xgboost import (
#     XGBClassifier,
#     XGBRegressor
# )

# from app.repositories.training_repository import save_training_result

# MODEL_FOLDER = "app/models/trained_models"

# # Prefer SQLite over file store for production/local stability
# tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# mlflow.set_tracking_uri(tracking_uri)
# mlflow.set_experiment("AutoMLOps")

# os.makedirs(MODEL_FOLDER, exist_ok=True)


# def log_mlflow_model(model, is_xgboost: bool, name: str = "model"):
#     """Helper to log models using cloudpickle to bypass untrusted numpy types."""
#     if is_xgboost:
#         mlflow.xgboost.log_model(
#             xgb_model=model,
#             artifact_path=name
#         )
#     else:
#         mlflow.sklearn.log_model(
#             sk_model=model,
#             artifact_path=name,
#             serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
#         )


# def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Detects and expands date/time columns into numerical components."""
#     df = df.copy()
#     for col in list(df.columns):
#         if df[col].dtype == "object" and ("date" in col.lower() or "time" in col.lower()):
#             try:
#                 parsed_dates = pd.to_datetime(df[col], errors="coerce")
#                 if parsed_dates.notnull().sum() > 0:
#                     if parsed_dates.isnull().any():
#                         parsed_dates = parsed_dates.ffill().bfill()
                    
#                     df[f"{col}_year"] = parsed_dates.dt.year
#                     df[f"{col}_month"] = parsed_dates.dt.month
#                     df[f"{col}_day"] = parsed_dates.dt.day
#                     df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
#                     df = df.drop(columns=[col])
#             except (ValueError, TypeError):
#                 continue
#     return df


# def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
#     """Builds a leak-free ColumnTransformer for numeric and categorical features."""
#     numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
#     categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

#     numeric_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])

#     categorical_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features)
#         ]
#     )
#     return preprocessor


# async def train_models(
#     file_path: str,
#     dataset_id: str,
#     target_column: str
# ):
#     start_time = time.time()

#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()
#     target_column = target_column.strip()

#     if target_column not in df.columns:
#         raise ValueError(f"Target column '{target_column}' not found in dataset.")

#     # 1. Clean Target Column: Drop rows where target (y) is NaN
#     df = df.dropna(subset=[target_column])

#     if df.empty:
#         raise ValueError(f"Target column '{target_column}' has no valid (non-null) samples.")

#     # 2. Extract Features and Target
#     y = df[target_column]
#     X_raw = df.drop(columns=[target_column])
#     X = extract_date_features(X_raw)
#     feature_columns = X.columns.tolist()

#     # 3. Determine Task Type
#     if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 2:
#         task_type = "classification"
#     elif y.nunique() < 10 and (y.dtype == "int64" or y.dtype == "int32"):
#         task_type = "classification"
#     else:
#         task_type = "regression"

#     # 4. Train-Test Split (Prior to any fitting to prevent data leakage)
#     stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=stratify
#     )

#     # 5. Cross-Validation Configuration
#     if task_type == "classification":
#         min_class_samples = y_train.value_counts().min()
#         n_splits = max(2, min(5, min_class_samples))
#         cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     else:
#         n_splits = max(2, min(5, len(X_train)))
#         cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     preprocessor = build_preprocessor(X_train)
#     results = {}
#     pipelines = {}

#     # -------------------------------------------------------------
#     # MODEL DEFINITIONS
#     # -------------------------------------------------------------
#     if task_type == "classification":
#         base_models = {
#             "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
#             "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
#             "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
#             "KNN": KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train) - 1))),
#             "XGBoost": XGBClassifier(
#                 max_depth=3,
#                 n_estimators=50,
#                 learning_rate=0.1,
#                 random_state=42,
#                 eval_metric="logloss"
#             )
#         }
#         scoring_metric = "accuracy"
#     else:
#         base_models = {
#             "LinearRegression": LinearRegression(),
#             "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
#             "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
#             "XGBoost": XGBRegressor(
#                 max_depth=3,
#                 n_estimators=50,
#                 learning_rate=0.1,
#                 random_state=42,
#                 objective="reg:squarederror"
#             )
#         }
#         scoring_metric = "r2"

#     # -------------------------------------------------------------
#     # TRAINING & CROSS-VALIDATION
#     # -------------------------------------------------------------
#     for name, model in base_models.items():
#         model_pipeline = Pipeline(steps=[
#             ("preprocessor", preprocessor),
#             ("model", model)
#         ])

#         with mlflow.start_run(run_name=name):
#             mlflow.log_param("model", name)
#             mlflow.log_param("task_type", task_type)
#             mlflow.log_param("dataset_id", dataset_id)
#             mlflow.log_param("target_column", target_column)
#             mlflow.log_param("rows", len(df))
#             mlflow.log_param("features", len(feature_columns))

#             # Cross-validation over train partition
#             cv_scores = cross_val_score(
#                 model_pipeline,
#                 X_train,
#                 y_train,
#                 cv=cv,
#                 scoring=scoring_metric
#             )
#             cv_mean = float(cv_scores.mean())
#             cv_std = float(cv_scores.std())

#             # Fit on full train partition and evaluate on test partition
#             model_pipeline.fit(X_train, y_train)
#             predictions = model_pipeline.predict(X_test)

#             if task_type == "classification":
#                 acc = float(accuracy_score(y_test, predictions))
#                 mlflow.log_metric("accuracy", acc)
#                 mlflow.log_metric("cv_mean_accuracy", cv_mean)
#                 mlflow.log_metric("cv_std_accuracy", cv_std)

#                 results[name] = {
#                     "accuracy": acc,
#                     "cv_mean_accuracy": cv_mean,
#                     "cv_std_accuracy": cv_std
#                 }
#             else:
#                 r2 = float(r2_score(y_test, predictions))
#                 mae = float(mean_absolute_error(y_test, predictions))
#                 rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

#                 mlflow.log_metric("r2_score", r2)
#                 mlflow.log_metric("mae", mae)
#                 mlflow.log_metric("rmse", rmse)
#                 mlflow.log_metric("cv_mean_r2", cv_mean)
#                 mlflow.log_metric("cv_std_r2", cv_std)

#                 results[name] = {
#                     "r2_score": r2,
#                     "mae": mae,
#                     "rmse": rmse,
#                     "cv_mean_r2": cv_mean,
#                     "cv_std_r2": cv_std
#                 }

#             log_mlflow_model(
#                 model=model_pipeline,
#                 is_xgboost=False,  # Serialized as unified sklearn pipeline
#                 name="pipeline"
#             )

#             pipelines[name] = model_pipeline

#     # -------------------------------------------------------------
#     # BEST MODEL SELECTION & ARTIFACT PERSISTENCE
#     # -------------------------------------------------------------
#     if task_type == "regression":
#         best_model_name = max(results, key=lambda x: results[x]["cv_mean_r2"])
#     else:
#         best_model_name = max(results, key=lambda x: results[x]["cv_mean_accuracy"])

#     best_pipeline = pipelines[best_model_name]

#     with mlflow.start_run(run_name=f"best_model_{dataset_id}"):
#         mlflow.log_param("best_model", best_model_name)
#         mlflow.log_param("dataset_id", dataset_id)
#         mlflow.log_param("task_type", task_type)

#         for metric_name, val in results[best_model_name].items():
#             mlflow.log_metric(metric_name, val)

#         log_mlflow_model(
#             model=best_pipeline,
#             is_xgboost=False,
#             name="best_pipeline"
#         )

#     # Save the complete pipeline (.pkl)
#     model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")
#     joblib.dump(best_pipeline, model_path)

#     training_time = time.time() - start_time

#     result = {
#         "dataset_id": dataset_id,
#         "target_column": target_column,
#         "task_type": task_type,
#         "feature_columns": feature_columns,
#         "model_scores": results,
#         "best_model": best_model_name,
#         "best_model_metrics": results[best_model_name],
#         "model_path": model_path,
#         "training_time_seconds": training_time,
#         "created_at": datetime.now(timezone.utc)
#     }

#     training_id = await save_training_result(result)
#     result.pop("_id", None)
#     result["training_id"] = training_id

#     return result
# import io
# import os
# import time
# import joblib
# import mlflow
# import mlflow.sklearn
# import mlflow.xgboost
# import pandas as pd
# import numpy as np

# from datetime import datetime, timezone
# from typing import Optional

# from fastapi import HTTPException, status

# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import (
#     cross_val_score,
#     train_test_split,
#     KFold,
#     StratifiedKFold
# )
# from sklearn.metrics import (
#     accuracy_score,
#     r2_score,
#     mean_absolute_error,
#     mean_squared_error
# )

# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     RandomForestRegressor
# )
# from sklearn.neighbors import KNeighborsClassifier

# from xgboost import (
#     XGBClassifier,
#     XGBRegressor
# )

# from app.repositories.training_repository import save_training_result

# # Directory configurations
# RAW_FOLDER = "datasets/raw"
# MODEL_FOLDER = "app/models/trained_models"

# os.makedirs(RAW_FOLDER, exist_ok=True)
# os.makedirs(MODEL_FOLDER, exist_ok=True)

# # Prefer SQLite over file store for stability
# tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# mlflow.set_tracking_uri(tracking_uri)
# mlflow.set_experiment("AutoMLOps")


# def log_mlflow_model(model, is_xgboost: bool, name: str = "model"):
#     """Helper to log models using cloudpickle to bypass untrusted numpy types."""
#     if is_xgboost:
#         mlflow.xgboost.log_model(
#             xgb_model=model,
#             artifact_path=name
#         )
#     else:
#         mlflow.sklearn.log_model(
#             sk_model=model,
#             artifact_path=name,
#             serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
#         )


# def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Detects and expands explicit date/time columns into numerical components."""
#     df = df.copy()
#     for col in list(df.columns):
#         if df[col].dtype == "object" and ("date" in col.lower() or "time" in col.lower()):
#             try:
#                 parsed_dates = pd.to_datetime(df[col], errors="coerce")
#                 if parsed_dates.notnull().sum() > 0:
#                     if parsed_dates.isnull().any():
#                         parsed_dates = parsed_dates.ffill().bfill()
                    
#                     df[f"{col}_year"] = parsed_dates.dt.year
#                     df[f"{col}_month"] = parsed_dates.dt.month
#                     df[f"{col}_day"] = parsed_dates.dt.day
#                     df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
#                     df = df.drop(columns=[col])
#             except (ValueError, TypeError):
#                 continue
#     return df


# def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
#     """Builds a leak-free ColumnTransformer for numeric and categorical features."""
#     numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
#     categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

#     numeric_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])

#     categorical_transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_features),
#             ("cat", categorical_transformer, categorical_features)
#         ]
#     )
#     return preprocessor


# async def train_models(
#     file_path: str,
#     dataset_id: str,
#     target_column: str
# ):
#     start_time = time.time()

#     df = pd.read_csv(file_path)
#     df.columns = df.columns.str.strip()
#     target_column = target_column.strip()

#     if target_column not in df.columns:
#         raise ValueError(f"Target column '{target_column}' not found in dataset.")

#     # 1. Clean Target Column: Drop rows where target (y) is NaN
#     df = df.dropna(subset=[target_column])

#     if df.empty:
#         raise ValueError(f"Target column '{target_column}' has no valid (non-null) samples.")

#     # 2. Extract Features and Target
#     y = df[target_column]
#     X_raw = df.drop(columns=[target_column])
#     X = extract_date_features(X_raw)
#     feature_columns = X.columns.tolist()

#     # 3. Determine Task Type
#     if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 2:
#         task_type = "classification"
#     elif y.nunique() < 10 and (y.dtype == "int64" or y.dtype == "int32"):
#         task_type = "classification"
#     else:
#         task_type = "regression"

#     # 4. Train-Test Split (Prior to any fitting to prevent data leakage)
#     stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=stratify
#     )

#     # 5. Cross-Validation Configuration
#     if task_type == "classification":
#         min_class_samples = y_train.value_counts().min()
#         n_splits = max(2, min(5, min_class_samples))
#         cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     else:
#         n_splits = max(2, min(5, len(X_train)))
#         cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     preprocessor = build_preprocessor(X_train)
#     results = {}
#     pipelines = {}

#     # -------------------------------------------------------------
#     # MODEL DEFINITIONS
#     # -------------------------------------------------------------
#     if task_type == "classification":
#         base_models = {
#             "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
#             "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
#             "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
#             "KNN": KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train) - 1))),
#             "XGBoost": XGBClassifier(
#                 max_depth=3,
#                 n_estimators=50,
#                 learning_rate=0.1,
#                 random_state=42,
#                 eval_metric="logloss"
#             )
#         }
#         scoring_metric = "accuracy"
#     else:
#         base_models = {
#             "LinearRegression": LinearRegression(),
#             "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
#             "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
#             "XGBoost": XGBRegressor(
#                 max_depth=3,
#                 n_estimators=50,
#                 learning_rate=0.1,
#                 random_state=42,
#                 objective="reg:squarederror"
#             )
#         }
#         scoring_metric = "r2"

#     # -------------------------------------------------------------
#     # TRAINING & CROSS-VALIDATION
#     # -------------------------------------------------------------
#     for name, model in base_models.items():
#         model_pipeline = Pipeline(steps=[
#             ("preprocessor", preprocessor),
#             ("model", model)
#         ])

#         with mlflow.start_run(run_name=name):
#             mlflow.log_param("model", name)
#             mlflow.log_param("task_type", task_type)
#             mlflow.log_param("dataset_id", dataset_id)
#             mlflow.log_param("target_column", target_column)
#             mlflow.log_param("rows", len(df))
#             mlflow.log_param("features", len(feature_columns))

#             # Cross-validation over train partition
#             cv_scores = cross_val_score(
#                 model_pipeline,
#                 X_train,
#                 y_train,
#                 cv=cv,
#                 scoring=scoring_metric
#             )
#             cv_mean = float(cv_scores.mean())
#             cv_std = float(cv_scores.std())

#             # Fit on full train partition and evaluate on test partition
#             model_pipeline.fit(X_train, y_train)
#             predictions = model_pipeline.predict(X_test)

#             if task_type == "classification":
#                 acc = float(accuracy_score(y_test, predictions))
#                 mlflow.log_metric("accuracy", acc)
#                 mlflow.log_metric("cv_mean_accuracy", cv_mean)
#                 mlflow.log_metric("cv_std_accuracy", cv_std)

#                 results[name] = {
#                     "accuracy": acc,
#                     "cv_mean_accuracy": cv_mean,
#                     "cv_std_accuracy": cv_std
#                 }
#             else:
#                 r2 = float(r2_score(y_test, predictions))
#                 mae = float(mean_absolute_error(y_test, predictions))
#                 rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

#                 mlflow.log_metric("r2_score", r2)
#                 mlflow.log_metric("mae", mae)
#                 mlflow.log_metric("rmse", rmse)
#                 mlflow.log_metric("cv_mean_r2", cv_mean)
#                 mlflow.log_metric("cv_std_r2", cv_std)

#                 results[name] = {
#                     "r2_score": r2,
#                     "mae": mae,
#                     "rmse": rmse,
#                     "cv_mean_r2": cv_mean,
#                     "cv_std_r2": cv_std
#                 }

#             log_mlflow_model(
#                 model=model_pipeline,
#                 is_xgboost=False,  # Serialized as unified sklearn pipeline
#                 name="pipeline"
#             )

#             pipelines[name] = model_pipeline

#     # -------------------------------------------------------------
#     # BEST MODEL SELECTION & ARTIFACT PERSISTENCE
#     # -------------------------------------------------------------
#     if task_type == "regression":
#         best_model_name = max(results, key=lambda x: results[x]["cv_mean_r2"])
#     else:
#         best_model_name = max(results, key=lambda x: results[x]["cv_mean_accuracy"])

#     best_pipeline = pipelines[best_model_name]

#     with mlflow.start_run(run_name=f"best_model_{dataset_id}"):
#         mlflow.log_param("best_model", best_model_name)
#         mlflow.log_param("dataset_id", dataset_id)
#         mlflow.log_param("task_type", task_type)

#         for metric_name, val in results[best_model_name].items():
#             mlflow.log_metric(metric_name, val)

#         log_mlflow_model(
#             model=best_pipeline,
#             is_xgboost=False,
#             name="best_pipeline"
#         )

#     # Save the complete pipeline (.pkl)
#     model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")
#     joblib.dump(best_pipeline, model_path)

#     training_time = time.time() - start_time

#     result = {
#         "dataset_id": dataset_id,
#         "target_column": target_column,
#         "task_type": task_type,
#         "feature_columns": feature_columns,
#         "model_scores": results,
#         "best_model": best_model_name,
#         "best_model_metrics": results[best_model_name],
#         "model_path": model_path,
#         "training_time_seconds": training_time,
#         "created_at": datetime.now(timezone.utc)
#     }

#     training_id = await save_training_result(result)
#     result.pop("_id", None)
#     result["training_id"] = training_id

#     return result


# # -------------------------------------------------------------
# # ACTION DISPATCHER / HANDLER FUNCTION
# # -------------------------------------------------------------
# async def handle_training_action(
#     action: str,
#     dataset_id: str,
#     target_column: Optional[str] = None
# ):
#     """Router dispatcher for training operations."""
#     if action == "model_training":
#         if not target_column:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Missing 'target_column' required for model training."
#             )

#         # Always train from the raw dataset since sklearn Pipeline manages transformations
#         raw_path = os.path.join(
#             RAW_FOLDER,
#             f"{dataset_id}.csv"
#         )

#         if not os.path.exists(raw_path):
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=(
#                     f"No raw dataset found for ID: {dataset_id}. "
#                     "Please upload the dataset first."
#                 )
#             )

#         return await train_models(
#             file_path=raw_path,
#             dataset_id=dataset_id,
#             target_column=target_column
#         )

#     raise HTTPException(
#         status_code=status.HTTP_400_BAD_REQUEST,
#         detail=f"Unknown action: '{action}'."
#     )
import io
import os
import time
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    KFold,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsClassifier

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

from app.repositories.training_repository import save_training_result

# Directory configurations
RAW_FOLDER = "datasets/raw"
MODEL_FOLDER = "app/models/trained_models"

os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Prefer SQLite over file store for stability
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("AutoMLOps")


# -------------------------------------------------------------
# 1. IN-PIPELINE DATE TRANSFORMER
# -------------------------------------------------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Detects and expands date/time columns into numerical features inside the pipeline."""
    def __init__(self):
        self.date_columns_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.date_columns_ = [
            col for col in X.columns
            if (X[col].dtype == "object" or X[col].dtype.name == "category")
            and ("date" in str(col).lower() or "time" in str(col).lower())
        ]
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df)

        for col in self.date_columns_:
            if col in X_df.columns:
                try:
                    parsed_dates = pd.to_datetime(X_df[col], errors="coerce")
                    if parsed_dates.notnull().sum() > 0:
                        if parsed_dates.isnull().any():
                            parsed_dates = parsed_dates.ffill().bfill()
                        
                        X_df[f"{col}_year"] = parsed_dates.dt.year
                        X_df[f"{col}_month"] = parsed_dates.dt.month
                        X_df[f"{col}_day"] = parsed_dates.dt.day
                        X_df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek
                        X_df = X_df.drop(columns=[col])
                except (ValueError, TypeError):
                    continue
        return X_df


def log_mlflow_model(model, is_xgboost: bool, name: str = "model"):
    """Helper to log models using cloudpickle serialization."""
    if is_xgboost:
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=name
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )


def build_pipeline_preprocessor(X_sample: pd.DataFrame) -> Pipeline:
    """Builds a complete, leak-free preprocessing pipeline including date extraction."""
    date_extractor = DateFeatureExtractor()
    X_transformed_sample = date_extractor.fit_transform(X_sample)

    numeric_features = X_transformed_sample.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    
    categorical_features = X_transformed_sample.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return Pipeline(steps=[
        ("date_extractor", date_extractor),
        ("columns", column_transformer)
    ])


# -------------------------------------------------------------
# 2. TRAINING SERVICE
# -------------------------------------------------------------
async def train_models(
    file_path: str,
    dataset_id: str,
    target_column: str
):
    start_time = time.time()

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    target_column = target_column.strip()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # 1. Drop rows where target is NaN
    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Target column '{target_column}' has no valid (non-null) samples.")

    # 2. Extract Features and Target (Keep X raw so pipeline transforms dates)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    feature_columns = X.columns.tolist()

    # 3. Determine Task Type
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 2:
        task_type = "classification"
    elif y.nunique() < 10 and (y.dtype == "int64" or y.dtype == "int32"):
        task_type = "classification"
    else:
        task_type = "regression"

    # 4. Train-Test Split (Isolated prior to fitting)
    stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    # 5. Cross-Validation Splitter Configuration
    if task_type == "classification":
        min_class_samples = y_train.value_counts().min()
        n_splits = max(2, min(5, min_class_samples))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = max(2, min(5, len(X_train)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    preprocessor = build_pipeline_preprocessor(X_train)
    results = {}
    pipelines = {}

    # -------------------------------------------------------------
    # MODEL DEFINITIONS
    # -------------------------------------------------------------
    if task_type == "classification":
        base_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
            "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_train) - 1))),
            "XGBoost": XGBClassifier(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            )
        }
        scoring_metric = "accuracy"
    else:
        base_models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
            "XGBoost": XGBRegressor(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1,
                random_state=42,
                objective="reg:squarederror"
            )
        }
        scoring_metric = "r2"

    # -------------------------------------------------------------
    # TRAINING & CROSS-VALIDATION
    # -------------------------------------------------------------
    for name, model in base_models.items():
        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("dataset_id", dataset_id)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("rows", len(df))
            mlflow.log_param("features", len(feature_columns))

            cv_scores = cross_val_score(
                model_pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring_metric
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())

            model_pipeline.fit(X_train, y_train)
            predictions = model_pipeline.predict(X_test)

            if task_type == "classification":
                acc = float(accuracy_score(y_test, predictions))
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("cv_mean_accuracy", cv_mean)
                mlflow.log_metric("cv_std_accuracy", cv_std)

                results[name] = {
                    "accuracy": acc,
                    "cv_mean_accuracy": cv_mean,
                    "cv_std_accuracy": cv_std
                }
            else:
                r2 = float(r2_score(y_test, predictions))
                mae = float(mean_absolute_error(y_test, predictions))
                rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("cv_mean_r2", cv_mean)
                mlflow.log_metric("cv_std_r2", cv_std)

                results[name] = {
                    "r2_score": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "cv_mean_r2": cv_mean,
                    "cv_std_r2": cv_std
                }

            log_mlflow_model(
                model=model_pipeline,
                is_xgboost=False,
                name="pipeline"
            )

            pipelines[name] = model_pipeline

    # -------------------------------------------------------------
    # BEST MODEL SELECTION & PERSISTENCE
    # -------------------------------------------------------------
    if task_type == "regression":
        best_model_name = max(results, key=lambda x: results[x]["cv_mean_r2"])
    else:
        best_model_name = max(results, key=lambda x: results[x]["cv_mean_accuracy"])

    best_pipeline = pipelines[best_model_name]

    with mlflow.start_run(run_name=f"best_model_{dataset_id}"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("task_type", task_type)

        for metric_name, val in results[best_model_name].items():
            mlflow.log_metric(metric_name, val)

        log_mlflow_model(
            model=best_pipeline,
            is_xgboost=False,
            name="best_pipeline"
        )

    # Save complete pipeline containing DateFeatureExtractor + ColumnTransformer + Model
    model_path = os.path.join(MODEL_FOLDER, f"{dataset_id}_best_model.pkl")
    joblib.dump(best_pipeline, model_path)

    training_time = time.time() - start_time

    result = {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "task_type": task_type,
        "feature_columns": feature_columns,
        "model_scores": results,
        "best_model": best_model_name,
        "best_model_metrics": results[best_model_name],
        "model_path": model_path,
        "training_time_seconds": training_time,
        "created_at": datetime.now(timezone.utc)
    }

    training_id = await save_training_result(result)
    result.pop("_id", None)
    result["training_id"] = training_id

    return result


# -------------------------------------------------------------
# ACTION DISPATCHER
# -------------------------------------------------------------
async def handle_training_action(
    action: str,
    dataset_id: str,
    target_column: Optional[str] = None
):
    if action == "model_training":
        if not target_column:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'target_column' required for model training."
            )

        raw_path = os.path.join(
            RAW_FOLDER,
            f"{dataset_id}.csv"
        )

        if not os.path.exists(raw_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No raw dataset found for ID: {dataset_id}. "
                    "Please upload the dataset first."
                )
            )

        return await train_models(
            file_path=raw_path,
            dataset_id=dataset_id,
            target_column=target_column
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown action: '{action}'."
    )