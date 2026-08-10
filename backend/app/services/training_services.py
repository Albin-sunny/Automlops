
# import os
# import time
# import joblib
# import mlflow
# import mlflow.sklearn
# import mlflow.xgboost
# import pandas as pd
# import numpy as np

# from datetime import datetime, UTC
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score, train_test_split
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
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("AutoMLOps")
# os.makedirs(
#     MODEL_FOLDER,
#     exist_ok=True
# )


# def log_mlflow_model(model, is_xgboost: bool, name: str = "model"):
#     """Helper to log models using the correct MLflow flavor and parameters."""
#     if is_xgboost:
#         mlflow.xgboost.log_model(
#             xgb_model=model,
#             name=name  # Replaces deprecated artifact_path
#         )
#     else:
#         mlflow.sklearn.log_model(
#             sk_model=model,
#             name=name  # Replaces deprecated artifact_path
#         )


# async def train_models(
#     file_path: str,
#     dataset_id: str,
#     target_column: str
# ):
#     start_time = time.time()

#     # Load dataset
#     df = pd.read_csv(file_path)

#     X = df.drop(columns=[target_column])
#     feature_columns = X.columns.tolist()
#     y = df[target_column]

#     # Detect task type
#     if y.dtype == "object" or y.nunique() <= 10:
#         task_type = "classification"
#     else:
#         task_type = "regression"

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42
#     )

#     # Safely compute CV folds based on train dataset size
#     cv_folds = min(5, max(2, len(X_train)))

#     results = {}

#     # -------------------------------------------------------------
#     # CLASSIFICATION PIPELINE
#     # -------------------------------------------------------------
#     if task_type == "classification":
#         models = {
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

#         for name, model in models.items():
#             with mlflow.start_run(run_name=name):
#                 mlflow.log_param("model", name)
#                 mlflow.log_param("task_type", task_type)
#                 mlflow.log_param("dataset_id", dataset_id)
#                 mlflow.log_param("target_column", target_column)
#                 mlflow.log_param("rows", len(df))
#                 mlflow.log_param("features", len(feature_columns))

#                 # Cross-validation evaluated ONLY on train set (prevents leakage)
#                 cv_scores = cross_val_score(
#                     model,
#                     X_train,
#                     y_train,
#                     cv=cv_folds,
#                     scoring="accuracy"
#                 )
#                 cv_mean = cv_scores.mean()
#                 cv_std = cv_scores.std()

#                 # Train model
#                 model.fit(X_train, y_train)

#                 prediction = model.predict(X_test)
#                 score = accuracy_score(y_test, prediction)

#                 mlflow.log_metric("accuracy", score)
#                 mlflow.log_metric("cv_mean_accuracy", cv_mean)
#                 mlflow.log_metric("cv_std_accuracy", cv_std)

#                 # Log model with correct flavor
#                 log_mlflow_model(
#                     model=model,
#                     is_xgboost=(name == "XGBoost"),
#                     name="model"
#                 )

#                 results[name] = {
#                     "accuracy": score,
#                     "cv_mean_accuracy": cv_mean,
#                     "cv_std_accuracy": cv_std
#                 }

#     # -------------------------------------------------------------
#     # REGRESSION PIPELINE
#     # -------------------------------------------------------------
#     else:
#         models = {
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

#         for name, model in models.items():
#             with mlflow.start_run(run_name=name):
#                 mlflow.log_param("model", name)
#                 mlflow.log_param("task_type", task_type)
#                 mlflow.log_param("dataset_id", dataset_id)
#                 mlflow.log_param("target_column", target_column)
#                 mlflow.log_param("rows", len(df))
#                 mlflow.log_param("features", len(feature_columns))

#                 # Cross-validation evaluated ONLY on train set (prevents leakage)
#                 cv_scores = cross_val_score(
#                     model,
#                     X_train,
#                     y_train,
#                     cv=cv_folds,
#                     scoring="r2"
#                 )
#                 cv_mean = cv_scores.mean()
#                 cv_std = cv_scores.std()

#                 # Train model
#                 model.fit(X_train, y_train)

#                 prediction = model.predict(X_test)
#                 r2 = r2_score(y_test, prediction)
#                 mae = mean_absolute_error(y_test, prediction)
#                 rmse = np.sqrt(mean_squared_error(y_test, prediction))

#                 mlflow.log_metric("r2_score", r2)
#                 mlflow.log_metric("mae", mae)
#                 mlflow.log_metric("rmse", rmse)
#                 mlflow.log_metric("cv_mean_r2", cv_mean)
#                 mlflow.log_metric("cv_std_r2", cv_std)

#                 # Log model with correct flavor
#                 log_mlflow_model(
#                     model=model,
#                     is_xgboost=(name == "XGBoost"),
#                     name="model"
#                 )

#                 results[name] = {
#                     "r2_score": r2,
#                     "mae": mae,
#                     "rmse": rmse,
#                     "cv_mean_r2": cv_mean,
#                     "cv_std_r2": cv_std
#                 }

#     # -------------------------------------------------------------
#     # SELECT & LOG BEST MODEL (Selected via CV Mean Score)
#     # -------------------------------------------------------------
#     if task_type == "regression":
#         best_model_name = max(results, key=lambda x: results[x]["cv_mean_r2"])
#     else:
#         best_model_name = max(results, key=lambda x: results[x]["cv_mean_accuracy"])

#     best_model = models[best_model_name]

#     with mlflow.start_run(run_name=f"best_model_{dataset_id}"):
#         mlflow.log_param("best_model", best_model_name)
#         mlflow.log_param("dataset_id", dataset_id)
#         mlflow.log_param("task_type", task_type)

#         if task_type == "classification":
#             mlflow.log_metric(
#                 "accuracy",
#                 results[best_model_name]["accuracy"]
#             )
#             mlflow.log_metric(
#                 "cv_mean_accuracy",
#                 results[best_model_name]["cv_mean_accuracy"]
#             )
#         else:
#             mlflow.log_metric(
#                 "r2_score",
#                 results[best_model_name]["r2_score"]
#             )
#             mlflow.log_metric(
#                 "cv_mean_r2",
#                 results[best_model_name]["cv_mean_r2"]
#             )
#             mlflow.log_metric(
#                 "mae",
#                 results[best_model_name]["mae"]
#             )
#             mlflow.log_metric(
#                 "rmse",
#                 results[best_model_name]["rmse"]
#             )

#         log_mlflow_model(
#             model=best_model,
#             is_xgboost=(best_model_name == "XGBoost"),
#             name="best_model"
#         )

#     # Save Scikit-Learn Pipeline locally
#     pipeline = Pipeline([("model", best_model)])
#     pipeline.fit(X_train, y_train)

#     model_path = os.path.join(
#         MODEL_FOLDER,
#         f"{dataset_id}_best_model.pkl"
#     )
#     joblib.dump(pipeline, model_path)

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
#         "created_at": datetime.now(UTC)
#     }

#     training_id = await save_training_result(result)
#     result.pop("_id", None)
#     result["training_id"] = training_id

#     return result



import os
import time
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np

from datetime import datetime, UTC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
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


MODEL_FOLDER = "app/models/trained_models"

# Use MLFLOW_TRACKING_URI if set; default to local file storage to avoid CI connection errors
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("AutoMLOps")

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


def log_mlflow_model(model, is_xgboost: bool, name: str = "model"):
    """Helper to log models using the correct MLflow flavor and parameters."""
    if is_xgboost:
        mlflow.xgboost.log_model(
            xgb_model=model,
            name=name  # Replaces deprecated artifact_path
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            name=name  # Replaces deprecated artifact_path
        )


async def train_models(
    file_path: str,
    dataset_id: str,
    target_column: str
):
    start_time = time.time()

    # Load dataset
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_column])
    feature_columns = X.columns.tolist()
    y = df[target_column]

    # Detect task type
    if y.dtype == "object" or y.nunique() <= 10:
        task_type = "classification"
    else:
        task_type = "regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Safely compute CV folds based on train dataset size
    cv_folds = min(5, max(2, len(X_train)))

    results = {}

    # -------------------------------------------------------------
    # CLASSIFICATION PIPELINE
    # -------------------------------------------------------------
    if task_type == "classification":
        models = {
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

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("dataset_id", dataset_id)
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("rows", len(df))
                mlflow.log_param("features", len(feature_columns))

                # Cross-validation evaluated ONLY on train set (prevents leakage)
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring="accuracy"
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # Train model
                model.fit(X_train, y_train)

                prediction = model.predict(X_test)
                score = accuracy_score(y_test, prediction)

                mlflow.log_metric("accuracy", score)
                mlflow.log_metric("cv_mean_accuracy", cv_mean)
                mlflow.log_metric("cv_std_accuracy", cv_std)

                # Log model with correct flavor
                log_mlflow_model(
                    model=model,
                    is_xgboost=(name == "XGBoost"),
                    name="model"
                )

                results[name] = {
                    "accuracy": score,
                    "cv_mean_accuracy": cv_mean,
                    "cv_std_accuracy": cv_std
                }

    # -------------------------------------------------------------
    # REGRESSION PIPELINE
    # -------------------------------------------------------------
    else:
        models = {
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

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("dataset_id", dataset_id)
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("rows", len(df))
                mlflow.log_param("features", len(feature_columns))

                # Cross-validation evaluated ONLY on train set (prevents leakage)
                cv_scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv_folds,
                    scoring="r2"
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # Train model
                model.fit(X_train, y_train)

                prediction = model.predict(X_test)
                r2 = r2_score(y_test, prediction)
                mae = mean_absolute_error(y_test, prediction)
                rmse = np.sqrt(mean_squared_error(y_test, prediction))

                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("cv_mean_r2", cv_mean)
                mlflow.log_metric("cv_std_r2", cv_std)

                # Log model with correct flavor
                log_mlflow_model(
                    model=model,
                    is_xgboost=(name == "XGBoost"),
                    name="model"
                )

                results[name] = {
                    "r2_score": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "cv_mean_r2": cv_mean,
                    "cv_std_r2": cv_std
                }

    # -------------------------------------------------------------
    # SELECT & LOG BEST MODEL (Selected via CV Mean Score)
    # -------------------------------------------------------------
    if task_type == "regression":
        best_model_name = max(results, key=lambda x: results[x]["cv_mean_r2"])
    else:
        best_model_name = max(results, key=lambda x: results[x]["cv_mean_accuracy"])

    best_model = models[best_model_name]

    with mlflow.start_run(run_name=f"best_model_{dataset_id}"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("dataset_id", dataset_id)
        mlflow.log_param("task_type", task_type)

        if task_type == "classification":
            mlflow.log_metric(
                "accuracy",
                results[best_model_name]["accuracy"]
            )
            mlflow.log_metric(
                "cv_mean_accuracy",
                results[best_model_name]["cv_mean_accuracy"]
            )
        else:
            mlflow.log_metric(
                "r2_score",
                results[best_model_name]["r2_score"]
            )
            mlflow.log_metric(
                "cv_mean_r2",
                results[best_model_name]["cv_mean_r2"]
            )
            mlflow.log_metric(
                "mae",
                results[best_model_name]["mae"]
            )
            mlflow.log_metric(
                "rmse",
                results[best_model_name]["rmse"]
            )

        log_mlflow_model(
            model=best_model,
            is_xgboost=(best_model_name == "XGBoost"),
            name="best_model"
        )

    # Save Scikit-Learn Pipeline locally
    pipeline = Pipeline([("model", best_model)])
    pipeline.fit(X_train, y_train)

    model_path = os.path.join(
        MODEL_FOLDER,
        f"{dataset_id}_best_model.pkl"
    )
    joblib.dump(pipeline, model_path)

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
        "created_at": datetime.now(UTC)
    }

    training_id = await save_training_result(result)
    result.pop("_id", None)
    result["training_id"] = training_id

    return result