import os
import time
import joblib
import pandas as pd

from datetime import datetime, UTC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import numpy as np

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

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


async def train_models(
    file_path: str,
    dataset_id: str,
    target_column: str
):

    start_time = time.time()

    # Load dataset
    df = pd.read_csv(file_path)


    X = df.drop(
        columns=[target_column]
    )

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


    results = {}


    if task_type == "classification":

        models = {

            "LogisticRegression":
                LogisticRegression(
                    max_iter=1000
                ),

            "DecisionTree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "RandomForest":
                RandomForestClassifier(
                    random_state=42
                ),

            "KNN":
                KNeighborsClassifier(),

            "XGBoost":
                XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                )
        }


        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv= min(5,len(X)),
                scoring="accuracy"
            )

            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            

            prediction = model.predict(
                X_test
            )

            score = accuracy_score(
                y_test,
                prediction
            )

            results[name] = {
                "accuracy": score,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std
            }


    else:

        models = {

            "LinearRegression":
                LinearRegression(),

            "DecisionTree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "RandomForest":
                RandomForestRegressor(
                random_state=42
                ),

            "XGBoost":
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror"
                )

        }


        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=min(5,len(X)),
                scoring="r2"
            )

            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            prediction = model.predict(
                X_test
            )

            r2 = r2_score(
                y_test,
                prediction
            )

            mae = mean_absolute_error(
                y_test,
                prediction
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    prediction
                )
            )


            results[name] = {
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse,
                "cv_mean_r2": cv_mean,
                "cv_std_r2": cv_std
            }



    # Select best model

    if task_type == "regression":

        best_model_name = max(
            results,
            key=lambda x: results[x]["r2_score"]
        )

    else:

        best_model_name = max(
            results,
            key=lambda x: results[x]["accuracy"]
        )


    best_model = models[best_model_name]


    pipeline = Pipeline(
        [
            ("model", best_model)
        ]
    )


    model_path = os.path.join(
        MODEL_FOLDER,
        f"{dataset_id}_best_model.pkl"
    )


    joblib.dump(
        pipeline,
        model_path
    )


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


    training_id = await save_training_result(
        result
    )


    result.pop("_id", None)

    result["training_id"] = training_id


    return result