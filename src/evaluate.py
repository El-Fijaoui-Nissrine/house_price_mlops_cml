# src/evaluate.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.config import load_params
from sklearn.model_selection import train_test_split


def load_testset():
    params = load_params()
    data_path = params["paths"]["raw_data"]

    # Check if test_data exists
    X_test_path = "test_data/X_test.csv"
    y_test_path = "test_data/y_test.csv"

    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()
        return X_test, y_test

    # Fallback : re-split dataset
    df = pd.read_csv(data_path)

    candidate_cols = [
        "sqft", "beds", "baths", "stories", "status",
        "propertyType", "city", "state", "fireplace",
        "schools", "zipcode"
    ]
    features = [c for c in candidate_cols if c in df.columns]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["random_state"]
    )

    return X_test, y_test


def evaluate():
    params = load_params()
    model_path = params["paths"]["model_out"]
    metrics_path = params["paths"]["metrics_out"]

    # Load model
    pipeline = joblib.load(model_path)

    # Test set
    X_test, y_test = load_testset()

    preds = pipeline.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "mape_percent": float(mape)
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Save predictions CSV
    preds_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": preds
    })
    preds_df.to_csv("metrics/predictions.csv", index=False)

    print("✔ Metrics sauvegardées dans :", metrics_path)
    print(metrics)

    return metrics


if __name__ == "__main__":
    evaluate()
