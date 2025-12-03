# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.model import create_model
from src.config import load_params

def load_and_prepare(path):
    df = pd.read_csv(path)

    candidate_cols = [
        "sqft", "beds", "baths", "stories", "status",
        "propertyType", "city", "state", "fireplace",
        "schools", "zipcode"
    ]

    features = [c for c in candidate_cols if c in df.columns]
    if len(features) == 0:
        raise ValueError("Aucune colonne feature trouvée dans processed_data.csv")

    X = df[features]
    y = df["target"]
    return X, y


def train(save_path=None):
    params = load_params()
    paths = params["paths"]

    data_path = paths["raw_data"]      # OK
    model_out = save_path or paths["model_out"]

    test_size = params["model"]["test_size"]
    random_state = params["model"]["random_state"]

    X, y = load_and_prepare(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = create_model()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"✔ Modèle entraîné et sauvegardé dans : {model_out}")

    # Save test data
    os.makedirs("test_data", exist_ok=True)
    X_test.to_csv("test_data/X_test.csv", index=False)
    y_test.to_csv("test_data/y_test.csv", index=False)
    print("✔ Test set sauvegardé dans test_data/")

    return pipeline, (X_test, y_test)


if __name__ == "__main__":
    train()
