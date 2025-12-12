import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.model import create_model
from src.config import load_params


def load_and_prepare(path):
    df = pd.read_csv(path)

    feature_cols = [
        "sqft", "beds", "baths", "stories",
        "status", "propertyType", "city", "state",
        "fireplace", "schools", "zipcode"
    ]

    X = df[feature_cols]
    y = df["target"]

    return X, y


def train(save_path=None):
    params = load_params()
    paths = params["paths"]

    data_path = paths["raw_data"]
    model_out = save_path or paths["model_out"]

    test_size = params["model"]["test_size"]
    random_state = params["model"]["random_state"]

    # Charger données
    X, y = load_and_prepare(data_path)
    
    # Vérifier les NaN avant le split
    print("Vérification des NaN dans X:")
    print(X.isna().sum())
    
    # S'assurer qu'il n'y a pas de NaN
    X = X.fillna({
        'sqft': X['sqft'].mean(),
        'beds': X['beds'].median(),
        'baths': X['baths'].median(),
        'stories': X['stories'].median(),
        'fireplace': 'Unknown',
        'schools': 'Unknown',
        'zipcode': 'Unknown',
        'status': 'Unknown',
        'propertyType': 'Unknown',
        'city': 'Unknown',
        'state': 'Unknown'
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # colonnes
    numeric_features = ["sqft", "beds", "baths", "stories"]
    categorical_features = ["status", "propertyType", "city", "state",
                            "fireplace", "schools", "zipcode"]

    # Préprocesseur complet avec imputation
    from sklearn.impute import SimpleImputer
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )

    # Modèle final
    model = create_model()

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Test du pipeline sur un exemple
    test_sample = X_test.iloc[0:1].copy()
    print("\nTest du pipeline sur un exemple:")
    print("Input:", test_sample.values)
    try:
        prediction = pipeline.predict(test_sample)
        print("Prédiction réussie:", prediction)
    except Exception as e:
        print("Erreur pendant le test:", str(e))
    
    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f"✔ Modèle entraîné et sauvegardé dans : {model_out}")

    # Sauvegarde test set
    os.makedirs("test_data", exist_ok=True)
    X_test.to_csv("test_data/X_test.csv", index=False)
    y_test.to_csv("test_data/y_test.csv", index=False)

    print("✔ Jeu de test sauvegardé dans test_data/")

    return pipeline, (X_test, y_test)


if __name__ == "__main__":
    train()