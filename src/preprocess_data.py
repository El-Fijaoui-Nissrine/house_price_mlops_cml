import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Fonctions utilitaires
# ----------------------------

def clean_sqft(value):
    """Convertir sqft comme '1,947 sqft' en float"""
    if pd.isna(value):
        return np.nan
    try:
        value = str(value).replace(",", "").split()[0]
        return float(value)
    except:
        return np.nan

def clean_numeric(value):
    """Extraire le nombre de baths, beds, stories"""
    if pd.isna(value):
        return np.nan
    try:
        # garder seulement le chiffre
        return float(str(value).split()[0])
    except:
        return np.nan

def clean_price(value):
    """Convertir target '$418,000' en float"""
    if pd.isna(value):
        return np.nan
    try:
        value = str(value).replace("$", "").replace(",", "").strip()
        return float(value)
    except:
        return np.nan

# ----------------------------
# Fonction principale
# ----------------------------

def preprocess_data(input_path="data/house_price.csv",
                    output_path="data/processed/processed_data.csv"):

    df = pd.read_csv(input_path)

    # -------------------------------
    # 1) Supprimer colonnes inutiles
    # -------------------------------
    drop_cols = ["mls-id", "MlsId", "PrivatePool", "private pool", "homeFacts", "street"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # -------------------------------
    # 2) Nettoyer colonnes numériques
    # -------------------------------
    df["sqft"] = df["sqft"].apply(clean_sqft)
    df["beds"] = df["beds"].apply(clean_numeric)
    df["baths"] = df["baths"].apply(clean_numeric)
    df["stories"] = df["stories"].apply(clean_numeric)
    df["target"] = df["target"].apply(clean_price)

    # -------------------------------
    # 3) Supprimer lignes sans target ou sqft
    # -------------------------------
    df = df.dropna(subset=["target", "sqft"])

    # Remplir autres NaN numériques par la médiane
    numeric_cols = ["beds", "baths", "stories"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Remplir colonnes catégorielles manquantes par 'Unknown'
    cat_cols = ["status", "propertyType", "city", "state", "fireplace","schools","zipcode" ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # -------------------------------
    # 4) Encoder colonnes catégorielles
    # -------------------------------
    for col in cat_cols:
        if col in df.columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    # -------------------------------
    # 5) Sauvegarder dataset prétraité
    # -------------------------------
    folder = os.path.dirname(output_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    df.to_csv(output_path, index=False)
    print("✔ Prétraitement terminé. Données sauvegardées dans :", output_path)

    return df

# ----------------------------
# Execution directe
# ----------------------------
if __name__ == "__main__":
    preprocess_data("data/house_price.csv", "data/processed/processed_data.csv")
