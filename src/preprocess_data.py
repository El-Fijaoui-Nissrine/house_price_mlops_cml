import pandas as pd
import numpy as np
import os
import json
import ast
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

def extract_school_info(schools_str):
    """
    Extrait une métrique simple de la colonne schools.
    Retourne le nombre d'écoles ou une valeur par défaut.
    """
    if pd.isna(schools_str) or schools_str == "Unknown" or schools_str == "":
        return "0"
    
    try:
        # Essayer de parser comme JSON
        if isinstance(schools_str, str):
            # Nettoyer la chaîne
            schools_str = schools_str.replace("'", '"')
            
            # Extraire le nombre d'écoles basé sur les patterns
            if 'name' in schools_str:
                # Compter le nombre d'écoles dans le champ name
                if '[]' in schools_str or "name': []" in schools_str:
                    return "0"
                # Compter les apostrophes simples comme approximation
                name_count = schools_str.count("'name': [")
                if name_count > 0:
                    # Extraire la liste des noms
                    start = schools_str.find("'name': [") + 9
                    end = schools_str.find("]", start)
                    if start > 8 and end > start:
                        names_str = schools_str[start:end]
                        # Compter les noms approximativement
                        school_count = names_str.count("'") // 2
                        return str(min(school_count, 10))  # Limiter à 10
            return "0"
    except:
        pass
    
    return "0"

# ----------------------------
# Fonction principale
# ----------------------------

def preprocess_data(input_path="data/house_price.csv",
                    output_path="data/processed/processed_data.csv"):

    # Vérifier si le fichier de sortie est en cours d'utilisation
    if os.path.exists(output_path):
        try:
            # Essayer de l'ouvrir en mode écriture pour vérifier
            with open(output_path, 'a') as f:
                pass
        except PermissionError:
            print(f"❌ Erreur: Le fichier {output_path} est utilisé par un autre programme.")
            print("Veuillez fermer Excel, Notepad++, VS Code, ou tout autre programme utilisant ce fichier.")
            return None

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
    # 3) Simplifier la colonne schools
    # -------------------------------
    if "schools" in df.columns:
        print("Traitement de la colonne 'schools'...")
        df["schools"] = df["schools"].apply(extract_school_info)
        print("Colonne 'schools' simplifiée. Valeurs uniques:", df["schools"].unique()[:10])

    # -------------------------------
    # 4) Nettoyer les colonnes catégorielles
    # -------------------------------
    cat_cols = ["status", "propertyType", "city", "state", "fireplace", "zipcode"]
    
    for col in cat_cols:
        if col in df.columns:
            # Convertir en string
            df[col] = df[col].astype(str)
            # Nettoyer les valeurs
            df[col] = df[col].str.strip()
            # Standardiser les valeurs
            df[col] = df[col].replace(['nan', 'NaN', 'None', 'none', 'null', 'Null', 'NULL', '', 'Not Applicable', 'N/A'], 'Unknown')
    
    # Standardiser les valeurs spécifiques
    if "status" in df.columns:
        df["status"] = df["status"].replace(['Active', 'For sale', 'P'], 'for sale')
        df["status"] = df["status"].replace(['Active/Contingent', 'Pending', 'Under Contract'], 'pending')
    
    if "propertyType" in df.columns:
        df["propertyType"] = df["propertyType"].replace(['Single Family Home', 'single-family home', 'Single Family'], 'Single Family')
        df["propertyType"] = df["propertyType"].replace(['townhouse', 'Townhouse'], 'Townhouse')
        df["propertyType"] = df["propertyType"].replace(['coop', 'multi-family'], 'Multi Family')
    
    if "fireplace" in df.columns:
        df["fireplace"] = df["fireplace"].replace(['Gas Logs', 'yes', 'Yes', 'Fireplace YN'], 'Yes')
        df["fireplace"] = df["fireplace"].replace(['0', 'Not Applicable'], 'No')
        # Garder les nombres comme catégories
        df["fireplace"] = df["fireplace"].apply(lambda x: str(x) if str(x).isdigit() else x)

    # -------------------------------
    # 5) Supprimer lignes sans target ou sqft
    # -------------------------------
    df = df.dropna(subset=["target", "sqft"])
    
    # Remplir autres NaN numériques par la médiane
    numeric_cols = ["beds", "baths", "stories"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # S'assurer qu'il n'y a plus de NaN
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
    
    # -------------------------------
    # 6) Vérification finale
    # -------------------------------
    print("\n" + "="*50)
    print("VÉRIFICATION FINALE DES DONNÉES")
    print("="*50)
    
    print("\n1. Valeurs manquantes:")
    print(df.isna().sum())
    
    print("\n2. Types de données:")
    print(df.dtypes)
    
    print("\n3. Statistiques des colonnes numériques:")
    numeric_cols_all = ["sqft", "beds", "baths", "stories", "target"]
    for col in numeric_cols_all:
        if col in df.columns:
            print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    print("\n4. Valeurs uniques dans les colonnes catégorielles (premières 5):")
    for col in cat_cols + ["schools"]:
        if col in df.columns:
            print(f"{col}: {df[col].unique()[:5]}")
    
    print("\n5. Dimensions du dataset:")
    print(f"Lignes: {len(df)}, Colonnes: {len(df.columns)}")

    # -------------------------------
    # 7) Sauvegarder dataset prétraité
    # -------------------------------
    folder = os.path.dirname(output_path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    
    try:
        # Sauvegarder dans un fichier temporaire d'abord
        temp_path = output_path + ".temp"
        df.to_csv(temp_path, index=False)
        
        # Remplacer l'ancien fichier
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
        print("\n" + "="*50)
        print(f"✔ Prétraitement terminé avec succès!")
        print(f"Données sauvegardées dans : {output_path}")
        print(f"Taille du fichier: {os.path.getsize(output_path) / 1024:.2f} KB")
        print("="*50)
        
        return df
        
    except PermissionError:
        print(f"\n❌ ERREUR: Impossible de sauvegarder le fichier {output_path}")
        print("Le fichier est probablement ouvert par un autre programme.")
        print("Veuillez fermer:")
        print("  - Excel")
        print("  - Notepad++ ou autre éditeur de texte")
        print("  - VS Code (fermez l'onglet du fichier)")
        print("  - Tout autre programme utilisant ce fichier")
        
        # Sauvegarder dans un fichier alternatif
        alt_path = "data/processed/processed_data_backup.csv"
        df.to_csv(alt_path, index=False)
        print(f"\n⚠️ Fichier sauvegardé temporairement dans: {alt_path}")
        
        return df
    except Exception as e:
        print(f"\n❌ Erreur lors de la sauvegarde: {str(e)}")
        return None

# ----------------------------
# Script de nettoyage des permissions
# ----------------------------
def cleanup_locked_files():
    """Essaie de libérer les fichiers verrouillés"""
    import subprocess
    
    print("Tentative de libération des fichiers verrouillés...")
    
    try:
        # Pour Windows
        if os.name == 'nt':
            # Essayer de tuer les processus qui pourraient bloquer le fichier
            pass
        # Pour Linux/Mac
        else:
            # Utiliser lsof pour trouver les processus
            result = subprocess.run(['lsof', '+D', 'data/processed/'], 
                                   capture_output=True, text=True)
            if result.stdout:
                print("Processus utilisant les fichiers:")
                print(result.stdout)
    except:
        pass

# ----------------------------
# Execution directe
# ----------------------------
if __name__ == "__main__":
    # Essayer de nettoyer d'abord
    cleanup_locked_files()
    
    # Exécuter le prétraitement
    df = preprocess_data("data/house_price.csv", "data/processed/processed_data.csv")
    
    if df is not None:
        # Afficher un aperçu
        print("\n" + "="*50)
        print("APERÇU DES DONNÉES TRAITÉES")
        print("="*50)
        print(df.head())
        print("\nColonnes finales:", list(df.columns))