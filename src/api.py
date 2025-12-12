from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any

# Chargement du modèle
MODEL_PATH = "models/model.joblib"
pipeline = None

def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print(f"✔ Modèle chargé depuis {MODEL_PATH}")
    else:
        print(f"⚠️ Modèle non trouvé : {MODEL_PATH}")

def extract_metrics_from_report(content: str) -> str:
    """Extrait seulement les sections métriques du rapport, excluant les plots"""
    lines = content.split('\n')
    filtered_lines = []
    skip_plots = False
    
    for i, line in enumerate(lines):
        # Si on trouve la section des plots, on arrête
        if "## 📊 Comparaison des plots" in line:
            skip_plots = True
            continue
        
        # Si on trouve la prochaine section après les plots, on reprend
        if skip_plots and line.startswith('## '):
            skip_plots = False
        
        # Si on est en train de sauter les plots, continuer
        if skip_plots:
            continue
        
        # Ajouter les lignes qui ne sont pas des plots
        if not skip_plots:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

# Charger au démarrage avec lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown (si nécessaire)
    pass

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle au démarrage déjà géré par lifespan

# ===== MODELS =====
class PredictionInput(BaseModel):
    sqft: float
    beds: int
    baths: float
    stories: int
    status: str = "for sale"
    propertyType: str = "Single Family"
    city: str = "New York"
    state: str = "NY"
    fireplace: int = 0
    schools: int = 0
    zipcode: int = 10001

class PredictionResponse(BaseModel):
    prediction: float
    input_data: Dict[str, Any]

class MetricsResponse(BaseModel):
    rmse: float
    r2: float
    mae: float
    mape_percent: float

# ===== ENDPOINTS =====

@app.get("/")
def read_root():
    return {
        "message": "House Price Prediction API",
        "status": "running",
        "model_loaded": pipeline is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    """Prédit le prix d'une maison"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Conversion en DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        print("Input DataFrame:\n", df)
        df["sqft"] = df["sqft"].astype(float)
        df["beds"] = df["beds"].astype(int)
        df["baths"] = df["baths"].astype(float)
        df["stories"] = df["stories"].astype(int)
        df["fireplace"] = df["fireplace"].astype(str)
        df["schools"] = df["schools"].astype(str)
        df["zipcode"] = df["zipcode"].astype(str)
        df["status"] = df["status"].astype(str)
        df["propertyType"] = df["propertyType"].astype(str)
        df["city"] = df["city"].astype(str)
        df["state"] = df["state"].astype(str)
        # Prédiction
        prediction = pipeline.predict(df)[0]
        print("Prediction:", prediction)
        return PredictionResponse(
            prediction=float(prediction),
            input_data=input_dict
        )
    except Exception as e:
        print("Erreur de prédiction:", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Prédictions sur un fichier CSV"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Lecture CSV
        df = pd.read_csv(file.file)
        
        # Prédictions
        predictions = pipeline.predict(df)
        
        # Ajout des prédictions au DataFrame
        df["prediction"] = predictions
        
        # Sauvegarde temporaire
        output_path = "temp_predictions.csv"
        df.to_csv(output_path, index=False)
        
        return FileResponse(
            output_path,
            media_type="text/csv",
            filename="predictions.csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Récupère les métriques actuelles"""
    metrics_path = "metrics/metrics.json"
    
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Métriques non trouvées")
    
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return MetricsResponse(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/comparison")
def get_comparison():
    """Récupère le rapport de comparaison (métriques seulement)"""
    report_path = "reports/comparison_report.md"
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Rapport non trouvé")
    
    try:
        with open(report_path, "r") as f:
            content = f.read()
        
        # Extraire seulement les métriques, exclure les plots
        metrics_only = extract_metrics_from_report(content)
        
        return {"content": metrics_only}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/comparison_full")
def get_comparison_full():
    """Récupère le rapport de comparaison complet (avec plots)"""
    report_path = "reports/comparison_report.md"
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Rapport non trouvé")
    
    try:
        with open(report_path, "r") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/plots/{plot_name}")
def get_plot(plot_name: str):
    """Récupère une image de plot"""
    plot_path = f"metrics/{plot_name}"
    
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot non trouvé")
    
    return FileResponse(plot_path, media_type="image/png")

@app.post("/reload_model")
def reload_model():
    """Recharge le modèle"""
    try:
        load_model()
        return {"message": "Modèle rechargé avec succès", "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)