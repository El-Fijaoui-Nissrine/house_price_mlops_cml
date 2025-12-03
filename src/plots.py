# src/plots.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import load_params
import json
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")

def plot_pred_vs_true():
    params = load_params()
    metrics_dir = "metrics"
    predictions_csv = os.path.join(metrics_dir, "predictions.csv")
    out_path = params["paths"]["plot_pred"]

    if not os.path.exists(predictions_csv):
        raise FileNotFoundError("predictions.csv introuvable. Exécute evaluate.py d'abord.")

    df = pd.read_csv(predictions_csv)

    # Calcul métriques
    rmse = sqrt(mean_squared_error(df["y_true"], df["y_pred"]))
    r2 = r2_score(df["y_true"], df["y_pred"])

    plt.figure(figsize=(7,7))
    # Scatter simple, plus sûr
    plt.scatter(df["y_true"], df["y_pred"], alpha=0.3, s=10, color="blue")
    plt.xlabel("Prix réels")
    plt.ylabel("Prix prédits")
    plt.title("Vrais vs Prédits")

    minv = min(df["y_true"].min(), df["y_pred"].min())
    maxv = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--", color="red")

    plt.text(minv, maxv*0.9, f"RMSE={rmse:,.0f}\nR²={r2:.2f}", fontsize=12, color="black",
             bbox=dict(facecolor="white", alpha=0.5))

    os.makedirs(metrics_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("✔ Plot pred_vs_true sauvegardé dans :", out_path)
    return out_path


def plot_residuals():
    params = load_params()
    metrics_dir = "metrics"
    predictions_csv = os.path.join(metrics_dir, "predictions.csv")
    out_path = params["paths"]["plot_resid"]

    if not os.path.exists(predictions_csv):
        raise FileNotFoundError("predictions.csv introuvable. Exécute evaluate.py d'abord.")

    df = pd.read_csv(predictions_csv)
    df["residual"] = df["y_true"] - df["y_pred"]

    plt.figure(figsize=(8,4))
    sns.histplot(df["residual"], bins=80, kde=True, color="skyblue")
    plt.axvline(df["residual"].mean(), color="red", linestyle="--", label=f"Moyenne={df['residual'].mean():,.0f}")
    plt.xlabel("Résiduel (y_true - y_pred)")
    plt.ylabel("Nombre de maisons")
    plt.title("Distribution des résiduels")
    plt.legend()

    os.makedirs(metrics_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("✔ Plot residuals sauvegardé dans :", out_path)
    return out_path

if __name__ == "__main__":
    plot_pred_vs_true()
    plot_residuals()
