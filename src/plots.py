import matplotlib.pyplot as plt
import pandas as pd
import joblib
import json
from cml import publish

def plot_pred_vs_true(data_path="metrics/predictions.csv"):
    df = pd.read_csv(data_path)
    plt.figure()
    plt.scatter(df["true"], df["pred"])
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    plt.title("Vrais vs Prédits")
    out = "metrics/pred_vs_true.png"
    plt.savefig(out)
    plt.close()
    return publish(out, "pred_vs_true.png")


def plot_residuals(data_path="metrics/predictions.csv"):
    df = pd.read_csv(data_path)
    residuals = df["true"] - df["pred"]
    plt.figure()
    plt.scatter(df["pred"], residuals)
    plt.xlabel("Prédictions")
    plt.ylabel("Résiduels")
    plt.title("Résiduels vs prédictions")
    out = "metrics/residuals.png"
    plt.savefig(out)
    plt.close()
    return publish(out, "residuals.png")


def main():
    print("📌 Génération des plots...")
    pred_url = plot_pred_vs_true()
    resid_url = plot_residuals()

    # Sauvegarder les URL dans un fichier JSON pour le compare script
    urls = {
        "pred_vs_true": pred_url,
        "residuals": resid_url
    }

    with open("metrics/plot_urls.json", "w") as f:
        json.dump(urls, f, indent=4)

    print("✔ Plots générés et publiés !")


if __name__ == "__main__":
    main()
