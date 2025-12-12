# src/compare.py
import json
import os

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_links():
    """Charge les URLs CML des images publiées"""
    path = "reports/links.env"
    if not os.path.exists(path):
        return {}
    links = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                links[k] = v
    return links

def get_comparison_emoji_and_color(key, diff):
    """
    Retourne emoji et couleur selon la métrique.
    Pour loss-like metrics (RMSE, MAE, MAPE): plus bas = meilleur (vert)
    Pour quality metrics (R²): plus haut = meilleur (vert)
    """
    # Métriques où une baisse est positive
    loss_metrics = ["rmse", "mae", "mape_percent", "mse"]
    
    if key.lower() in loss_metrics:
        # Plus bas = meilleur
        if diff < 0:  # Amélioration
            return "🟢", "green"
        elif diff > 0:  # Dégradation
            return "🔴", "red"
        else:
            return "⚪", "gray"
    else:
        # Pour R² et autres: plus haut = meilleur
        if diff > 0:  # Amélioration
            return "🟢", "green"
        elif diff < 0:  # Dégradation
            return "🔴", "red"
        else:
            return "⚪", "gray"

def format_metric_with_color(key, base, new_val, diff):
    """Formate une métrique avec code couleur HTML"""
    emoji, color = get_comparison_emoji_and_color(key, diff)
    
    # Couleur pour le diff
    if color == "green":
        diff_color = "#28a745"
    elif color == "red":
        diff_color = "#dc3545"
    else:
        diff_color = "#6c757d"
    
    return f"""
<tr>
    <td><strong>{key}</strong></td>
    <td>{base:.4f}</td>
    <td>{new_val:.4f}</td>
    <td style="color: {diff_color}; font-weight: bold;">{diff:+.4f} {emoji}</td>
</tr>
"""

def compare_metrics(baseline, new, links):
    report = []
    report.append("# 📊 Model Metrics Comparison\n")
    
    # ---- METRICS TABLE ----
    report.append("\n## 📈 Métriques")
    report.append("<table>")
    report.append("<thead><tr><th>Métrique</th><th>Baseline</th><th>Nouveau</th><th>Différence</th></tr></thead>")
    report.append("<tbody>")
    
    for key in baseline.keys():
        base = baseline[key]
        new_val = new.get(key, None)
        if new_val is None:
            report.append(f"<tr><td colspan='4'>❌ `{key}` missing in new metrics</td></tr>")
            continue
        
        diff = new_val - base
        report.append(format_metric_with_color(key, base, new_val, diff))
    
    report.append("</tbody></table>")
    
    # ---- INTERPRETATION ----
    report.append("\n## 💡 Interprétation")
    report.append("- 🟢 **Vert** : Amélioration par rapport au baseline")
    report.append("- 🔴 **Rouge** : Dégradation par rapport au baseline")
    report.append("- Pour RMSE, MAE, MAPE : plus bas = meilleur")
    report.append("- Pour R² : plus haut = meilleur")

    # ---- PLOTS ----
    report.append("\n## 📊 Comparaison des plots")

    plots = [
        ("Vrais vs Prédits", "pred_vs_true", links.get("NEW_PRED_URL", "metrics/pred_vs_true.png"), links.get("BASE_PRED_URL", "metrics/pred_vs_true_baseline.png")),
        ("Résiduels", "residuals", links.get("NEW_RES_URL", "metrics/residuals.png"), links.get("BASE_RES_URL", "metrics/residuals_baseline.png"))
    ]

    for title, name, new_url, base_url in plots:
        report.append(f"\n### {title}")
        report.append(f"**Nouveau modèle:**\n\n![]({new_url})\n")
        report.append(f"**Baseline:**\n\n![]({base_url})\n")

    return "\n".join(report)

def main():
    baseline_path = "metrics/baseline_metrics.json"
    new_path = "metrics/metrics.json"
    output_path = "reports/comparison_report.md"

    baseline = load_json(baseline_path)
    new = load_json(new_path)
    links = load_links()

    report = compare_metrics(baseline, new, links)

    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✔ Comparison report saved to {output_path}")

if __name__ == "__main__":
    main()