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

def compare_metrics(baseline, new, links):
    report = []
    report.append("# 📊 Model Metrics Comparison\n")

    # ---- METRICS ----
    for key in baseline.keys():
        base = baseline[key]
        new_val = new.get(key, None)
        if new_val is None:
            report.append(f"- ❌ `{key}` missing in new metrics\n")
            continue
        
        diff = new_val - base
        emoji = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
        report.append(f"- **{key}**: baseline={base:.4f}, new={new_val:.4f}, diff={diff:.4f} {emoji}")

    # ---- PLOTS ----
    report.append("\n## 📈 Comparaison des plots")

    plots = [
        ("Vrais vs Prédits", "pred_vs_true", links.get("NEW_PRED_URL", "metrics/pred_vs_true.png"),links.get("BASE_PRED_URL", "metrics/pred_vs_true_baseline.png")),
        ("Résiduels", "residuals", links.get("NEW_RES_URL", "metrics/residuals.png"), links.get("BASE_RES_URL", "metrics/residuals_baseline.png"))
    ]

    for title, name, new_url , base_url in plots:
        report.append(f"\n### {title}")
        report.append(f"**Nouveau modèle:** ![]({new_url})")
        report.append(f"**Baseline:** ![]({base_url})")

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
