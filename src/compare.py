
import os
import json

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compare_metrics(baseline, new):
    lines = ["# 📊 Model Metrics Comparison\n"]

    for key in baseline:
        base = baseline[key]
        new_val = new[key]
        diff = new_val - base

        emoji = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
        lines.append(
            f"* **{key}**: baseline={base:.4f}, new={new_val:.4f}, diff={diff:.4f} {emoji}"
        )

    return "\n".join(lines)


def main():
    baseline = load_json("metrics/baseline_metrics.json")
    new = load_json("metrics/metrics.json")
    urls = load_json("metrics/plot_urls.json")

    report = []
    report.append(compare_metrics(baseline, new))

    report.append("\n## 📈 Comparaison des plots\n")

    # Images hébergées chez CML (URLs)
    report.append("### Vrais vs Prédits\n")
    report.append(f"**Nouveau modèle :** ![]({urls['pred_vs_true']})")
    report.append("")

    report.append("### Résiduels\n")
    report.append(f"**Nouveau modèle :** ![]({urls['residuals']})")
    report.append("")

    os.makedirs("reports", exist_ok=True)
    with open("reports/comparison_report.md", "w") as f:
        f.write("\n".join(report))

    print("✔ Rapport généré : reports/comparison_report.md")


if __name__ == "__main__":
    main()
