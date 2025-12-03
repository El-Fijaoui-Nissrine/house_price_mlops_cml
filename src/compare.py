import json
import os

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def compare_metrics(baseline, new):
    report = []
    report.append("# 📊 Model Metrics Comparison\n")

    for key in baseline.keys():
        base = baseline[key]
        new_val = new.get(key, None)

        if new_val is None:
            report.append(f"- ❌ `{key}` missing in new metrics\n")
            continue
        
        diff = new_val - base
        emoji = "🟢" if diff > 0 else "🔴" if diff < 0 else "⚪"
        report.append(
            f"- **{key}**: baseline={base:.4f}, new={new_val:.4f}, diff={diff:.4f} {emoji}"
        )

    # Ajouter images (nouveau vs baseline)
    report.append("\n## 📈 Comparaison des plots")
    plots = [
        ("Vrais vs Prédits", "plot_pred", "pred_vs_true"),
        ("Résiduels", "plot_resid", "residuals")
    ]
    for title, key, base_name in plots:
        new_img = f"metrics/{base_name}.png"
        baseline_img = f"metrics/{base_name}_baseline.png"
        report.append(f"### {title}")
        if os.path.exists(new_img):
            report.append(f"**Nouveau modèle:** ![](metrics/pred_vs_true.png)")
        if os.path.exists(baseline_img):
            report.append(f"**Baseline:** ![](metrics/pred_vs_true_baseline.png)")

    return "\n".join(report)

def main():
    baseline_path = "metrics/baseline_metrics.json"
    new_path = "metrics/metrics.json"
    output_path = "reports/comparison_report.md"

    baseline = load_json(baseline_path)
    new = load_json(new_path)

    report = compare_metrics(baseline, new)

    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✔ Comparison report saved to {output_path}")

if __name__ == "__main__":
    main()
