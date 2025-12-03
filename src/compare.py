# src/compare.py
import json
import os

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def compare_metrics(baseline, new):
    """
    Retourne un rapport Markdown avec table :
    | Metric | Baseline | New | Diff | Status |
    """
    report = ["# 📊 Model Metrics Comparison\n"]
    report.append("| Metric | Baseline | New | Diff | Status |")
    report.append("|--------|----------|-----|------|--------|")

    for key in baseline.keys():
        base = baseline[key]
        new_val = new.get(key, None)

        if new_val is None:
            report.append(f"| {key} | {base:.4f} | - | - | ❌ Missing |")
            continue

        diff = new_val - base
        status = "🟢 Improved" if diff < 0 else "🔴 Worse" if diff > 0 else "⚪ Same"
        # Attention RMSE/MAE : plus petit = meilleur, R2 : plus grand = meilleur
        if key in ["r2"]:  # plus grand meilleur
            status = "🟢 Improved" if diff > 0 else "🔴 Worse" if diff < 0 else "⚪ Same"

        report.append(f"| {key} | {base:.4f} | {new_val:.4f} | {diff:.4f} | {status} |")

    return "\n".join(report)

def main():
    baseline_path = "metrics/baseline_metrics.json"
    new_path = "metrics/metrics.json"
    output_path = "reports/comparison_report.md"

    if not os.path.exists(baseline_path):
        print("⚠ Baseline metrics not found. Run main branch first to set baseline.")
        baseline = {}
    else:
        baseline = load_json(baseline_path)

    new = load_json(new_path)

    report = compare_metrics(baseline, new)

    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✔ Comparison report saved to {output_path}")

if __name__ == "__main__":
    main()
