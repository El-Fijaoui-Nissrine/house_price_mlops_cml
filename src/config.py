# src/config.py
import yaml
from pathlib import Path

DEFAULT_PARAMS = {
    "model": {
        "type": "GradientBoostingRegressor",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42,
        "test_size": 0.2
    },
    "paths": {
        "raw_data": "data/processed/processed_data.csv",
        "model_out": "models/model.joblib",
        "metrics_out": "metrics/metrics.json",
        "plot_pred": "metrics/pred_vs_true.png",
        "plot_resid": "metrics/residuals.png"
    }
}

def load_params(path: str = "params.yaml") -> dict:
    """
    Charge params depuis params.yaml si existe, sinon retourne DEFAULT_PARAMS.
    """
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            user = yaml.safe_load(f) or {}
        # fusion simple user->default (pas de deep merge complexe)
        params = DEFAULT_PARAMS.copy()
        params.update(user)
        return params
    return DEFAULT_PARAMS
