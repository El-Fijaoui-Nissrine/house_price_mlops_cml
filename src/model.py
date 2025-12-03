# src/model.py
from sklearn.ensemble import GradientBoostingRegressor
from src.config import load_params

def create_model():
    params = load_params()["model"]
    return GradientBoostingRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        random_state=int(params["random_state"])
    )
