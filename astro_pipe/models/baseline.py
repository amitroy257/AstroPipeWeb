from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def build_model(model_cfg: dict):
    model_type = model_cfg.get("type", "logreg")

    if model_type == "logreg":
        return LogisticRegression(
            max_iter=int(model_cfg.get("max_iter", 1000)),
            class_weight=model_cfg.get("class_weight", "balanced"),
        )

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            class_weight=model_cfg.get("class_weight", "balanced"),
            random_state=42,
        )

    raise ValueError(f"Unknown model type: {model_type}")