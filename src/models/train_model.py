"""
Train a LightGBM model on the processed Allstate data, log metrics, generate
SHAP explanations, and save all artifacts.

Run:
    python src/models/train_model.py  # uses configs/base.yaml by default
"""

import json
from pathlib import Path
from typing import Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from features.build_features import make_feature_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


# Helper functions
def _plot_shap_summary(model, X_val, feature_names, out_path: Path, sample: int = 1000):
    """Generate and save SHAP beeswarm + waterfall plots."""
    logger.info("Generating SHAP values on a sample of %d rows …", sample)
    sample_idx = np.random.choice(X_val.shape[0], size=min(sample, X_val.shape[0]), replace=False)
    X_sample = X_val[sample_idx]

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    # Beeswarm summary
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(out_path / "shap_summary.png")
    plt.close()
    logger.info("Saved SHAP summary → %s", out_path / "shap_summary.png")

    # Waterfall for the top‑impact row
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(out_path / "shap_waterfall_sample0.png")
    plt.close()
    logger.info("Saved SHAP waterfall → %s", out_path / "shap_waterfall_sample0.png")


def _train_lightgbm(
    X_train, y_train, X_val, y_val, params: dict
) -> Tuple[lgb.LGBMRegressor, float]:
    """Train model and return fitted model + validation MAE."""
    model = lgb.LGBMRegressor(**params)
    logger.info("Training LightGBM …")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    logger.info("Validation MAE: %.3f", mae)
    return model, mae


# Main training routine
def main(cfg_path: str = "configs/base.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    # Load data
    df = pd.read_parquet(cfg["data"]["processed"])
    target_col = cfg["model"]["target"]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Build preprocessing pipeline & transform
    pipe, _ = make_feature_pipeline(df)
    X_enc = pipe.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_enc,
        y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
    )

    # Train model
    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": cfg["model"]["n_estimators"],
        "learning_rate": cfg["model"]["learning_rate"],
        "max_depth": cfg["model"]["max_depth"],
        "random_state": cfg["model"]["random_state"],
    }
    model, mae = _train_lightgbm(X_train, y_train, X_val, y_val, lgb_params)

    # SHAP explanations
    plots_dir = Path("models")
    plots_dir.mkdir(exist_ok=True)
    _plot_shap_summary(model.booster_, X_val, pipe.get_feature_names_out(), plots_dir)


    # Persist artifacts
    model.booster_.save_model(cfg["output"]["model_path"])
    json.dump({"mae": mae}, open(cfg["output"]["metrics_path"], "w"))

    # Full model & pipeline (for inference web app)
    joblib.dump(model, plots_dir / "model.pkl", compress=3)
    joblib.dump(pipe, plots_dir / "pipeline.pkl", compress=3)
    logger.info("Saved model, pipeline, and metrics. All done!")


if __name__ == "__main__":
    main()