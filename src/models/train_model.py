"""
Train LightGBM on processed data, save model, and metrics.
"""

import json
from pathlib import Path
import yaml
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from features.build_features import make_feature_pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def main(cfg_path: str = "configs/base.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = pd.read_parquet(cfg["data"]["processed"])

    y = df[cfg["model"]["target"]]
    X = df.drop(columns=[cfg["model"]["target"]])

    pipe, _ = make_feature_pipeline(df)
    X_enc = pipe.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_enc,
        y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
    )

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": cfg["model"]["n_estimators"],
        "learning_rate": cfg["model"]["learning_rate"],
        "max_depth": cfg["model"]["max_depth"],
        "random_state": cfg["model"]["random_state"],
    }

    model = lgb.LGBMRegressor(**lgb_params)
    logger.info("Training LightGBM …")
    model.fit(X_train, y_train)

    # Limit to 1000 samples as default (should take ~1 minute)
    logger.info("Generating SHAP values (sample of 1000)...")
    # Rebuild dense DataFrame with feature names
    feature_names = pipe.get_feature_names_out()
    sample_df = pd.DataFrame(X_val[:1000].toarray(), columns=feature_names)
    explainer = shap.Explainer(model.booster_)
    shap_values = explainer(sample_df)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary Plot (sample)")
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    logger.info("Saved SHAP summary plot → models/shap_summary.png")

      # Limit to 1000 samples as default (should take ~1 minute)
    logger.info("Generating SHAP values (sample of 1000)...")
    # Rebuild dense DataFrame with feature names
    feature_names = pipe.get_feature_names_out()
    sample_df = pd.DataFrame(X_val[:1000].toarray(), columns=feature_names)
    explainer = shap.Explainer(model.booster_)
    shap_values = explainer(sample_df)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)

    # Save individual SHAP explanation (waterfall plot for first sample)
    plt.clf()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("models/shap_waterfall_sample0.png")
    logger.info("Saved SHAP waterfall → models/shap_waterfall_sample0.png")
    
    plt.title("SHAP Summary Plot (sample)")
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    logger.info("Saved SHAP summary plot → models/shap_summary.png")

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    logger.info("Validation MAE: %.3f", mae)

    Path("models").mkdir(exist_ok=True)
    model.booster_.save_model(cfg["output"]["model_path"])
    json.dump({"mae": mae}, open(cfg["output"]["metrics_path"], "w"))
    logger.info("Artifacts saved!")

    # Save model and pipeline for use in Streamlit
    joblib.dump(model, "models/model.pkl", compress=3)
    joblib.dump(pipe, "models/pipeline.pkl", compress=3)
    logger.info("Saved model and pipeline for app.")

if __name__ == "__main__":
    main()