from pathlib import Path
import lightgbm as lgb
import pandas as pd
from features.build_features import make_feature_pipeline

MODEL_PATH = "models/lightgbm_model.txt"

def load_model():
    return lgb.Booster(model_file=MODEL_PATH)

def predict(df: pd.DataFrame):
    pipeline, _ = make_feature_pipeline(df)
    X = pipeline.fit_transform(df)
    model = load_model()
    return model.predict(X)