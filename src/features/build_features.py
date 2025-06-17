"""
Feature engineering pipeline for the Kaggle Allstate Claims Severity dataset.

- Target encodes high‑cardinality categoricals (cat1‑cat116)
- Adds normalized frequency (count) encoding for the same columns
- Scales continuous features (cont1‑cont14)
- Optionally adds pair‑wise interaction terms among continuous variables
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from category_encoders import TargetEncoder, CountEncoder

from utils.logger import get_logger

logger = get_logger(__name__)

def make_feature_pipeline(df: pd.DataFrame, add_interactions: bool = True):
    """
    Build a preprocessing pipeline tailored to the Allstate Claims dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw training dataframe including the `loss` column.
    add_interactions : bool, default True
        Whether to generate pair‑wise interaction terms among continuous vars.

    Returns
    -------
    transformer : ColumnTransformer
        Fitted/ready preprocessing transformer.
    feature_names : list[str]
        Ordered list of feature names after transformation (approximate – some
        estimators ignore feature names).
    """
    # Identify columns
    cat_cols = [c for c in df.columns if c.startswith("cat")]
    cont_cols = [c for c in df.columns if c.startswith("cont")]

    # Target column
    if "loss" in df.columns:
        df_features = df.drop(columns=["loss"])
    else:
        df_features = df.copy()

    logger.info("Categorical: %s", cat_cols)
    logger.info("Continuous : %s", cont_cols)

    # Encoders
    target_enc = TargetEncoder(smoothing=0.3)
    count_enc  = CountEncoder(normalize=True)

    cont_pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2,
                                        interaction_only=True,
                                        include_bias=False))
            if add_interactions else ("identity", "passthrough"),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            # Target encoded categoricals
            ("cat_te",  target_enc, cat_cols),
            # Frequency encoded categoricals
            ("cat_cnt", count_enc,  cat_cols),
            # Continuous numeric features (with optional interactions)
            ("cont",    cont_pipeline, cont_cols),
        ],
        remainder="drop"
    )

    # Return list of approximate feature names (not perfect for encoded cats)
    feature_names = (
        [f"{col}_te" for col in cat_cols] +
        [f"{col}_cnt" for col in cat_cols] +
        cont_cols  # base names; polynomial names ignored for brevity
    )

    return transformer, feature_names
