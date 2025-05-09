"""
Creates a simple numeric/categorical split and one‑hot encodes categoricals.
Replace or extend with Target Encoding for better performance.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from utils.logger import get_logger

logger = get_logger(__name__)

def make_feature_pipeline(df: pd.DataFrame):
    cat_cols = df.select_dtypes("object").columns.tolist()
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    num_cols.remove("Claim_Amount")  # target

    logger.info("Categorical: %s", cat_cols)
    logger.info("Numerical: %s", num_cols)

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    return transformer, num_cols + cat_cols
