"""
Quick sanity test for the saved pipeline + model.

* Loads a 10-row processed sample (or full processed file if sample missing).
* Transforms with the fitted pipeline.
* Predicts with the saved LightGBM model.
"""

from pathlib import Path
import pandas as pd
from joblib import load

# ------------------------------------------------------------------ #
# File paths
# ------------------------------------------------------------------ #
MODEL_PATH = Path("models/model.pkl")
PIPE_PATH  = Path("models/pipeline.pkl")

PARQUET_SAMPLE = Path("data/processed/claims_sample.parquet")
PARQUET_FULL   = Path("data/processed/claims.parquet")

N_ROWS = 10  # adjust as needed

# ------------------------------------------------------------------ #
# Load artefacts
# ------------------------------------------------------------------ #
print("Loading pipeline and model …")
pipeline = load(PIPE_PATH)
model    = load(MODEL_PATH)

# ------------------------------------------------------------------ #
# Load sample data
# ------------------------------------------------------------------ #
if PARQUET_SAMPLE.exists():
    df = pd.read_parquet(PARQUET_SAMPLE)
elif PARQUET_FULL.exists():
    df = pd.read_parquet(PARQUET_FULL).head(N_ROWS)
else:
    raise FileNotFoundError(
        "No processed parquet file found. Run the data-processing step first."
    )

# Remove target if still present
df = df.drop(columns=[c for c in ["loss", "Claim_Amount"] if c in df.columns])

print(f"Sample loaded: {df.shape[0]} rows × {df.shape[1]} features")

# ------------------------------------------------------------------ #
# Predict
# ------------------------------------------------------------------ #
X = pipeline.transform(df)  # do NOT fit!
preds = model.predict(X)

assert len(preds) == len(df), "Prediction length mismatch"
assert (preds >= 0).all(), "Negative predictions detected"

print("Success! First predictions:", preds[:5])