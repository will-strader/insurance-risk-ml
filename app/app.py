import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model and feature pipeline
model_path = Path("models/model.pkl")
pipeline_path = Path("models/pipeline.pkl")

if not model_path.exists() or not pipeline_path.exists():
    st.error("Model or pipeline not found. Please train the model first.")
    st.stop()

model = joblib.load(model_path)
pipeline = joblib.load(pipeline_path)

# Path to fallback sample data
sample_csv_path = Path("data/raw/train.csv")

# App UI
st.title("Insurance Claim Prediction Dashboard")

st.markdown(
    """
    Upload a CSV file of policyholder data or use a sample file to get claim amount predictions.
    """
)

# Upload or use sample
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
use_sample = st.checkbox("Use built-in sample file instead", value=False)

# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom file uploaded.")
elif use_sample and sample_csv_path.exists():
    df = pd.read_csv(sample_csv_path)
    st.info("Using sample file: `train.csv`")
else:
    st.warning("Please upload a file or use the sample file.")
    st.stop()

# Show preview
st.subheader("Input Data Preview")
st.write(df.head())

# Run prediction
try:
    X_transformed = pipeline.transform(df)
    predictions = model.predict(X_transformed)
    df["Predicted_Claim_Amount"] = predictions

    st.subheader("Predicted Claims")
    st.write(df[["Predicted_Claim_Amount"]].head())

    # Download button
    st.download_button(
        label="Download Full Predictions as CSV",
        data=df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Prediction failed: {e}")