from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io

app = Flask(__name__)
app.config["SECRET_KEY"] = "replace‑with‑a‑random‑secret"

# Load trained model and preprocessing pipeline
BASE_DIR = Path(__file__).resolve().parent           # /home/site/wwwroot when deployed
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.pkl"
TRAIN_CSV_PATH = BASE_DIR / "train.csv"            # small sample file placed alongside webapp.py

if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
    raise FileNotFoundError(
        "models/model.pkl or models/pipeline.pkl not found. "
        "Make sure the 'models' folder is included in the deployment package."
    )

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    """Home page that accepts a file upload and returns predictions."""
    if request.method == "POST":
        # Check which option user selected
        if "predict_file" in request.files and request.files["predict_file"].filename:
            # User uploaded a custom CSV
            csv_file = request.files["predict_file"]
            df = pd.read_csv(csv_file)
        elif request.form.get("use_sample"):
            # User chose built‑in sample
            df = pd.read_csv(TRAIN_CSV_PATH)
        else:
            flash("Please upload a CSV file or select the sample option.", "warning")
            return redirect(url_for("index"))

        try:
            # Transform features and predict
            X_transformed = pipeline.transform(df)
            preds = model.predict(X_transformed)
            df["Predicted_Claim_Amount"] = preds

            # Serialize predictions to CSV in‑memory for download link
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            return send_file(
                io.BytesIO(buf.getvalue().encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name="predictions.csv",
            )
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")
            return redirect(url_for("index"))

    # GET request – render upload form
    return render_template("index.html")


if __name__ == "__main__":
    # Note: In production (Azure App Service) you'll use gunicorn:
    #   gunicorn app:app
    app.run(host="0.0.0.0", port=5000, debug=True)