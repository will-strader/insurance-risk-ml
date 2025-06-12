# Insurance Claims Severity Predictor

A full-stack machine learning web application that predicts the severity of insurance claims using structured policyholder data. Built on Flask and deployed to Azure App Service with automated CI/CD using GitHub Actions.

## 🔍 Project Overview

This app utilizes a LightGBM-based regression model trained on the [Allstate Claims Severity dataset (Kaggle)](https://www.kaggle.com/c/allstate-claims-severity) to predict the monetary value of insurance claims. The web interface allows users to upload CSV files or test using a provided template.

## Features

- **Supervised ML Model**: Built using LightGBM and scikit-learn
- **Feature Engineering**: Extensive preprocessing to handle 130+ categorical and continuous features
- **Web App**: Developed using Flask, HTML/CSS, and JavaScript
- **Real-time Inference**: Upload your CSV to generate predictions dynamically
- **Azure Deployment**: Hosted on Azure App Service using a Basic B1 plan
- **CI/CD**: Integrated with GitHub Actions for seamless updates

## File Structure
```
insurance-risk-ml/
│
├── static/                   # Includes sample CSV data
│   └── sample_data.csv
│
├── templates/               # HTML template for UI
│   └── index.html
│
├── models/                  # Trained model artifacts
│   ├── model.pkl
│   ├── pipeline.pkl
│   └── shap_summary.png
│
├── src/                     # Data pipeline and utilities
│   ├── data/                # make_dataset.py for preprocessing
│   ├── features/            # Feature engineering scripts
│   ├── models/              # Training and prediction code
│   └── utils/               # Logging tools
│
├── data/                    # Raw and processed datasets
├── webapp.py                # Flask app
├── requirements.txt         # Python dependencies
└── README.md                # You're here!
```

## Sample CSV Template
Download a [CSV template](https://insurance-claims-severity-model-app-gdc8dyg7dub2hwc2.canadacentral-01.azurewebsites.net/static/sample_data.csv) to try the model with example input. For best results, keep files under 50MB.

## Setup Instructions

### Local Setup
```bash
git clone https://github.com/will-strader/insurance-risk-ml.git
cd insurance-risk-ml
pip install -r requirements.txt
python webapp.py
```
Visit `http://localhost:5000` in your browser.

### Azure Deployment
1. Zip the repo root (include app.py, src/, models/, data/, templates/, static/)
2. Deploy via Azure App Service
3. Set app startup command: `gunicorn webapp:app --bind=0.0.0.0 --timeout 600`
4. Configure GitHub Actions for CI/CD

## Model Details
- LightGBM Regressor with 200+ features
- Evaluation metric: Mean Absolute Error (MAE)
- Tuned via GridSearchCV with early stopping

## Technologies Used
- Python, scikit-learn, LightGBM
- Flask, HTML/CSS, JavaScript
- Azure App Service, GitHub Actions

## License
MIT License

## Author
**Will Strader** – [GitHub](https://github.com/will-strader) | [LinkedIn](https://www.linkedin.com/in/william-strader-1a4879221/)

---

_Disclaimer: This application is for demonstration purposes and not production-grade for actual claim evaluation._
