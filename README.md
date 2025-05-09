# Insurance Risk ML 

Predict claim severity from raw policy & loss data, interpret the drivers, and surface insights in a one‑click Streamlit dashboard.  
Built as a showcase of how data science can save insurers $$ by triaging high‑severity claims earlier and optimising reserving.

![Python](https://img.shields.io/badge/Python-3.9-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📈 Key Features
| Stage | What it does | Tech |
|-------|--------------|------|
| **Data prep** | Cleans & joins 188 k rows of anonymised auto‑claim data (Kaggle “Allstate Claims Severity”) | `pandas`, `pyarrow` |
| **Feature engineering** | Target encoding, interaction terms, holiday flags | `scikit‑learn`, `category_encoders` |
| **Model** | Gradient‑boosted trees (LightGBM) with Bayesian hyper‑tuning | `lightgbm`, `optuna` |
| **Explainability** | Global + local SHAP values with plain‑English summaries | `shap` |
| **Dashboard** | Upload a CSV → get severity prediction bands, driver plots, and what‑if sliders | `streamlit` |
| **Deployment** | Dockerfile + GitHub Actions push to AWS ECR or Heroku | `docker`, `gh‑actions` |

---

## Quick Start

```bash
# 1. Clone repo & install
git clone https://github.com/<your‑handle>/insurance-risk-ml.git
cd insurance-risk-ml
conda env create -f environment.yml
conda activate insurance-risk-ml

# 2. Train model & evaluate
python src/train.py --config configs/base.yaml

# 3. Launch dashboard
streamlit run app/app.py
