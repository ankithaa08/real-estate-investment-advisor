# Real Estate Investment Advisor (PropIQ)
### Predicting Property Profitability & Future Capital Appreciation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/XGBoost-Champion-green.svg)](https://xgboost.readthedocs.io/)
[![Tracking](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Database](https://img.shields.io/badge/SQLite-Indexed-003B57.svg)](https://www.sqlite.org/)

A full-stack, enterprise-grade machine learning project developed to empower real estate investors, homebuyers, and financial institutions with predictive intelligence on property profitability, risk classification, and 5-year capital appreciation.

---

## 🌟 Key Deliverables Overview

1. **Cleaned & Processed Datasets & Database:**
   - `data/processed/cleaned_housing_prices.csv`: 250,000 cleaned real estate transaction records.
   - `data/processed/housing_features.csv`: 43 engineered attributes including composite livability score, locality price ratios, and future valuation targets.
   - `data/processed/real_estate_database.db`: Production-ready SQLite database with indexed tables for high-speed queries.

2. **Full Exploratory Data Analysis (EDA):**
   - Answers to all **20 analytical questions** specified in the capstone curriculum.
   - 20 high-resolution charts saved in `reports/figures/` (`eda_01.png` to `eda_20.png`).
   - Structured JSON findings summary in `reports/eda_summary.json` and Markdown report in `reports/eda_summary_report.md`.

3. **Production Machine Learning Models:**
   - **Supervised Classification:** Predicts whether an asset is a "Good Investment" (`Is_Good_Investment`).
     - **Champion:** XGBoost Classifier (**Accuracy: 95.37%**, **F1-Score: 0.9304**, **ROC-AUC: 0.9918**).
   - **Supervised Regression:** Forecasts estimated property price after 5 years (`Future_Price_5Y`).
     - **Champion:** XGBoost Regressor (**RMSE: ₹7.772 Lakhs**, **R²: 0.9982**, **MAPE: 1.68%**).
   - Saved pre-trained model artifacts in `models/classification_model.joblib` and `models/regression_model.joblib`.

4. **Interactive Streamlit Web Application:**
   - Multi-tab application featuring:
     - 🏠 **Investment Advisor & Forecast Form:** Real-time dual prediction, confidence gauge, and Year 0–5 price growth trajectory chart.
     - 🔍 **Property Finder & Portfolio Screener:** Dynamic SQL filtering across 250,000 properties with CSV export.
     - 📊 **Market Analytics & 20 EDA Insights:** Interactive visual browser for all 20 questions.
     - 🧪 **Model Benchmark & MLflow:** Live metrics, confusion matrix, ROC curve, and feature importances.
     - 📄 **Executive Capstone Report:** In-app mentor documentation.

5. **MLflow Experiment Tracking:**
   - Complete tracking pipeline logging runs, hyperparameters, metrics, and serialized artifacts in `mlruns/mlflow.db`.

6. **Comprehensive Documentation:**
   - Full capstone submission document in `reports/final_project_report.md`.

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the interactive dashboard locally:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

### 3. Re-running the Pipelines (Optional)
If you wish to re-execute any stage of the project:

- **Step 1: Data Preprocessing & SQLite Export**
  ```bash
  python src/data_pipeline.py
  ```
- **Step 2: Generate 20 EDA Visualizations & Reports**
  ```bash
  python notebooks/01_eda_and_analysis.py
  ```
- **Step 3: Train & Evaluate Classification Models**
  ```bash
  python notebooks/03_classification_modeling.py
  ```
- **Step 4: Train & Evaluate Regression Models**
  ```bash
  python notebooks/04_regression_modeling.py
  ```
- **Step 5: Run MLflow Experiment Tracking**
  ```bash
  python notebooks/05_mlflow_pipeline.py
  ```

### 4. Viewing MLflow UI
To open the MLflow dashboard and inspect model runs:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
Navigate to `http://localhost:5000` to review runs, metrics, and logged artifacts.

---

## 📊 Summary of Model Performance

### Classification Benchmark (Predicting "Good Investment")
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 90.69% | 87.01% | 85.10% | 0.8605 | 0.9706 |
| Random Forest Classifier | 90.12% | 85.17% | 85.59% | 0.8538 | 0.9693 |
| **XGBoost Classifier (Champion)** | **95.37%** | **94.52%** | **91.60%** | **0.9304** | **0.9918** |

### Regression Benchmark (Forecasting 5-Year Property Price in ₹ Lakhs)
| Model | RMSE (Lakhs) | MAE (Lakhs) | R² Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression (Baseline) | 16.535 | 12.726 | 0.9919 | 7.63% |
| Ridge Regression | 16.535 | 12.726 | 0.9919 | 7.63% |
| Random Forest Regressor | 14.431 | 10.607 | 0.9938 | 3.06% |
| **XGBoost Regressor (Champion)** | **7.772** | **5.518** | **0.9982** | **1.68%** |

---

## 📁 Repository Directory Structure

```
real_estate_advisor/
├── app/
│   └── app.py                                # Streamlit Web Application
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv          # Original raw dataset
│   └── processed/
│       ├── cleaned_housing_prices.csv        # Cleaned dataset
│       ├── housing_features.csv              # Processed features & targets
│       └── real_estate_database.db           # SQLite Database with indexes
├── mlruns/
│   └── mlflow.db                             # MLflow tracking backend
├── models/
│   ├── classification_model.joblib           # Champion XGBoost Classifier
│   ├── regression_model.joblib               # Champion XGBoost Regressor
│   ├── classification_metrics.json           # Classification benchmark
│   └── regression_metrics.json               # Regression benchmark
├── notebooks/
│   ├── 01_eda_and_analysis.py                # 20 EDA questions & charts generator
│   ├── 03_classification_modeling.py        # Classification training & tuning
│   ├── 04_regression_modeling.py            # Regression training & tuning
│   └── 05_mlflow_pipeline.py                # MLflow tracking script
├── reports/
│   ├── figures/                              # 20 EDA figures + ML diagnostics (26 total)
│   ├── eda_summary.json                      # JSON findings for all 20 questions
│   ├── eda_summary_report.md                 # Markdown summary of EDA
│   └── final_project_report.md               # Complete mentor Capstone Report
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                      # Preprocessing & database export
│   ├── feature_engineering.py                # Feature calculations
│   ├── models.py                             # Inference utilities
│   └── evaluate.py                           # Evaluation helpers
├── requirements.txt                          # Project dependencies
└── README.md                                 # Project documentation
```
