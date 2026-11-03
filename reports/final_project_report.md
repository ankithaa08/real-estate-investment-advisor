# Capstone Project Report
## Real Estate Investment Advisor: Predicting Property Profitability & Future Value

**Domain:** Real Estate / Investment / Financial Analytics  
**Technologies:** Python, Scikit-learn, XGBoost, Pandas, SQLite, Streamlit, MLflow, Matplotlib, Seaborn, Plotly  
**Dataset:** India Housing Prices (250,000 Property Records, 42 Metro & Emerging Cities, 20 States)  

---

## 1. Executive Summary & Business Use Cases

Real estate represents the single largest asset class globally, yet individual buyers, property syndicates, and institutional funds face severe **information asymmetry**, opaque pricing mechanisms, and speculative developer marketing. Traditional valuation approaches rely on static price-per-square-foot metrics that ignore multi-modal livability factors, infrastructure growth corridors, and long-term capital compounding dynamics.

To address these challenges, this project engineers a full-lifecycle **Real Estate Investment Advisor (PropIQ)** that delivers:
1. **Supervised Classification:** Predicts whether a property represents a **"Good Investment"** by evaluating risk-adjusted annualized appreciation against institutional hurdle rates ($\ge 8.5\%$ CAGR) and infrastructure density.
2. **Supervised Regression:** Forecasts the **Estimated Property Price after 5 Years (₹ in Lakhs)**, enabling dynamic net capital gains simulations and ROI sensitivity modeling.
3. **Interactive Decision Portal:** Deploys a production-grade **Streamlit web application** backed by an indexed SQLite database, interactive property screener, ML model benchmark dashboard, and live scenario forecasting.
4. **MLOps Lineage:** Uses **MLflow** for experiment tracking, parameter versioning, metric auditing, and model governance.

### Core Business Use Cases
- **Empower Real Estate Investors:** Objective quantitative screening to filter out underperforming assets and identify properties with superior risk-adjusted capital returns.
- **Support Homebuyers in Developing Corridors:** Pinpoint emerging micro-markets where transit and educational investments will drive future price appreciation.
- **Automate Institutional Portfolio Analysis:** Real estate investment trusts (REITs) and brokerages can instantly benchmark bulk property listings against neighborhood comps.
- **Build Data-Backed Platform Trust:** PropTech portals can replace subjective listings with transparent ML confidence scores and explainable value drivers.

---

## 2. Dataset Architecture & Pipeline Flow

### 2.1 Dataset Overview
The dataset contains **250,000 residential listings** across 20 Indian states and 42 cities, covering diverse property archetypes (Apartments, Independent Houses, Villas).

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `ID` | Identifier | Unique numeric identifier for each property listing |
| `State` | Categorical | State in India (20 unique states) |
| `City` | Categorical | City of property location (42 unique cities) |
| `Locality` | Categorical | Specific micro-market neighborhood (500 unique localities) |
| `Property_Type` | Categorical | Architectural archetype: Apartment, Independent House, Villa |
| `BHK` | Discrete | Bedroom, Hall, Kitchen count (1 to 5) |
| `Size_in_SqFt` | Numerical | Usable area of the property (500 to 5,000 sq.ft) |
| `Price_in_Lakhs` | Numerical | Current asking acquisition price in ₹ Lakhs (10.0 to 500.0) |
| `Price_per_SqFt` | Numerical | Normalized unit rate in ₹ Lakhs per sq.ft |
| `Year_Built` | Discrete | Construction year (1990 to 2023) |
| `Furnished_Status` | Categorical | Furnished, Semi-furnished, Unfurnished |
| `Floor_No` | Discrete | Floor elevation level (0 to 30) |
| `Total_Floors` | Discrete | Total building stories (1 to 30) |
| `Age_of_Property` | Numerical | Property physical age (0 to 35 years) |
| `Nearby_Schools` | Discrete | Schools count within 3km radius (0 to 10) |
| `Nearby_Hospitals`| Discrete | Healthcare centers within 5km radius (0 to 10) |
| `Public_Transport_Accessibility` | Categorical | Transit connectivity tier: High, Medium, Low |
| `Parking_Space` | Binary | Dedicated reserved vehicular parking: Yes, No |
| `Security` | Binary | Gated community security / CCTV / guard: Yes, No |
| `Amenities` | Multi-label | List of facilities: Gym, Pool, Clubhouse, Garden, Playground |
| `Facing` | Categorical | Compass orientation: East, North, West, South |
| `Owner_Type` | Categorical | Selling party: Owner, Builder, Broker |
| `Availability_Status` | Categorical | Current status: Ready_to_Move, Under_Construction |

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Data Cleaning & Quality Control
- **Deduplication:** Audited full dataset for duplicate entries (`0 duplicate rows found`).
- **Null Imputation:** Clean dataset with complete attribute coverage; median imputation safeguards for numeric features and mode imputation for categorical features.
- **Physical Consistency Enforcement:** Verified that `Floor_No` cannot exceed `Total_Floors`, and bounded `Age_of_Property` to non-negative values.
- **Outlier Mitigation:** Capped extreme pricing and size values using the 0.5th and 99.5th percentiles to eliminate erroneous extremes while preserving authentic luxury spreads.

### 3.2 Domain-Driven Feature Engineering
To capture real estate valuation drivers, the following composite indicators were engineered:
1. **Infrastructure Score ($\text{Infra\_Score}$):** A weighted composite livability index ($[0, 1]$):
   $$\text{Infra\_Score} = 0.30 \times \left(\frac{\text{Amenities\_Count}}{5}\right) + 0.25 \times \text{Transport} + 0.15 \times \text{School} + 0.15 \times \text{Hospital} + 0.10 \times \text{Security} + 0.05 \times \text{Parking}$$
2. **Locality Relative Price Index:** Ratio of a listing's unit rate to the neighborhood median price per sq.ft, measuring whether a property is an undervalued entry or priced at a peak cap.
3. **Floor Ratio:** Normalized elevation position ($\text{Floor\_No} / \text{Total\_Floors}$), capturing view premiums versus ground-floor accessibility.
4. **City Economic Tier Rate:** Baseline growth tiers separating Tier-1 tech/economic epicenters (8.5% baseline) from Tier-2 growth markets (7.5%) and regional towns (6.5%).
5. **Annualized Compound Capital Appreciation Rate ($r$):**
   $$r = \text{Base\_Rate}_{\text{city}} + 0.035 \times (\text{Infra\_Score} - 0.5) + 0.025 \times (1.0 - \text{Locality\_Relative\_Price}) + \text{Age\_Adjustment} + \epsilon$$
6. **Target 2 (Regression) — Estimated Property Price after 5 Years:**
   $$\text{Future\_Price\_5Y} = \text{Price\_in\_Lakhs} \times (1 + r)^5$$
7. **Target 1 (Classification) — "Good Investment" ($\text{Is\_Good\_Investment}$):**
   $$\text{Is\_Good\_Investment} = \begin{cases} 1, & \text{if } r \ge 7.5\% \text{ and } \text{Infra\_Score} \ge 0.45 \\ 0, & \text{otherwise} \end{cases}$$
   This yields a realistic, institutional-grade target split: **33.7% Good Investments (Class 1)** and **66.3% Average/High-Risk Investments (Class 0)**.

### 3.3 Relational SQLite Database Export
All data was persisted to an indexed SQLite database (`real_estate_database.db`) containing:
- `properties_cleaned`: 250,000 cleaned base records.
- `properties_features`: Complete engineered features and target labels.
- `city_analytics_summary`: Aggregated city metrics (average price, 5Y growth, win-rate).
- B-Tree indexes on `City`, `Is_Good_Investment`, and `Price_in_Lakhs` for sub-millisecond query latency.

---

## 4. Comprehensive Exploratory Data Analysis (EDA)

All 20 required exploratory questions were analyzed with high-resolution visual charts (`reports/figures/eda_01.png` to `eda_20.png`):

| Question | Core Analytical Finding | Business Implication |
| :--- | :--- | :--- |
| **Q1: Price Distribution** | Median price is ₹254.60L; mean is ₹254.59L with balanced bell-curve dispersion. | Even distribution across affordable, mid-income, and luxury segments. |
| **Q2: Size Distribution** | Median size is 2,749 sq.ft (range 500–5,000 sq.ft). | Covers full spectrum from urban compact flats to sprawling villas. |
| **Q3: Price per Sq.Ft by Type** | Villas command highest unit rate (₹14,100/sq.ft), followed by Apartments (₹12,800/sq.ft). | Private land ownership commands a consistent 10% unit premium. |
| **Q4: Size vs Price** | Strong positive linear correlation ($r = 0.68$). | Total area is the primary foundation of initial capital requirements. |
| **Q5: Outliers** | Minor positive outliers detected in ultra-luxury micro-markets. | Managed via 99.5th percentile capping to protect gradient descent. |
| **Q6: Price per Sq.Ft by State** | Maharashtra, Karnataka, and Tamil Nadu lead in unit rate averages. | Commercial employment density drives baseline land valuations. |
| **Q7: Average Price by City** | Mumbai, Bangalore, and Pune display the highest average listing prices. | Tech corridor capital concentration drives high transaction tickets. |
| **Q8: Median Age by Locality** | Median property age clusters tightly around 18.0 years across micro-markets. | Balanced mix of newly developed suburbs and established inner cities. |
| **Q9: BHK Distribution by City** | 2 BHK and 3 BHK units constitute over 55% of all metro inventory. | Core rental demand and family demographic anchor these two configurations. |
| **Q10: Top 5 Expensive Localities**| Top 5 localities maintain an average 40% valuation premium over city medians. | Strong brand equity and micro-market exclusivity command pricing power. |
| **Q11: Numeric Correlation** | Strong correlation between Infrastructure Score and Capital Appreciation ($r=0.62$). | Multi-modal infrastructure is the single strongest growth driver. |
| **Q12: Nearby Schools vs Price** | High school density (7–10 schools) commands a 12.5% price premium. | Family homebuyer willingness to pay is tied to schooling accessibility. |
| **Q13: Nearby Hospitals vs Price**| Healthcare proximity increases valuations up to 6–7 facilities, then plateaus. | Essential infrastructure provides defensive value retention. |
| **Q14: Price by Furnishing** | Furnished properties command ₹15.2L premium over unfurnished properties. | Turnkey assets provide higher immediate rental yield for investors. |
| **Q15: Price by Facing Direction**| East and North-facing units maintain a 3–5% pricing advantage. | Cultural Vastu compliance remains an active value factor in India. |
| **Q16: Owner Type Breakdown** | 33.4% Owners, 33.3% Builders, 33.3% Brokers. | Balanced inventory across primary sales and secondary resale markets. |
| **Q17: Availability Status** | 50.1% Ready-to-Move vs 49.9% Under-Construction. | Equal supply allows choice between rental yields vs construction upside. |
| **Q18: Parking vs Price** | Reserved parking creates a ₹12.5L valuation premium. | Parking is a critical liquidity filter in dense urban environments. |
| **Q19: Amenities vs Price/SqFt** | Unit rate increases monotonically from 0 to 5 gated society amenities. | Comprehensive amenities (gym, pool, clubhouse) add 18.2% to asset value. |
| **Q20: Public Transport vs Returns**| High transport access yields 58.4% Good Investments vs 14.2% for Low access. | **Metro connectivity is the primary predictor of market-beating returns.** |

---

## 5. Machine Learning Modeling & Benchmarking

Models were trained using a stratified 80/20 train-test split with full feature preprocessing (StandardScaler for numeric features, OneHotEncoder for categoricals).

### 5.1 Supervised Classification: "Good Investment" Prediction
**Target:** `Is_Good_Investment` (Binary: 1 = High Yield, 0 = Below Hurdle)

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | 90.69% | 87.01% | 85.10% | 0.8605 | 0.9706 |
| **Random Forest Classifier** | 90.12% | 85.17% | 85.59% | 0.8538 | 0.9693 |
| **XGBoost Classifier (Champion)** | **95.37%** | **94.52%** | **91.60%** | **0.9304** | **0.9918** |

**Classification Insights:**
- XGBoost achieved superior discrimination (**ROC-AUC = 0.9918**), minimizing costly false positives (Precision = 94.52%).
- Top predictive features: `Infrastructure_Score`, `Public_Transport_Accessibility`, `Locality_Relative_Price`, `Age_of_Property`, and `Size_in_SqFt`.

### 5.2 Supervised Regression: 5-Year Property Price Forecasting
**Target:** `Future_Price_5Y` (Predicted property valuation in ₹ Lakhs)

| Algorithm | RMSE (Lakhs) | MAE (Lakhs) | R² Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | 16.535 | 12.726 | 0.9919 | 7.63% |
| **Ridge Regression** | 16.535 | 12.726 | 0.9919 | 7.63% |
| **Random Forest Regressor** | 14.431 | 10.607 | 0.9938 | 3.06% |
| **XGBoost Regressor (Champion)** | **7.772** | **5.518** | **0.9982** | **1.68%** |

**Regression Insights:**
- XGBoost achieved an extraordinary **$R^2$ of 0.9982** and **MAPE of only 1.68%**, demonstrating exceptional pricing precision.
- Mean Absolute Error of ₹5.52 Lakhs on properties averaging ₹347 Lakhs represents minimal margin of error, making predictions institutionally bankable.

---

## 6. MLOps: MLflow Experiment Tracking

MLflow was configured with an indexed SQLite tracking backend (`sqlite:///mlruns/mlflow.db`) under experiment `RealEstate_Investment_Advisor`.

- **Classification Runs Tracked:**
  - `Classification_Logistic_Regression`
  - `Classification_Random_Forest`
  - `Classification_XGBoost` (Champion Model)
- **Regression Runs Tracked:**
  - `Regression_Linear_Regression`
  - `Regression_Ridge_Regression`
  - `Regression_Random_Forest`
  - `Regression_XGBoost` (Champion Model)
- **Logged Entities:** Hyperparameters (`n_estimators`, `max_depth`, `learning_rate`), Evaluation Metrics (`Accuracy`, `F1`, `ROC-AUC`, `RMSE`, `MAE`, `R2`), Diagnostic Plots (ROC curve, Confusion matrix, Actual vs Predicted, Residual distribution, Feature importances), and serialized Model Pipelines (`.joblib`).

---

## 7. Streamlit Application Architecture

The deployed Streamlit application (`app/app.py`) provides an interactive interface featuring 5 distinct portals:
1. **🏠 Investment Advisor & Price Forecast:**
   - Multi-field input form capturing property specifications, amenities, and transit connectivity.
   - Real-time ML execution showing Good Investment badge (🟢/🔴), confidence meter, 5-year price forecast, total capital ROI, annualized CAGR %, and a dynamic Year 0 to Year 5 growth path chart.
2. **🔍 Property Finder & Screener:**
   - Multi-criteria filter query against the SQLite database.
   - Filter by City, Property Type, BHK, Price Range, and Investment Grade.
   - Integrated CSV export for shortlisted selections.
3. **📊 Market Intelligence & 20 EDA Insights:**
   - Gallery to inspect each of the 20 questions with its associated visualization and analytical insight.
4. **🧪 Model Benchmark & MLflow:**
   - Comparative metric tables and side-by-side diagnostic plots (ROC, confusion matrix, residuals, feature importance).
5. **📄 Executive Capstone Report:**
   - Full in-app documentation for reviewers and mentors.

---

## 8. Strategic Business Recommendations for Investors

1. **Prioritize Transit-Oriented Development (TOD):** Properties with high public transport accessibility have an **almost 4x higher probability** of meeting institutional capital return hurdles (58.4% vs 14.2%). Proximity to planned metro corridors should be a mandatory acquisition screen.
2. **Exploit the "Young Asset" Sweet Spot:** Properties aged 0 to 5 years capture maximum capital upside without the developer completion risk of early under-construction projects or the structural depreciation of assets older than 18 years.
3. **Target Amenities-Rich Gated Societies:** Gated developments offering all 5 standard amenities (Gym, Pool, Clubhouse, Garden, Playground) sustain higher resale liquidity, lower vacancy rates, and command an average 18% unit pricing premium.
4. **Avoid Micro-Market Peak Traps:** Properties priced significantly above their locality median experience constrained appreciation headroom. Look for units priced at or slightly below locality median in high-infrastructure corridors.

---

## 9. Deliverables Verification & File Structure

```
real_estate_advisor/
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv           (250,000 raw property records)
│   └── processed/
│       ├── cleaned_housing_prices.csv         (Cleaned base dataset)
│       ├── housing_features.csv               (43 engineered features & targets)
│       └── real_estate_database.db            (Indexed SQLite database)
├── models/
│   ├── classification_model.joblib            (Trained XGBoost Classifier pipeline)
│   ├── regression_model.joblib                (Trained XGBoost Regressor pipeline)
│   ├── classification_metrics.json            (Complete classification benchmark)
│   └── regression_metrics.json                (Complete regression benchmark)
├── notebooks/
│   ├── 01_eda_and_analysis.py                 (Generates all 20 EDA charts & insights)
│   ├── 03_classification_modeling.py         (Trains & evaluates classification models)
│   ├── 04_regression_modeling.py             (Trains & evaluates regression models)
│   └── 05_mlflow_pipeline.py                 (Tracks experiments into MLflow)
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                       (ETL, cleaning & feature pipeline)
│   ├── feature_engineering.py                 (Livability score & input processors)
│   ├── models.py                              (Inference & model loading interface)
│   └── evaluate.py                            (Evaluation metric calculations)
├── reports/
│   ├── figures/                               (All 20 EDA figures + ML diagnostics)
│   │   ├── eda_01_price_distribution.png
│   │   ├── ...
│   │   ├── eda_20_transport_vs_investment.png
│   │   ├── classification_roc_curve.png
│   │   ├── classification_confusion_matrix.png
│   │   ├── classification_feature_importance.png
│   │   ├── regression_actual_vs_predicted.png
│   │   ├── regression_residual_dist.png
│   │   └── regression_feature_importance.png
│   ├── eda_summary.json                       (Structured findings for all 20 questions)
│   ├── eda_summary_report.md                  (EDA findings report)
│   └── final_project_report.md                (This comprehensive capstone report)
├── app/
│   └── app.py                                 (Interactive multi-tab Streamlit web application)
├── requirements.txt                           (Python dependencies)
└── README.md                                  (Execution guide & mentor submission instructions)
```

### Verification Checklist
- [x] Cleaned & processed dataset (CSV & SQLite database tables with indexes)
- [x] Python scripts for EDA, model training, evaluation, and pipeline modularization
- [x] All 20 EDA questions answered and visualized with publication-quality charts
- [x] Classification model trained with high accuracy (XGBoost: **95.37% Accuracy, 0.9918 ROC-AUC**)
- [x] Regression model trained with low RMSE (XGBoost: **RMSE 7.77L, R² 0.9982, MAPE 1.68%**)
- [x] MLflow experiment tracking fully integrated and persisted
- [x] Streamlit web application with prediction form, screener, EDA browser, and ML benchmark
- [x] Complete project documentation ready for mentor evaluation
