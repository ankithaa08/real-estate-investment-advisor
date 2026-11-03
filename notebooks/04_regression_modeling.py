"""
Regression Modeling Module - 5-Year Future Property Price Forecasting
Target: Future_Price_5Y (₹ in Lakhs)
Algorithms: Linear Regression, Ridge, Random Forest Regressor, XGBoost Regressor
Outputs: RMSE, MAE, R², MAPE, Actual vs Predicted Plot, Residuals, Saved Model Artifact
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_regression_pipeline(features_path: str, models_dir: str, figures_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"[Regression] Loading dataset from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"[Regression] Dataset shape: {df.shape}")

    num_cols = [
        'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
        'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools',
        'Nearby_Hospitals', 'Amenities_Count', 'Infrastructure_Score',
        'Floor_Ratio', 'Locality_Relative_Price'
    ]

    cat_cols = [
        'Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility',
        'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status',
        'City'
    ]

    target_col = 'Future_Price_5Y'

    X = df[num_cols + cat_cols]
    y = df[target_col]

    sample_size = min(60000, len(df))
    print(f"[Regression] Subsampling {sample_size} records for model training & evaluation...")
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=sample_size, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
        'XGBoost': XGBRegressor(
            n_estimators=140, max_depth=6, learning_rate=0.08,
            random_state=42, n_jobs=-1
        )
    }

    results = {}
    best_rmse = float('inf')
    best_model_name = None
    best_pipeline = None

    for name, reg in models.items():
        print(f"\n[Regression] Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results[name] = {
            'RMSE': round(float(rmse), 3),
            'MAE': round(float(mae), 3),
            'R2_Score': round(float(r2), 4),
            'MAPE_Pct': round(float(mape), 3)
        }
        print(f" -> {name}: RMSE={rmse:.3f} Lakhs, MAE={mae:.3f} Lakhs, R2={r2:.4f}, MAPE={mape:.2f}%")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n[Regression] Champion Model: {best_model_name} (RMSE = {best_rmse:.3f} Lakhs)")

    # Actual vs Predicted Plot
    y_best_pred = best_pipeline.predict(X_test)
    residuals = y_test - y_best_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_best_pred, alpha=0.3, color='#1f77b4', s=15, edgecolors='none')
    p_min = min(y_test.min(), y_best_pred.min())
    p_max = max(y_test.max(), y_best_pred.max())
    plt.plot([p_min, p_max], [p_min, p_max], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    plt.title(f'Actual vs Predicted 5-Year Property Price ({best_model_name})', weight='bold')
    plt.xlabel('Actual Future Price (₹ Lakhs)')
    plt.ylabel('Predicted Future Price (₹ Lakhs)')
    plt.legend()
    plt.tight_layout()
    pred_fig_path = os.path.join(figures_dir, 'regression_actual_vs_predicted.png')
    plt.savefig(pred_fig_path, dpi=200)
    plt.close()

    # Residuals Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=40, color='#e377c2', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title(f'Residuals Distribution - {best_model_name} (Mean Error = {residuals.mean():.2f}L)', weight='bold')
    plt.xlabel('Prediction Residual (Actual - Predicted in ₹ Lakhs)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    res_fig_path = os.path.join(figures_dir, 'regression_residual_dist.png')
    plt.savefig(res_fig_path, dpi=200)
    plt.close()

    # Feature Importance for Regression
    try:
        reg_step = best_pipeline.named_steps['regressor']
        ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_features_out = ohe.get_feature_names_out(cat_cols).tolist()
        all_feature_names = num_cols + cat_features_out

        if hasattr(reg_step, 'feature_importances_'):
            importances = reg_step.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, y='Feature', x='Importance', palette='magma')
            plt.title(f'Top 15 Feature Importances for Price Forecasting ({best_model_name})', weight='bold')
            plt.xlabel('Relative Feature Importance')
            plt.tight_layout()
            reg_feat_path = os.path.join(figures_dir, 'regression_feature_importance.png')
            plt.savefig(reg_feat_path, dpi=200)
            plt.close()
    except Exception as e:
        print(f"[Warning] Could not extract regression feature importances: {e}")

    # Save Best Model Pipeline
    model_save_path = os.path.join(models_dir, 'regression_model.joblib')
    joblib.dump(best_pipeline, model_save_path)
    print(f"[Regression] Saved best model pipeline to: {model_save_path}")

    # Save benchmark metrics
    metrics_path = os.path.join(models_dir, 'regression_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'champion_model': best_model_name,
            'models_comparison': results
        }, f, indent=2)

    return results

if __name__ == '__main__':
    base_dir = r'C:\Users\Admin\.gemini\antigravity\scratch\real_estate_advisor'
    feat_path = os.path.join(base_dir, 'data', 'processed', 'housing_features.csv')
    models_dir = os.path.join(base_dir, 'models')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    run_regression_pipeline(feat_path, models_dir, figures_dir)
