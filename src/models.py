"""
Model Training and Inference Interfaces for PropIQ Advisor
"""

import os
import joblib
import pandas as pd
from typing import Tuple, Any

def load_models(models_dir: str) -> Tuple[Any, Any]:
    """Loads saved classification and regression pipelines."""
    clf_path = os.path.join(models_dir, 'classification_model.joblib')
    reg_path = os.path.join(models_dir, 'regression_model.joblib')
    
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Classification model not found at: {clf_path}")
    if not os.path.exists(reg_path):
        raise FileNotFoundError(f"Regression model not found at: {reg_path}")
        
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return clf, reg

def predict_investment(clf_model: Any, reg_model: Any, input_df: pd.DataFrame) -> dict:
    """Generates dual classification & regression advisory outputs."""
    is_good = int(clf_model.predict(input_df)[0])
    prob_good = float(clf_model.predict_proba(input_df)[0][1])
    
    curr_price = float(input_df['Price_in_Lakhs'].iloc[0])
    pred_5y_price = float(reg_model.predict(input_df)[0])
    # Minimum sanity guard
    pred_5y_price = max(curr_price * 1.02, pred_5y_price)

    net_gain = pred_5y_price - curr_price
    roi_pct = (net_gain / curr_price) * 100.0
    cagr_pct = ((pred_5y_price / curr_price) ** (1/5.0) - 1.0) * 100.0

    return {
        'is_good_investment': is_good,
        'confidence_score': round(prob_good if is_good == 1 else (1 - prob_good), 4),
        'probability_good': round(prob_good, 4),
        'current_price_lakhs': round(curr_price, 2),
        'future_price_5y_lakhs': round(pred_5y_price, 2),
        'net_capital_gain_lakhs': round(net_gain, 2),
        'roi_percentage_5y': round(roi_pct, 2),
        'cagr_percentage': round(cagr_pct, 2),
        'recommendation': "HIGH-YIELD INVESTMENT (BUY)" if is_good == 1 else "BELOW-HURDLE ASSET (HOLD/AVOID)"
    }
