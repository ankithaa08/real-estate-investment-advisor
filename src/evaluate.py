"""
Model Evaluation and Metrics Utility Module
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def evaluate_classifier(y_true, y_pred, y_prob=None) -> dict:
    """Calculates classification evaluation metrics."""
    metrics = {
        'Accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'Precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_true, y_pred)), 4),
        'F1_Score': round(float(f1_score(y_true, y_pred)), 4)
    }
    if y_prob is not None:
        metrics['ROC_AUC'] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return metrics

def evaluate_regressor(y_true, y_pred) -> dict:
    """Calculates regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return {
        'RMSE': round(float(rmse), 3),
        'MAE': round(float(mae), 3),
        'R2_Score': round(float(r2), 4),
        'MAPE_Pct': round(float(mape), 3)
    }
