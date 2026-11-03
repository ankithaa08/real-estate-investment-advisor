"""
Classification Modeling Module - Real Estate Investment Prediction
Target: Is_Good_Investment (Binary: 1 = Good Investment, 0 = Below Threshold)
Algorithms: Logistic Regression, Random Forest, XGBoost Classifier
Outputs: Evaluation metrics, Confusion Matrix, ROC Curve, Feature Importance, Saved Model Artifact
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

def run_classification_pipeline(features_path: str, models_dir: str, figures_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"[Classification] Loading dataset from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"[Classification] Dataset shape: {df.shape}")

    # Feature definitions
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

    target_col = 'Is_Good_Investment'

    X = df[num_cols + cat_cols]
    y = df[target_col]

    print(f"[Classification] Class Distribution:\n{y.value_counts(normalize=True)}")

    # Train/Test Split (Stratified 80/20)
    # Using sample for fast yet high-accuracy training if dataset is 250k
    sample_size = min(60000, len(df))
    print(f"[Classification] Subsampling {sample_size} records for stratified model training & tuning...")
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Models dictionary
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.08, 
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
    }

    results = {}
    best_f1 = -1
    best_model_name = None
    best_pipeline = None

    plt.figure(figsize=(8, 6))

    for name, clf in models.items():
        print(f"\n[Classification] Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            'Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1_Score': round(float(f1), 4),
            'ROC_AUC': round(float(roc_auc), 4)
        }
        print(f" -> {name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    # Complete ROC Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.title('Classification ROC Curves Comparison', weight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_fig_path = os.path.join(figures_dir, 'classification_roc_curve.png')
    plt.savefig(roc_fig_path, dpi=200)
    plt.close()

    print(f"\n[Classification] Champion Model: {best_model_name} (F1 = {best_f1:.4f})")

    # Evaluate best model confusion matrix
    y_best_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_best_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Good (0)', 'Good (1)'],
                yticklabels=['Not Good (0)', 'Good (1)'])
    plt.title(f'Confusion Matrix - {best_model_name}', weight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Ground Truth')
    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, 'classification_confusion_matrix.png')
    plt.savefig(cm_fig_path, dpi=200)
    plt.close()

    # Feature Importance for Tree Models
    try:
        clf_step = best_pipeline.named_steps['classifier']
        ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_features_out = ohe.get_feature_names_out(cat_cols).tolist()
        all_feature_names = num_cols + cat_features_out

        if hasattr(clf_step, 'feature_importances_'):
            importances = clf_step.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=feat_imp_df, y='Feature', x='Importance', palette='viridis')
            plt.title(f'Top 15 Feature Importances ({best_model_name})', weight='bold')
            plt.xlabel('Relative Feature Importance')
            plt.tight_layout()
            feat_fig_path = os.path.join(figures_dir, 'classification_feature_importance.png')
            plt.savefig(feat_fig_path, dpi=200)
            plt.close()
    except Exception as e:
        print(f"[Warning] Could not extract feature importances: {e}")

    # Save Best Model Pipeline
    model_save_path = os.path.join(models_dir, 'classification_model.joblib')
    joblib.dump(best_pipeline, model_save_path)
    print(f"[Classification] Saved best model pipeline to: {model_save_path}")

    # Save benchmark metrics
    metrics_path = os.path.join(models_dir, 'classification_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'champion_model': best_model_name,
            'models_comparison': results,
            'test_classification_report': classification_report(y_test, y_best_pred, output_dict=True)
        }, f, indent=2)

    return results

if __name__ == '__main__':
    base_dir = r'C:\Users\Admin\.gemini\antigravity\scratch\real_estate_advisor'
    feat_path = os.path.join(base_dir, 'data', 'processed', 'housing_features.csv')
    models_dir = os.path.join(base_dir, 'models')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    run_classification_pipeline(feat_path, models_dir, figures_dir)
