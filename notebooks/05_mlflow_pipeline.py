"""
MLflow Experiment Tracking & Model Governance Module
Domain: Real Estate Investment Advisor
Tracks: Experiments, Model Runs, Hyperparameters, Performance Metrics, Evaluation Figures, and Models
"""

import os
import json
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

def run_mlflow_tracking(base_dir: str):
    mlruns_dir = os.path.join(base_dir, 'mlruns')
    models_dir = os.path.join(base_dir, 'models')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    mlflow_db_path = os.path.join(base_dir, 'mlruns', 'mlflow.db').replace('\\', '/')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    experiment_name = "RealEstate_Investment_Advisor"
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Tracking URI set to: {mlruns_dir}")
    print(f"[MLflow] Active Experiment: {experiment_name}")

    # Load benchmark metrics
    clf_metrics_file = os.path.join(models_dir, 'classification_metrics.json')
    reg_metrics_file = os.path.join(models_dir, 'regression_metrics.json')

    if not os.path.exists(clf_metrics_file) or not os.path.exists(reg_metrics_file):
        raise FileNotFoundError("Classification or regression metrics not found. Run model training first!")

    with open(clf_metrics_file, 'r') as f:
        clf_data = json.load(f)

    with open(reg_metrics_file, 'r') as f:
        reg_data = json.load(f)

    # 1. Log Classification Runs
    print("\n[MLflow] Logging Classification Models...")
    for model_name, metrics in clf_data['models_comparison'].items():
        run_name = f"Classification_{model_name.replace(' ', '_')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("task_type", "classification")
            mlflow.set_tag("target", "Is_Good_Investment")
            mlflow.set_tag("algorithm", model_name)
            
            # Log params
            if 'XGBoost' in model_name:
                mlflow.log_params({"n_estimators": 120, "max_depth": 6, "learning_rate": 0.08, "eval_metric": "logloss"})
            elif 'Random Forest' in model_name:
                mlflow.log_params({"n_estimators": 100, "max_depth": 12, "criterion": "gini"})
            else:
                mlflow.log_params({"max_iter": 1000, "solver": "lbfgs"})

            # Log metrics
            mlflow.log_metrics({
                "accuracy": metrics['Accuracy'],
                "precision": metrics['Precision'],
                "recall": metrics['Recall'],
                "f1_score": metrics['F1_Score'],
                "roc_auc": metrics['ROC_AUC']
            })

            # If champion model, log artifacts
            if model_name == clf_data['champion_model']:
                mlflow.set_tag("status", "CHAMPION_MODEL")
                roc_fig = os.path.join(figures_dir, 'classification_roc_curve.png')
                cm_fig = os.path.join(figures_dir, 'classification_confusion_matrix.png')
                feat_fig = os.path.join(figures_dir, 'classification_feature_importance.png')
                model_file = os.path.join(models_dir, 'classification_model.joblib')

                for art in [roc_fig, cm_fig, feat_fig, model_file]:
                    if os.path.exists(art):
                        mlflow.log_artifact(art)

            print(f" -> Logged {run_name} with F1 = {metrics['F1_Score']:.4f}")

    # 2. Log Regression Runs
    print("\n[MLflow] Logging Regression Models...")
    for model_name, metrics in reg_data['models_comparison'].items():
        run_name = f"Regression_{model_name.replace(' ', '_')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("task_type", "regression")
            mlflow.set_tag("target", "Future_Price_5Y")
            mlflow.set_tag("algorithm", model_name)

            # Log params
            if 'XGBoost' in model_name:
                mlflow.log_params({"n_estimators": 140, "max_depth": 6, "learning_rate": 0.08})
            elif 'Random Forest' in model_name:
                mlflow.log_params({"n_estimators": 100, "max_depth": 14})
            elif 'Ridge' in model_name:
                mlflow.log_params({"alpha": 1.0})
            else:
                mlflow.log_params({"fit_intercept": True})

            # Log metrics
            mlflow.log_metrics({
                "rmse": metrics['RMSE'],
                "mae": metrics['MAE'],
                "r2_score": metrics['R2_Score'],
                "mape_pct": metrics['MAPE_Pct']
            })

            # If champion model, log artifacts
            if model_name == reg_data['champion_model']:
                mlflow.set_tag("status", "CHAMPION_MODEL")
                pred_fig = os.path.join(figures_dir, 'regression_actual_vs_predicted.png')
                res_fig = os.path.join(figures_dir, 'regression_residual_dist.png')
                feat_fig = os.path.join(figures_dir, 'regression_feature_importance.png')
                model_file = os.path.join(models_dir, 'regression_model.joblib')

                for art in [pred_fig, res_fig, feat_fig, model_file]:
                    if os.path.exists(art):
                        mlflow.log_artifact(art)

            print(f" -> Logged {run_name} with RMSE = {metrics['RMSE']:.3f}L, R2 = {metrics['R2_Score']:.4f}")

    print("\n[MLflow] All model runs successfully tracked and persisted to MLflow repository!")

if __name__ == '__main__':
    base_dir = r'C:\Users\Admin\.gemini\antigravity\scratch\real_estate_advisor'
    run_mlflow_tracking(base_dir)
