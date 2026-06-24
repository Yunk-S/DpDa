import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
import json
import traceback
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, precision_recall_curve
)

from model_utilities import smooth_probability

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

def load_models_and_data(dataset_name=None):
    """Load models and preprocessed data"""
    models = {}
    data = {}

    model_dir = 'output/models'
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                model_path = os.path.join(model_dir, filename)
                model_name = filename.replace('.pkl', '')

                if dataset_name and not model_name.startswith(dataset_name):
                    continue

                try:
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")

    data_dir = 'output/processed_data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                data_name = filename.replace('_processed.csv', '')

                if dataset_name and data_name != dataset_name:
                    continue

                try:
                    data[data_name] = pd.read_csv(data_path)
                    logger.info(f"Loaded data: {data_name}")
                except Exception as e:
                    logger.error(f"Failed to load data {data_name}: {e}")

    return models, data

def prepare_test_data(data):
    """Prepare test data"""
    test_datasets = {}

    for dataset_name, df in data.items():
        if dataset_name == 'stroke':
            X = df.drop(['stroke', 'id', 'N_Days'], axis=1, errors='ignore')
            y = df['stroke']
            test_datasets[dataset_name] = (X, y)
        elif dataset_name == 'heart':
            X = df.drop(['HeartDisease', 'id', 'N_Days'], axis=1, errors='ignore')
            y = df['HeartDisease']
            test_datasets[dataset_name] = (X, y)
        elif dataset_name == 'cirrhosis':
            X = df.drop(['Stage', 'id', 'N_Days'], axis=1, errors='ignore')
            y = df['Stage']
            test_datasets[dataset_name] = (X, y)

    return test_datasets

def generate_roc_curves(models, test_datasets):
    """Generate ROC curves"""
    logger.info("Generating ROC curves...")

    for dataset_name, (X_test, y_test) in test_datasets.items():
        if dataset_name == 'cirrhosis':
            logger.info(f"{dataset_name} is a regression task, skipping ROC curve generation")
            continue

        best_model_name = f"{dataset_name}_best_baseline_model"

        if best_model_name not in models:
            logger.warning(f"Best model for {dataset_name} not found, skipping ROC curve generation")
            continue

        model = models[best_model_name]

        try:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)

                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]

                    y_prob_smoothed = np.zeros_like(y_prob)
                    for i, prob in enumerate(y_prob):
                        if prob < 0.2:
                            risk_level = 'low'
                            min_prob, max_prob = 0.02, 0.90
                        elif prob < 0.5:
                            risk_level = 'medium'
                            min_prob, max_prob = 0.05, 0.95
                        else:
                            risk_level = 'high'
                            min_prob, max_prob = 0.10, 0.98

                        y_prob_smoothed[i] = smooth_probability(prob, method='sigmoid_quantile',
                                                              min_prob=min_prob, max_prob=max_prob)

                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)

                    fpr_smooth, tpr_smooth, thresholds_smooth = roc_curve(y_test, y_prob_smoothed)
                    roc_auc_smooth = auc(fpr_smooth, tpr_smooth)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Original ROC Curve (AUC = {roc_auc:.2f})')
                    plt.plot(fpr_smooth, tpr_smooth, color='green', lw=2, label=f'Smoothed ROC Curve (AUC = {roc_auc_smooth:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Prediction (AUC = 0.5)')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve for {dataset_name.capitalize()} Model')
                    plt.legend(loc="lower right")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_roc_curve.png')
                    plt.close()

                    logger.info(f"ROC curve generated for {dataset_name}")
                else:
                    logger.warning(f"{dataset_name} is not a binary classification problem, skipping ROC curve generation")
            else:
                logger.warning(f"{dataset_name} model does not support probability prediction, skipping ROC curve generation")

        except Exception as e:
            logger.error(f"Error generating ROC curve for {dataset_name}: {e}")
            logger.error(traceback.format_exc())

def generate_feature_importance(models, test_datasets):
    """Generate feature importance charts"""
    logger.info("Generating feature importance charts...")

    for dataset_name, (X_test, y_test) in test_datasets.items():
        best_model_name = f"{dataset_name}_best_baseline_model"

        if best_model_name not in models:
            logger.warning(f"Best model for {dataset_name} not found, skipping feature importance chart generation")
            continue

        model = models[best_model_name]

        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                feature_names = X_test.columns

                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })

                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

                top_n = min(15, len(feature_importance_df))
                feature_importance_df = feature_importance_df.head(top_n)

                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance_df)
                plt.title(f'Feature Importance for {dataset_name.capitalize()} Model')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_feature_importance.png')
                plt.close()

                logger.info(f"Feature importance chart generated for {dataset_name}")

            elif hasattr(model, 'coef_'):
                coefs = model.coef_

                if coefs.ndim > 1:
                    coefs = np.abs(coefs).mean(axis=0)

                feature_names = X_test.columns

                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(coefs)
                })

                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

                top_n = min(15, len(feature_importance_df))
                feature_importance_df = feature_importance_df.head(top_n)

                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance_df)
                plt.title(f'Feature Importance (Absolute Coefficient) for {dataset_name.capitalize()} Model')
                plt.xlabel('Absolute Coefficient Value')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_feature_importance.png')
                plt.close()

                logger.info(f"Feature importance chart (based on coefficients) generated for {dataset_name}")

            else:
                logger.warning(f"{dataset_name} model does not support feature importance visualization, skipping feature importance chart generation")

        except Exception as e:
            logger.error(f"Error generating feature importance chart for {dataset_name}: {e}")

def generate_confusion_matrices(models, test_datasets):
    """Generate confusion matrices"""
    logger.info("Generating confusion matrices...")

    for dataset_name, (X_test, y_test) in test_datasets.items():
        if dataset_name == 'cirrhosis':
            logger.info(f"{dataset_name} is a regression task, skipping confusion matrix generation")
            continue

        best_model_name = f"{dataset_name}_best_baseline_model"

        if best_model_name not in models:
            logger.warning(f"Best model for {dataset_name} not found, skipping confusion matrix generation")
            continue

        model = models[best_model_name]

        try:
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {dataset_name.capitalize()} Model')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f'output/figures/{dataset_name}_confusion_matrix.png')
            plt.close()

            logger.info(f"Confusion matrix generated for {dataset_name}")

        except Exception as e:
            logger.warning(f"Skipped confusion matrix for {dataset_name}: {e}")

def generate_residual_plots(models, test_datasets):
    """Generate residual plots"""
    logger.info("Generating residual plots...")

    for dataset_name, (X_test, y_test) in test_datasets.items():
        best_model_name = f"{dataset_name}_best_baseline_model"

        if best_model_name not in models:
            logger.warning(f"Best model for {dataset_name} not found, skipping residual plot generation")
            continue

        model = models[best_model_name]

        try:
            y_pred = model.predict(X_test)

            if dataset_name in ['stroke', 'heart']:
                accuracy = accuracy_score(y_test, y_pred)

                plt.figure(figsize=(10, 6))
                plt.bar(['Accuracy'], [accuracy], color='blue')
                plt.ylim(0, 1)
                plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
                plt.title(f'Prediction Accuracy for {dataset_name.capitalize()} Model')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                plt.close()

                logger.info(f"Prediction accuracy chart generated for {dataset_name}")

                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]

                        plt.figure(figsize=(10, 6))
                        for target_class in [0, 1]:
                            sns.kdeplot(y_prob[y_test == target_class],
                                     label=f'True Class = {target_class}',
                                     fill=True, alpha=0.5)
                        plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
                        plt.title(f'Predicted Probability Distribution for {dataset_name.capitalize()} Model')
                        plt.xlabel('Probability of Positive Class')
                        plt.ylabel('Density')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(f'output/figures/{dataset_name}_prob_distribution.png')
                        plt.close()

                        logger.info(f"Predicted probability distribution chart generated for {dataset_name}")
                    except Exception as e:
                        logger.error(f"Error generating predicted probability distribution chart for {dataset_name}: {e}")

            else:
                residuals = y_test.values - y_pred

                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Predicted Value')
                plt.ylabel('Residual')
                plt.title(f'Residual Plot for {dataset_name.capitalize()} Model')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                plt.close()

                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Actual Value')
                plt.ylabel('Predicted Value')
                plt.title(f'Predicted vs Actual for {dataset_name.capitalize()} Model')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_pred_vs_actual.png')
                plt.close()

                logger.info(f"Residual plots generated for {dataset_name}")

        except Exception as e:
            logger.warning(f"Skipped residual plots for {dataset_name}: {e}")

def generate_metrics_json(models, test_datasets):
    """Generate model evaluation metrics JSON files"""
    logger.info("Generating model evaluation metrics...")

    for dataset_name, (X_test, y_test) in test_datasets.items():
        dataset_models = [m for m in models.keys() if m.startswith(dataset_name)]

        for model_name in dataset_models:
            model = models[model_name]
            metrics = {}

            try:
                y_pred = model.predict(X_test)

                if dataset_name in ['stroke', 'heart']:
                    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))

                    try:
                        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
                        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
                        metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
                    except:
                        metrics['precision'] = "N/A"
                        metrics['recall'] = "N/A"
                        metrics['f1'] = "N/A"

                    if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                        try:
                            y_prob = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            metrics['auc'] = float(auc(fpr, tpr))
                        except:
                            metrics['auc'] = "N/A"
                    else:
                        metrics['auc'] = "N/A"

                else:
                    metrics['mse'] = float(mean_squared_error(y_test, y_pred))
                    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    metrics['r2'] = float(r2_score(y_test, y_pred))

                with open(f'output/models/{model_name}_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=4)

                logger.info(f"Evaluation metrics generated for {model_name}")

            except Exception as e:
                logger.warning(f"Skipped metrics for {model_name}: {e}")

def generate_shap_visualizations(models, test_datasets):
    """Generate SHAP value visualizations"""
    logger.info("Attempting to generate SHAP value visualizations...")

    try:
        import shap

        for dataset_name, (X_test, y_test) in test_datasets.items():
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42)

            best_model_name = f"{dataset_name}_best_baseline_model"

            if best_model_name not in models:
                logger.warning(f"Best model for {dataset_name} not found, skipping SHAP value visualization")
                continue

            model = models[best_model_name]

            try:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
                    plt.title(f'SHAP Feature Importance for {dataset_name.capitalize()} Model')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_importance.png')
                    plt.close()

                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    plt.title(f'SHAP Value Summary for {dataset_name.capitalize()} Model')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_summary.png')
                    plt.close()

                    logger.info(f"SHAP value visualization generated for {dataset_name}")

                except Exception as e:
                    logger.warning(f"TreeExplainer failed, trying KernelExplainer: {e}")

                    smaller_sample = min(50, sample_size)
                    X_smaller = X_sample.iloc[:smaller_sample]

                    def model_predict(X):
                        return model.predict(X)

                    explainer = shap.KernelExplainer(model_predict, X_smaller)
                    shap_values = explainer.shap_values(X_smaller)

                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_smaller, plot_type='bar', show=False)
                    plt.title(f'SHAP Feature Importance for {dataset_name.capitalize()} Model')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_importance.png')
                    plt.close()

                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values, X_smaller, show=False)
                    plt.title(f'SHAP Value Summary for {dataset_name.capitalize()} Model')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_summary.png')
                    plt.close()

                    logger.info(f"SHAP value visualization (KernelExplainer) generated for {dataset_name}")

            except Exception as e:
                logger.warning(f"Skipped SHAP visualization for {dataset_name}: {e}")

    except ImportError:
        logger.warning("SHAP library not installed, skipping SHAP value visualization")

def run_chart_generation(dataset_name=None, chart_types=None):
    """Run all chart generation functions"""
    logger.info(f"Starting to generate {'all' if chart_types is None or 'all' in chart_types else ', '.join(chart_types) if chart_types else 'all'} charts for {'all' if dataset_name is None else dataset_name} datasets...")

    models, data = load_models_and_data(dataset_name)

    if not models:
        logger.error(f"No {'any' if dataset_name is None else dataset_name} models found!")
        return

    if not data:
        logger.error(f"No {'any' if dataset_name is None else dataset_name} data found!")
        return

    test_datasets = prepare_test_data(data)

    if chart_types is None or 'all' in chart_types or 'roc' in chart_types:
        generate_roc_curves(models, test_datasets)

    if chart_types is None or 'all' in chart_types or 'feature' in chart_types:
        generate_feature_importance(models, test_datasets)

    if chart_types is None or 'all' in chart_types or 'confusion' in chart_types:
        generate_confusion_matrices(models, test_datasets)

    if chart_types is None or 'all' in chart_types or 'residual' in chart_types:
        generate_residual_plots(models, test_datasets)

    if chart_types is None or 'all' in chart_types or 'metrics' in chart_types:
        generate_metrics_json(models, test_datasets)

    if chart_types is None or 'all' in chart_types or 'shap' in chart_types:
        generate_shap_visualizations(models, test_datasets)

    logger.info(f"{'All' if chart_types is None or 'all' in chart_types else ', '.join(chart_types) if chart_types else 'all'} charts generation completed for {'all' if dataset_name is None else dataset_name} datasets!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model evaluation charts and metrics')
    parser.add_argument('--dataset', type=str, choices=['stroke', 'heart', 'cirrhosis'],
                        help='Specify the dataset to process, process all if not specified')
    parser.add_argument('--chart-type', type=str, nargs='+',
                        choices=['roc', 'feature', 'confusion', 'residual', 'shap', 'metrics', 'all'],
                        default=['all'], help='Specify chart types to generate, default is all types')

    args = parser.parse_args()

    run_chart_generation(args.dataset, args.chart_type)
