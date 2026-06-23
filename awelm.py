"""
Problem 2: Adaptive Weighted Ensemble Learning Model (AWELM)
=====================================
Objective: Build high-accuracy disease prediction model, adapt to data characteristics of different diseases

Method:
1. Select base models: Logistic Regression, Random Forest, Gradient Boosting, SVM
2. Model training and cross-validation (5-fold cross-validation)
3. Weight optimization: Optimize loss function with balancing factor and regularization via gradient descent
4. Ensemble prediction: Weighted average of each model's prediction probabilities

Loss Function (with balancing factor and regularization):
    L(w) = -mean[β·y·log(p) + (1-y)·log(1-p)] + λ·H(w)
where p = Σw_i·p_i, β = negatives/positives, H(w) is negative entropy regularization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from scipy.optimize import minimize
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AdaptiveWeightedEnsemble:
    """
    Adaptive Weighted Ensemble Learning Model (AWELM)

    Core Method:
        1. Train multiple base models (LR, RF, GB, SVM)
        2. Get OOF prediction probabilities for each model through cross-validation
        3. Optimize ensemble weights via gradient descent (with class balancing and negative entropy regularization)
        4. Weighted fusion of model predictions

    Loss Function:
        L(w) = -mean[β·y·log(p) + (1-y)·log(1-p)] + λ·Σw_i·log(w_i)

    Constraints: w_i ≥ 0, Σw_i = 1
    """

    def __init__(self, dataset_name='disease'):
        self.dataset_name = dataset_name
        self.base_models = {}
        self.model_weights = {}
        self.cv_predictions = {}
        self.results = {}

    # ------------------------------------------------------------------
    # 1. Base Model Definition
    # ------------------------------------------------------------------
    def create_base_models(self):
        """Create four base learners"""
        self.base_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            'Random Forest': RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42
            ),
            'SVM': SVC(probability=False, random_state=42, kernel='rbf', C=1.0, cache_size=500),
        }
        logger.info(f"Created {len(self.base_models)} base models")
        return self.base_models

    # ------------------------------------------------------------------
    # 2. Class Imbalance Handling
    # ------------------------------------------------------------------
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """
        Handle class imbalance problem

        Parameters:
            X_train: Training features
            y_train: Training labels
            method: 'smote' or 'oversample'
        """
        unique, counts = np.unique(y_train, return_counts=True)
        imbalance_ratio = max(counts) / min(counts)

        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")

        if imbalance_ratio > 3:
            logger.info(f"Applying {method} oversampling...")
            if method == 'oversample':
                # Simple oversampling
                minority_class = y_train == unique[counts.argmin()]
                minority_indices = np.where(minority_class)[0]
                X_minority = X_train[minority_indices]
                y_minority = y_train[minority_indices]

                n_majority = counts.max()
                n_minority = len(y_minority)
                n_needed = n_majority - n_minority

                # Bootstrap resampling
                resample_idx = np.random.choice(
                    n_minority, size=n_needed, replace=True
                )
                X_resampled = np.vstack([X_train, X_minority[resample_idx]])
                y_resampled = np.concatenate([y_train, y_minority[resample_idx]])

                logger.info(f"After oversampling: {len(y_resampled)} samples")
                return X_resampled, y_resampled

        return X_train, y_train

    # ------------------------------------------------------------------
    # 3. Cross-Validation OOF Prediction
    # ------------------------------------------------------------------
    def get_cv_predictions(self, X, y, models, n_splits=5):
        """
        Get OOF (Out-Of-Fold) prediction probabilities for each model using stratified K-fold cross-validation

        Parameters:
            X: Feature matrix
            y: Labels
            models: Model dictionary
            n_splits: Number of folds

        Returns:
            cv_predictions: Dictionary of OOF prediction probabilities for each model
        """
        cv_predictions = {}
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for model_name, model in models.items():
            logger.info(f"Getting cross-validation predictions for {model_name}...")
            try:
                if hasattr(model, 'predict_proba'):
                    y_cv_pred_proba = cross_val_predict(
                        model, X, y, cv=skf, method='predict_proba'
                    )[:, 1]
                else:
                    y_cv_scores = cross_val_predict(
                        model, X, y, cv=skf, method='decision_function'
                    )
                    y_cv_pred_proba = 1 / (1 + np.exp(-y_cv_scores))
                    y_cv_pred_proba = np.clip(y_cv_pred_proba, 1e-15, 1-1e-15)

                cv_predictions[model_name] = y_cv_pred_proba
                logger.info(f"  {model_name} CV AUC: {roc_auc_score(y, y_cv_pred_proba):.4f}")
            except Exception as e:
                logger.warning(f"  {model_name} CV prediction failed: {e}")

        self.cv_predictions = cv_predictions
        return cv_predictions

    # ------------------------------------------------------------------
    # 4. Weight Optimization (Core)
    # ------------------------------------------------------------------
    def optimize_weights(self, cv_predictions, y_true, lambda_reg=0.01):
        """
        Optimize ensemble model weights via gradient descent

        Loss function (with balancing factor and negative entropy regularization):
            L(w) = -mean[β·y·log(p) + (1-y)·log(1-p)] + λ·H(w)
        Where:
            - p = Σw_i·p_i (weighted average probability)
            - β = negatives/positives (class balancing factor)
            - H(w) = -Σw_i·log(w_i) (negative entropy regularization)

        Constraints: w_i ≥ 0, Σw_i = 1

        Parameters:
            cv_predictions: OOF prediction probabilities for each model
            y_true: True labels
            lambda_reg: Regularization coefficient

        Returns:
            optimal_weights: Optimal weight dictionary
        """
        model_names = list(cv_predictions.keys())
        n_models = len(model_names)
        n_samples = len(y_true)

        # Stack prediction probabilities into matrix (n_samples, n_models)
        P = np.column_stack([cv_predictions[m] for m in model_names])

        # Calculate class balancing factor β = negatives/positives
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        beta = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Class balancing factor β = {beta:.2f}")

        # Define objective function (weighted cross-entropy loss with regularization)
        def objective(weights):
            # Normalize weights
            w = np.clip(weights, 0, 1)
            w = w / (w.sum() + 1e-10)

            # Weighted average probability
            p = P @ w
            p = np.clip(p, 1e-15, 1 - 1e-15)

            # Balanced weighted cross-entropy loss
            # Positive samples weight β, negative samples weight 1
            sample_weights = np.where(y_true == 1, beta, 1.0)
            loss = -np.mean(
                sample_weights * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            )

            # Negative entropy regularization (encourage uniform weight distribution)
            if lambda_reg > 0:
                w_safe = np.clip(w, 1e-15, 1)
                entropy_reg = lambda_reg * np.sum(w * np.log(w_safe))
                loss += entropy_reg

            return loss

        # Initialize to uniform distribution
        initial_weights = np.ones(n_models) / n_models

        # Constraints
        bounds = [(0, 1) for _ in range(n_models)]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        # Normalize final weights
        optimal_weights_raw = np.clip(result.x, 0, 1)
        optimal_weights_norm = optimal_weights_raw / (optimal_weights_raw.sum() + 1e-10)

        optimal_weights = dict(zip(model_names, optimal_weights_norm))

        logger.info(f"Optimized weights: {optimal_weights}")
        logger.info(f"Optimization loss: {result.fun:.6f}")
        self.model_weights = optimal_weights
        return optimal_weights

    # ------------------------------------------------------------------
    # 5. Ensemble Prediction
    # ------------------------------------------------------------------
    def ensemble_predict(self, predictions, weights):
        """
        Weighted ensemble prediction using optimized weights

        Parameters:
            predictions: Dictionary of prediction probabilities for each model on test set
            weights: Optimized weight dictionary

        Returns:
            ensemble_pred: Ensemble prediction classes
            ensemble_proba: Ensemble prediction probabilities
        """
        # Weighted average
        ensemble_proba = np.zeros_like(next(iter(predictions.values())))
        for model_name, proba in predictions.items():
            w = weights.get(model_name, 0)
            ensemble_proba += w * proba

        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        return ensemble_pred, ensemble_proba

    # ------------------------------------------------------------------
    # 6. Evaluation Functions
    # ------------------------------------------------------------------
    def evaluate_model(self, y_true, y_pred, y_proba):
        """Evaluate single model performance"""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        try:
            results['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            results['auc'] = None

        return results

    def plot_model_comparison(self, base_results, ensemble_result,
                               save_path=None):
        """Plot model performance comparison bar chart"""
        model_names = list(base_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        valid_metrics = [m for m in metrics if base_results[model_names[0]].get(m) is not None]

        fig, axes = plt.subplots(1, len(valid_metrics), figsize=(4 * len(valid_metrics), 5))
        if len(valid_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, valid_metrics):
            values = [base_results[m].get(metric, 0) for m in model_names]
            values.append(ensemble_result.get(metric, 0))
            names = model_names + ['Ensemble Model']
            colors = ['steelblue'] * len(model_names) + ['crimson']

            bars = ax.bar(names, values, color=colors)
            ax.set_ylim(0, 1.1)
            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            for bar, val in zip(bars, values):
                if val is not None:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.3f}', ha='center', fontsize=8)

        plt.suptitle(f'{self.dataset_name} - Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, base_predictions, ensemble_proba,
                         y_test, save_path=None):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.Set2(np.linspace(0, 1, len(base_predictions)))

        for (model_name, proba), color in zip(base_predictions.items(), colors):
            try:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.5,
                        label=f'{model_name} (AUC={roc_auc:.3f})')
            except Exception as e:
                logger.warning(f"ROC curve plotting failed ({model_name}): {e}")

        # Ensemble model ROC
        fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_proba)
        auc_ens = auc(fpr_ens, tpr_ens)
        ax.plot(fpr_ens, tpr_ens, color='crimson', lw=2.5,
                label=f'Ensemble Model (AUC={auc_ens:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
        ax.set_title(f'{self.dataset_name} - ROC Curve Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_weight_distribution(self, weights, save_path=None):
        """Plot ensemble weight pie chart"""
        weights_clean = {k: v for k, v in weights.items() if v > 0.001}

        fig, ax = plt.subplots(figsize=(7, 7))
        colors = plt.cm.Set2(np.linspace(0, 1, len(weights_clean)))
        wedges, texts, autotexts = ax.pie(
            weights_clean.values(),
            labels=weights_clean.keys(),
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(weights_clean),
            startangle=90
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title(f'{self.dataset_name} - Ensemble Model Weight Distribution', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # 7. Full Training Pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(self, X, y, test_size=0.2, random_state=42,
                          n_splits=5, lambda_reg=0.01):
        """
        Run complete AWELM pipeline

        Parameters:
            X: Feature matrix
            y: Labels
            test_size: Test set proportion
            random_state: Random seed
            n_splits: Cross-validation folds
            lambda_reg: Regularization coefficient

        Returns:
            results: All model evaluation results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {self.dataset_name} AWELM Training")
        logger.info(f"{'='*60}")

        # Data split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        logger.info(f"Training set: {len(y_train)} samples, Test set: {len(y_test)} samples")

        # Handle class imbalance
        X_train_bal, y_train_bal = self.handle_imbalance(X_train, y_train)

        # Create base models
        self.create_base_models()

        # Train each model on balanced training set
        base_results = {}
        base_predictions = {}

        for model_name, model in self.base_models.items():
            logger.info(f"\nTraining model: {model_name}")
            try:
                model.fit(X_train_bal, y_train_bal)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)[:, 1]
                else:
                    scores = model.decision_function(X_test)
                    proba = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
                pred = (proba >= 0.5).astype(int)

                results = self.evaluate_model(y_test, pred, proba)
                base_results[model_name] = results
                base_predictions[model_name] = proba

                logger.info(f"  AUC: {results.get('auc', 'N/A'):.4f} | "
                           f"F1: {results['f1']:.4f} | "
                           f"ACC: {results['accuracy']:.4f}")
            except Exception as e:
                logger.error(f"  {model_name} training failed: {e}")

        # Cross-validation OOF prediction (use original imbalanced data for unbiased estimate)
        logger.info(f"\nGetting cross-validation OOF predictions (n_splits={n_splits})...")
        cv_preds = self.get_cv_predictions(X_train, y_train,
                                           self.base_models, n_splits=n_splits)

        # Weight optimization
        logger.info(f"\nOptimizing ensemble weights (λ={lambda_reg})...")
        optimal_weights = self.optimize_weights(cv_preds, y_train, lambda_reg=lambda_reg)

        # Ensemble prediction
        logger.info(f"\nEnsemble prediction...")
        ensemble_pred, ensemble_proba = self.ensemble_predict(
            base_predictions,
            optimal_weights
        )
        ensemble_result = self.evaluate_model(y_test, ensemble_pred, ensemble_proba)

        logger.info(f"Ensemble Model AUC: {ensemble_result.get('auc', 'N/A'):.4f} | "
                   f"F1: {ensemble_result['f1']:.4f} | "
                   f"ACC: {ensemble_result['accuracy']:.4f}")

        # Select best model
        all_results = {**base_results, 'Ensemble Model': ensemble_result}
        best_model = max(all_results, key=lambda m: all_results[m].get('auc', 0))
        logger.info(f"\nBest model: {best_model} (AUC: {all_results[best_model].get('auc', 'N/A'):.4f})")

        # Visualization
        logger.info(f"\nGenerating visualization charts...")
        self.plot_model_comparison(
            base_results, ensemble_result,
            save_path=f'output/figures/{self.dataset_name}_model_comparison.png'
        )
        self.plot_roc_curves(
            base_predictions, ensemble_proba, y_test,
            save_path=f'output/figures/{self.dataset_name}_roc_curves.png'
        )
        self.plot_weight_distribution(
            optimal_weights,
            save_path=f'output/figures/{self.dataset_name}_ensemble_weights.png'
        )

        # Save results
        results_df = pd.DataFrame(all_results).T
        results_df.to_excel(f'output/{self.dataset_name}_awelm_results.xlsx')

        weights_df = pd.DataFrame([optimal_weights]).T
        weights_df.columns = ['Weight']
        weights_df.to_excel(f'output/{self.dataset_name}_optimal_weights.xlsx')

        self.results = {
            'base_models': base_results,
            'ensemble': ensemble_result,
            'weights': optimal_weights,
            'best_model': best_model
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"{self.dataset_name} AWELM Training Complete")
        logger.info(f"Optimal weights: {optimal_weights}")
        logger.info(f"{'='*60}\n")

        return self.results


def prepare_data(csv_path, target_col):
    """Load and preprocess data"""
    df = pd.read_csv(csv_path)

    # Exclude irrelevant columns
    exclude = ['id', 'ID']
    feature_cols = [c for c in df.columns if c not in exclude + [target_col]]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    # Standardize
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X.values, y.values, X.columns.tolist()


def run_all_datasets():
    """Run complete AWELM pipeline on three datasets"""
    datasets = [
        ('heart.csv', 'HeartDisease', 'heart'),
        ('stroke.csv', 'stroke', 'stroke'),
        ('cirrhosis.csv', 'Stage', 'cirrhosis'),
    ]

    all_results = {}

    for csv_file, target_col, name in datasets:
        if not os.path.exists(csv_file):
            logger.warning(f"File does not exist: {csv_file}, skipping")
            continue

        try:
            X, y, _ = prepare_data(csv_file, target_col)
            evaluator = AdaptiveWeightedEnsemble(dataset_name=name)
            results = evaluator.run_full_pipeline(X, y)

            # Print results
            print(f"\n{'='*60}")
            print(f"{name.upper()} AWELM Results Summary")
            print(f"{'='*60}")
            print(f"{'Model':<15} {'AUC':<8} {'F1':<8} {'Accuracy':<8} {'Precision':<8} {'Recall':<8}")
            print('-' * 60)

            for model_name, res in results['base_models'].items():
                print(f"{model_name:<15} {res.get('auc', 0):.4f}    "
                      f"{res['f1']:.4f}    {res['accuracy']:.4f}    "
                      f"{res['precision']:.4f}    {res['recall']:.4f}")

            ens = results['ensemble']
            print('-' * 60)
            print(f"{'Ensemble Model':<15} {ens.get('auc', 0):.4f}    "
                  f"{ens['f1']:.4f}    {ens['accuracy']:.4f}    "
                  f"{ens['precision']:.4f}    {ens['recall']:.4f}")

            print(f"\nOptimal weights: {results['weights']}")

            all_results[name] = results

        except Exception as e:
            logger.error(f"{name} AWELM failed: {e}")

    return all_results


if __name__ == '__main__':
    results = run_all_datasets()
