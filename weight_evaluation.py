"""
Problem 1: Multi-Factor Impact Weight Evaluation Model
==================================
Objective: Quantify the impact of each factor on disease risk, identify key risk factors
Method: Combine statistical significance and predictive contribution to build a comprehensive weight evaluation model

Comprehensive Weight Formula:
  W_j = |β_j_full| × (1 - p_j) × (1 + ΔAUC_j)
  Where:
    - |β_j_full|: Absolute value of multivariate regression coefficient
    - p_j: p-value (statistical significance)
    - ΔAUC_j: AUC drop after removing feature j (independent contribution to model performance)

Normalized Weight:
  W_normalized = 100 × W_j / ΣW_j
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import glm
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FeatureWeightEvaluator:
    """
    Multi-Factor Impact Weight Evaluation Model

    Comprehensive Weight Formula:
        W_j = |β_j_full| × (1 - p_j) × (1 + ΔAUC_j)

    Includes:
        1. Data preprocessing
        2. Univariate analysis (univariate AUC, OR, regression coefficient)
        3. Multivariate analysis (adjusted regression coefficient, ΔAUC contribution)
        4. Comprehensive weight calculation and ranking
        5. Key feature selection (cumulative impact curve)
    """

    def __init__(self, dataset_name='disease'):
        self.dataset_name = dataset_name
        self.results = {}
        self.weights = None
        self.summary = None

    # ------------------------------------------------------------------
    # 1. Data Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, df, target_col, categorical_cols=None, numeric_cols=None):
        """
        Data preprocessing

        Parameters:
            df: Raw DataFrame
            target_col: Target variable column name
            categorical_cols: List of categorical feature column names
            numeric_cols: List of numeric feature column names

        Returns:
            X: Processed feature matrix
            y: Target variable
            feature_names: List of feature names
        """
        data = df.copy()
        self.target_col = target_col
        self.original_feature_names = []

        # Handle target variable
        if data[target_col].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(data[target_col])
        else:
            y = data[target_col].values

        # Separate features
        if target_col in data.columns:
            X_raw = data.drop(columns=[target_col])

        # Auto-detect column types
        if categorical_cols is None:
            categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        if numeric_cols is None:
            numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

        # Record original feature names
        self.original_feature_names = categorical_cols + numeric_cols

        # Handle missing values
        if X_raw[categorical_cols + numeric_cols].isnull().sum().sum() > 0:
            for col in numeric_cols:
                if X_raw[col].isnull().any():
                    X_raw[col].fillna(X_raw[col].median(), inplace=True)
            for col in categorical_cols:
                if X_raw[col].isnull().any():
                    X_raw[col].fillna(X_raw[col].mode()[0], inplace=True)

        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

        # Standardize numeric features
        if numeric_cols:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(X_raw[numeric_cols])
            for i, col in enumerate(numeric_cols):
                X_raw[col] = scaled[:, i]

        X = X_raw[categorical_cols + numeric_cols].values
        feature_names = categorical_cols + numeric_cols

        logger.info(f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names

    # ------------------------------------------------------------------
    # 2. Univariate Analysis
    # ------------------------------------------------------------------
    def univariate_analysis(self, X, y, feature_names):
        """
        Perform univariate logistic regression analysis for each feature

        Calculate: regression coefficient (β), p-value, odds ratio (OR), AUC (univariate model predictive ability)

        Parameters:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: List of feature names

        Returns:
            univariate_results: DataFrame with univariate analysis results
        """
        results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for i, name in enumerate(feature_names):
            X_i = X[:, i].reshape(-1, 1)

            # Use statsmodels for logistic regression to get coefficients and p-values
            try:
                X_with_const = sm.add_constant(X_i)
                model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=200)
                coef = model.params[1]
                pvalue = model.pvalues[1]
                odds_ratio = np.exp(coef)
            except Exception as e:
                logger.warning(f"Feature {name} logistic regression failed: {e}")
                coef, pvalue, odds_ratio = 0, 1.0, 1.0

            # Use sklearn to calculate univariate AUC
            try:
                auc_scores = cross_val_score(
                    LogisticRegression(max_iter=500, random_state=42),
                    X_i, y, cv=skf, scoring='roc_auc'
                )
                auc = auc_scores.mean()
            except Exception:
                auc = 0.5

            results.append({
                'Feature': name,
                'Coefficient': coef,
                'P_Value': pvalue,
                'Odds_Ratio': odds_ratio,
                'AUC': auc,
                'Significant': pvalue < 0.05
            })

        df = pd.DataFrame(results)
        df = pd.DataFrame({
            'Feature': [str(r['Feature']) for r in results],
            'Coefficient': [r['Coefficient'] for r in results],
            'P_Value': [r['P_Value'] for r in results],
            'Odds_Ratio': [r['Odds_Ratio'] for r in results],
            'AUC': [r['AUC'] for r in results],
            'Significant': [r['Significant'] for r in results],
        })

        # Multiple testing correction (Bonferroni)
        reject, pvals_corrected, _, _ = multipletests(
            df['P_Value'].values, method='bonferroni'
        )
        df['P_Value_Corrected'] = pvals_corrected
        df['Significant_Corrected'] = reject

        logger.info("Univariate analysis complete")
        self.univariate_results = df
        return df

    # ------------------------------------------------------------------
    # 3. Multivariate Analysis
    # ------------------------------------------------------------------
    def multivariate_analysis(self, X, y, feature_names):
        """
        Build multivariate logistic regression model, calculate adjusted regression coefficients

        Calculate the independent contribution of each feature to model performance (ΔAUC)

        Parameters:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names

        Returns:
            multivariate_results: Multivariate analysis results
            full_auc: Full model AUC
            delta_aucs: ΔAUC after removing each feature
        """
        # Full model
        X_with_const = sm.add_constant(X)
        full_model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=500)

        # Calculate full model AUC
        full_prob = full_model.predict(X_with_const)
        full_auc = roc_auc_score(y, full_prob)

        # Get coefficients and p-values
        coefs = full_model.params[1:]  # Skip intercept
        pvalues = full_model.pvalues[1:]
        conf_int_arr = full_model.conf_int()[1:]  # skip intercept row, shape (n_features, 2)
        assert len(feature_names) == len(coefs) == len(pvalues) == len(conf_int_arr), \
            f"Length mismatch: fn={len(feature_names)}, coef={len(coefs)}, pval={len(pvalues)}, ci={len(conf_int_arr)}"

        multivariate_results = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient_Full': np.asarray(coefs),
            'P_Value_Full': np.asarray(pvalues),
            'CI_Lower': conf_int_arr[:, 0],
            'CI_Upper': conf_int_arr[:, 1],
        })

        # Calculate ΔAUC (AUC drop after feature removal)
        delta_aucs = []
        for i, name in enumerate(feature_names):
            X_reduced = np.delete(X, i, axis=1)
            try:
                X_red_const = sm.add_constant(X_reduced)
                reduced_model = sm.Logit(y, X_red_const).fit(disp=0, maxiter=200)
                reduced_prob = reduced_model.predict(X_red_const)
                reduced_auc = roc_auc_score(y, reduced_prob)
                delta_auc = full_auc - reduced_auc
            except Exception:
                delta_auc = 0
            delta_aucs.append(delta_auc)

        multivariate_results['Delta_AUC'] = delta_aucs
        multivariate_results['Full_AUC'] = full_auc

        logger.info(f"Multivariate analysis complete, Full AUC: {full_auc:.4f}")
        self.multivariate_results = multivariate_results
        self.full_auc = full_auc
        return multivariate_results, full_auc, delta_aucs

    # ------------------------------------------------------------------
    # 4. Comprehensive Weight Calculation
    # ------------------------------------------------------------------
    def calculate_weights(self):
        """
        Comprehensive Weight Formula:
            W_j = |β_j_full| × (1 - p_j) × (1 + ΔAUC_j)

        Normalized Weight (percentage form):
            W_normalized = 100 × W_j / ΣW_j
        """
        if not hasattr(self, 'univariate_results') or not hasattr(self, 'multivariate_results'):
            raise ValueError("Please run univariate and multivariate analysis first")

        uni = self.univariate_results
        multi = self.multivariate_results

        # Merge results
        merged = multi.merge(
            uni[['Feature', 'AUC', 'Odds_Ratio', 'Significant']],
            on='Feature'
        )

        # Comprehensive weight calculation
        abs_coef = np.abs(merged['Coefficient_Full'].values)
        p_vals = merged['P_Value_Full'].values
        delta_auc = merged['Delta_AUC'].values

        # Avoid p=0 causing (1-p)=1 issue, handle very small values
        p_vals = np.clip(p_vals, 1e-10, 1.0)

        # W_j = |β| × (1 - p) × (1 + ΔAUC)
        raw_weights = abs_coef * (1 - p_vals) * (1 + delta_auc)

        # Normalize
        normalized_weights = 100 * raw_weights / raw_weights.sum()

        merged['Raw_Weight'] = raw_weights
        merged['Normalized_Weight_Pct'] = normalized_weights

        # Sort by weight in descending order
        merged = merged.sort_values('Normalized_Weight_Pct', ascending=False).reset_index(drop=True)

        # Cumulative weight
        merged['Cumulative_Weight_Pct'] = merged['Normalized_Weight_Pct'].cumsum()

        self.weights = merged
        self.summary = merged[['Feature', 'Coefficient_Full', 'P_Value_Full',
                              'Odds_Ratio', 'AUC', 'Delta_AUC',
                              'Raw_Weight', 'Normalized_Weight_Pct',
                              'Cumulative_Weight_Pct', 'Significant']]

        logger.info("Weight calculation complete")
        return self.summary

    # ------------------------------------------------------------------
    # 5. Key Feature Selection (Cumulative Impact Curve)
    # ------------------------------------------------------------------
    def plot_cumulative_weight(self, threshold=80, save_path=None):
        """
        Plot cumulative weight curve, identify features needed to reach threshold% cumulative weight

        Parameters:
            threshold: Cumulative weight threshold, default 80%
            save_path: Save path

        Returns:
            key_features: List of key features
        """
        if self.weights is None:
            raise ValueError("Please run weight calculation first")

        df = self.weights.sort_values('Normalized_Weight_Pct', ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        features = df['Feature'].values
        cum_weights = df['Cumulative_Weight_Pct'].values
        weights = df['Normalized_Weight_Pct'].values

        # Bar chart + cumulative line
        bars = ax.bar(range(len(features)), weights, color='steelblue', alpha=0.7, label='Individual Weight')
        ax2 = ax.twinx()
        ax2.plot(range(len(features)), cum_weights, 'ro-', linewidth=2, markersize=6, label='Cumulative Weight')

        # Threshold line
        ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=1.5,
                    label=f'{threshold}% Threshold')

        # Annotate key points
        for i, (feat, cw, w) in enumerate(zip(features, cum_weights, weights)):
            ax.text(i, w + 0.5, f'{w:.1f}%', ha='center', fontsize=8, color='steelblue')
            ax2.text(i, cw + 1.5, f'{cw:.1f}%', ha='center', fontsize=7,
                      color='red', rotation=90, va='bottom')

        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Weight (%)', fontsize=11)
        ax2.set_ylabel('Cumulative Weight (%)', fontsize=11)
        ax.set_xlabel('Feature', fontsize=11)
        ax.set_title(f'{self.dataset_name} - Feature Weight and Cumulative Impact', fontsize=13, fontweight='bold')

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Extract key features
        key_mask = cum_weights <= threshold
        n_key = np.sum(key_mask) + 1
        key_features = list(features[:n_key])

        logger.info(f"Key features (explaining first {threshold}% weight): {key_features}")
        return key_features

    def plot_feature_weights(self, top_n=15, save_path=None):
        """
        Plot feature weight bar chart (horizontal)
        """
        if self.weights is None:
            raise ValueError("Please run weight calculation first")

        df = self.weights.sort_values('Normalized_Weight_Pct').tail(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))

        bars = ax.barh(df['Feature'], df['Normalized_Weight_Pct'], color=colors)

        # Add value labels
        for bar, val in zip(bars, df['Normalized_Weight_Pct']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)

        ax.set_xlabel('Weight (%)', fontsize=11)
        ax.set_title(f'{self.dataset_name} - Feature Weight Analysis (Top {top_n})',
                      fontsize=13, fontweight='bold')
        ax.set_xlim(0, df['Normalized_Weight_Pct'].max() * 1.15)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_heatmap_comparison(self, save_path=None):
        """
        Plot univariate AUC vs multivariate weight heatmap comparison
        """
        if self.weights is None:
            raise ValueError("Please run weight calculation first")

        df = self.weights.sort_values('Normalized_Weight_Pct', ascending=False)

        plot_data = df[['Feature', 'AUC', 'Normalized_Weight_Pct']].set_index('Feature')
        plot_data.columns = ['Univariate AUC', 'Comprehensive Weight (%)']

        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.4)))

        sns.heatmap(plot_data[['Univariate AUC']], annot=True, fmt='.3f',
                    cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'AUC'})
        axes[0].set_title('Univariate Predictive Ability (AUC)', fontsize=12)

        sns.heatmap(plot_data[['Comprehensive Weight (%)']], annot=True, fmt='.1f',
                    cmap='YlGnBu', ax=axes[1], cbar_kws={'label': 'Weight (%)'})
        axes[1].set_title('Comprehensive Weight Evaluation', fontsize=12)

        plt.suptitle(f'{self.dataset_name} - Multi-Dimensional Weight Evaluation Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def run_full_pipeline(self, df, target_col, dataset_name=None,
                          categorical_cols=None, numeric_cols=None,
                          top_n_display=15):
        """
        Run complete weight evaluation pipeline

        Parameters:
            df: Raw data
            target_col: Target column name
            dataset_name: Dataset name
            categorical_cols: Categorical features
            numeric_cols: Numeric features
            top_n_display: Display top N features

        Returns:
            summary: Comprehensive weight results table
        """
        if dataset_name:
            self.dataset_name = dataset_name

        logger.info(f"=== Starting {self.dataset_name} Weight Evaluation ===")

        # 1. Preprocess
        X, y, feature_names = self.preprocess(
            df, target_col, categorical_cols, numeric_cols
        )

        # 2. Univariate analysis
        logger.info("Step 1: Univariate logistic regression analysis...")
        self.univariate_analysis(X, y, feature_names)

        # 3. Multivariate analysis
        logger.info("Step 2: Multivariate logistic regression analysis + ΔAUC calculation...")
        self.multivariate_analysis(X, y, feature_names)

        # 4. Comprehensive weight
        logger.info("Step 3: Calculate comprehensive weight...")
        summary = self.calculate_weights()

        # 5. Visualization
        logger.info("Step 4: Generate visualization charts...")
        self.plot_cumulative_weight(
            threshold=80,
            save_path=f'output/figures/{self.dataset_name}_cumulative_weight_curve.png'
        )
        self.plot_feature_weights(
            top_n=top_n_display,
            save_path=f'output/figures/{self.dataset_name}_feature_weights.png'
        )
        self.plot_heatmap_comparison(
            save_path=f'output/figures/{self.dataset_name}_weight_heatmap.png'
        )

        # Save results
        summary.to_excel(f'output/{self.dataset_name}_weights.xlsx', index=False)
        logger.info(f"=== {self.dataset_name} Weight Evaluation Complete ===\n")

        return summary


def run_all_datasets():
    """Run complete weight evaluation on three datasets"""
    results_all = {}

    datasets_info = [
        ('heart.csv', 'HeartDisease', 'heart'),
        ('stroke.csv', 'stroke', 'stroke'),
        ('cirrhosis.csv', 'Stage', 'cirrhosis'),
    ]

    for csv_file, target_col, name in datasets_info:
        if not os.path.exists(csv_file):
            logger.warning(f"File does not exist: {csv_file}, skipping")
            continue

        df = pd.read_csv(csv_file)
        evaluator = FeatureWeightEvaluator(dataset_name=name)

        # Auto-detect feature columns (exclude ID and other irrelevant columns)
        exclude_cols = ['id', 'ID', 'N_Days']
        feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        try:
            summary = evaluator.run_full_pipeline(
                df, target_col,
                dataset_name=name,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                top_n_display=15
            )
            results_all[name] = summary
            print(f"\n{'='*60}")
            print(f"{name.upper()} Feature Weight Results (Top 10):")
            print(f"{'='*60}")
            print(summary[['Feature', 'Normalized_Weight_Pct', 'Cumulative_Weight_Pct',
                          'AUC', 'Odds_Ratio', 'Significant']].head(10).to_string(index=False))
        except Exception as e:
            logger.error(f"{name} Weight evaluation failed: {e}")

    return results_all


if __name__ == '__main__':
    results = run_all_datasets()
