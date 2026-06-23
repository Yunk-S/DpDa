"""
Problem 3: Bayesian Network Multi-Disease Association Probability Model (BNMDAP)
========================================
Objective: Quantify comorbidity associations between diseases, predict probability of multiple diseases occurring simultaneously

Method:
1. Network structure definition: nodes=diseases+risk factors, edges=conditional dependencies
2. Conditional Probability Table (CPT) estimation: based on data + medical prior knowledge
3. Joint probability calculation: P(A,B,C) = P(A)·P(B|A)·P(C|A,B)
4. Association strength quantification: Relative Risk (RR), Conditional Probability Ratio (CPR)

Core Formulas:
  - Conditional probability: P(A|B) = P(A,B) / P(B)
  - Bayes formula: P(B|A) = P(A|B)·P(B) / P(A)
  - Joint probability: P(A,B) = P(A)·P(B|A)
  - Relative Risk: RR = P(Disease| RiskFactor) / P(Disease)
  - Conditional Probability Ratio: CPR = P(A|B) / P(A|not B)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from collections import defaultdict
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BayesianDiseaseNetwork:
    """
    Bayesian Network Multi-Disease Association Probability Model (BNMDAP)

    Network Structure:
        Hypertension (Hypertension node)
            ↓
        HeartDisease (Heart Disease node)
            ↓
        Stroke (Stroke node)

        LiverDisease → Stroke (Cirrhosis → Stroke)

    Disease Association Matrix (based on medical prior knowledge):
        - Stroke-Heart Disease: RR ≈ 2.5 (strong positive correlation)
        - Stroke-Cirrhosis: RR ≈ 1.8 (moderate positive correlation)
        - Heart Disease-Cirrhosis: RR ≈ 1.5 (weak positive correlation)
        - Hypertension-Heart Disease: RR ≈ 2.8 (strong positive correlation)
        - Hypertension-Stroke: RR ≈ 3.2 (strong positive correlation)
    """

    def __init__(self):
        self.network_structure = {
            'hypertension': {
                'parents': [],
                'children': ['heart_disease', 'stroke'],
                'prior': None,
                'cpt': {}
            },
            'heart_disease': {
                'parents': ['hypertension'],
                'children': ['stroke'],
                'prior': None,
                'cpt': {}
            },
            'stroke': {
                'parents': ['heart_disease', 'hypertension'],
                'children': [],
                'prior': None,
                'cpt': {}
            }
        }

        # Relative Risk (RR) based on epidemiological literature
        self.relative_risk = {
            ('hypertension', 'heart_disease'): 2.8,
            ('hypertension', 'stroke'): 3.2,
            ('heart_disease', 'stroke'): 2.5,
            ('liver_disease', 'stroke'): 1.8,
            ('heart_disease', 'cirrhosis'): 1.5,
            ('hypertension', 'cirrhosis'): 1.3,
        }

        self.disease_base_rates = {
            'stroke': 0.028,      # ~2.8%
            'heart_disease': 0.047,  # ~4.7%
            'cirrhosis': 0.010,   # ~1.0%
            'hypertension': 0.300,  # ~30%
        }

    # ------------------------------------------------------------------
    # 1. Estimate Prior Probability from Data
    # ------------------------------------------------------------------
    def estimate_prior_from_data(self, stroke_df=None, heart_df=None, cirrhosis_df=None):
        """
        Estimate prior probability for each disease from datasets

        Parameters:
            stroke_df: Stroke dataset
            heart_df: Heart disease dataset
            cirrhosis_df: Cirrhosis dataset
        """
        priors = {}

        if stroke_df is not None:
            stroke_rate = stroke_df['stroke'].mean()
            priors['stroke'] = stroke_rate
            logger.info(f"Stroke prior probability (data estimated): {stroke_rate:.4f}")

        if heart_df is not None:
            heart_rate = heart_df['HeartDisease'].mean()
            priors['heart_disease'] = heart_rate
            logger.info(f"Heart disease prior probability (data estimated): {heart_rate:.4f}")

        if cirrhosis_df is not None:
            # Proportion of advanced cirrhosis (Stage 3-4)
            cirrhosis_rate = (cirrhosis_df['Stage'] >= 3).mean()
            priors['cirrhosis'] = cirrhosis_rate
            logger.info(f"Advanced cirrhosis prior probability (data estimated): {cirrhosis_rate:.4f}")

        # Estimate using hypertension feature
        if stroke_df is not None and 'hypertension' in stroke_df.columns:
            hyper_rate = stroke_df['hypertension'].mean()
            priors['hypertension'] = hyper_rate
            logger.info(f"Hypertension prior probability (data estimated): {hyper_rate:.4f}")

        self.disease_base_rates.update(priors)
        return priors

    # ------------------------------------------------------------------
    # 2. Estimate Conditional Probability Table (CPT)
    # ------------------------------------------------------------------
    def estimate_cpt(self):
        """
        Estimate Conditional Probability Table based on Bayes theorem and relative risk

        Core Formula:
            P(B|A) = RR(A,B) × P(B)
            P(B|A) = min(1.0, RR × base_rate)

        Where RR is Relative Risk (from literature)
        """
        cpt = {}

        for disease, info in self.network_structure.items():
            parents = info['parents']
            prior = self.disease_base_rates.get(disease, 0.5)

            if len(parents) == 0:
                # No parent nodes, use prior probability
                cpt[disease] = {(): prior}
            elif len(parents) == 1:
                # Single parent node
                parent = parents[0]
                p_parent = self.disease_base_rates.get(parent, 0.5)
                rr_key = (parent, disease)

                rr = self.relative_risk.get(rr_key, self.relative_risk.get(
                    (disease, parent), 2.0
                ))

                # P(disease | parent) = RR × P(disease)
                p_disease_given_parent = min(0.99, rr * prior)
                p_disease_given_not_parent = min(prior, prior / rr + 0.01)

                p_not_parent = 1 - p_parent
                p_not_disease_given_parent = 1 - p_disease_given_parent
                p_not_disease_given_not_parent = 1 - p_disease_given_not_parent

                cpt[disease] = {
                    (1,): p_disease_given_parent,
                    (0,): p_disease_given_not_parent,
                    'p_parent': p_parent,
                    'p_not_parent': p_not_parent,
                    'p_not_disease_given_parent': p_not_disease_given_parent,
                    'p_not_disease_given_not_parent': p_not_disease_given_not_parent,
                }

                logger.info(f"{disease} CPT: P(d|parent)={p_disease_given_parent:.3f}, "
                           f"P(d|~parent)={p_disease_given_not_parent:.3f}, RR={rr:.2f}")

        self.cpt = cpt
        return cpt

    # ------------------------------------------------------------------
    # 3. Joint Probability Calculation
    # ------------------------------------------------------------------
    def joint_probability(self, htn, hd, strk, cirr):
        """
        Calculate joint probability for given state

        P(H, HD, S, C) = P(H) · P(HD|H) · P(S|HD,H) · P(C|...)

        Use Conditional Probability Table to calculate joint distribution
        """
        P = self.disease_base_rates

        # P(Hypertension)
        p_htn = P['hypertension'] if htn else (1 - P['hypertension'])

        # P(HeartDisease | Hypertension)
        if self.cpt.get('heart_disease'):
            cpt_hd = self.cpt['heart_disease']
            p_hd_given_htn = cpt_hd[(1,)] if htn else cpt_hd[(0,)]
        else:
            base_hd = P.get('heart_disease', 0.5)
            rr = self.relative_risk.get(('hypertension', 'heart_disease'), 2.8)
            if htn:
                p_hd_given_htn = min(0.99, rr * base_hd)
            else:
                p_hd_given_htn = min(base_hd, base_hd / rr)

        p_hd = p_hd_given_htn if hd else (1 - p_hd_given_htn)

        # P(Stroke | HeartDisease, Hypertension)
        base_stroke = P.get('stroke', 0.5)
        rr_hd = self.relative_risk.get(('heart_disease', 'stroke'), 2.5)
        rr_htn = self.relative_risk.get(('hypertension', 'stroke'), 3.2)

        if hd and htn:
            # Both risk factors present, multiplicative risk stacking (with saturation)
            rr_combined = rr_hd * rr_htn ** 0.7
            p_strk_given = min(0.95, rr_combined * base_stroke)
        elif hd:
            p_strk_given = min(0.99, rr_hd * base_stroke)
        elif htn:
            p_strk_given = min(0.99, rr_htn * base_stroke)
        else:
            p_strk_given = base_stroke / (rr_hd * 0.7)

        p_strk = p_strk_given if strk else (1 - p_strk_given)

        # P(Cirrhosis)
        base_cirr = P.get('cirrhosis', 0.5)
        rr_cirr = self.relative_risk.get(('heart_disease', 'cirrhosis'), 1.5)
        if hd:
            p_cirr_given = min(0.95, rr_cirr * base_cirr)
        else:
            p_cirr_given = base_cirr / rr_cirr

        p_cirr = p_cirr_given if cirr else (1 - p_cirr_given)

        # Joint probability
        joint_prob = p_htn * p_hd * p_strk * p_cirr

        return {
            'joint_probability': joint_prob,
            'P_hypertension': p_htn,
            'P_heart_disease': p_hd,
            'P_stroke': p_strk,
            'P_cirrhosis': p_cirr,
            'P_stroke_given_heart': p_strk_given if htn else 0,
        }

    # ------------------------------------------------------------------
    # 4. Prediction Interface
    # ------------------------------------------------------------------
    def predict(self, hypertension=0, heart_disease=0, cirrhosis=0):
        """
        Predict multi-disease association probability

        Parameters:
            hypertension: Has hypertension (0/1)
            heart_disease: Has heart disease (0/1)
            cirrhosis: Has cirrhosis (0/1)

        Returns:
            Dictionary of probabilities for various disease combinations
        """
        P = self.disease_base_rates

        # Independent probabilities for each disease
        p_htn = P['hypertension']
        p_hd = P['heart_disease']
        p_strk = P['stroke']
        p_cirr = P['cirrhosis']

        # Adjustments considering correlations
        rr_hd_htn = self.relative_risk.get(('hypertension', 'heart_disease'), 2.8)
        rr_strk_hd = self.relative_risk.get(('heart_disease', 'stroke'), 2.5)
        rr_strk_htn = self.relative_risk.get(('hypertension', 'stroke'), 3.2)
        rr_cirr_hd = self.relative_risk.get(('heart_disease', 'cirrhosis'), 1.5)

        def adj_prob(base_p, rr):
            return min(0.99, rr * base_p)

        # Adjusted probabilities
        p_hd_adj = adj_prob(p_hd, rr_hd_htn) if hypertension else p_hd

        rr_combined_strk = rr_strk_hd * (rr_strk_htn ** 0.7) if hypertension else rr_strk_hd
        p_strk_adj = adj_prob(p_strk, rr_combined_strk) if heart_disease else (
            adj_prob(p_strk, rr_strk_htn) if hypertension else p_strk
        )

        p_cirr_adj = adj_prob(p_cirr, rr_cirr_hd) if heart_disease else p_cirr

        # Calculate various combination probabilities
        results = {
            # Single disease probability
            'stroke': p_strk_adj if hypertension == 0 and heart_disease == 0 else (
                p_strk_adj if hypertension == 1 else p_strk_adj
            ),
            'heart_disease': p_hd_adj if hypertension == 0 else p_hd_adj,
            'cirrhosis': p_cirr_adj if heart_disease == 0 else p_cirr_adj,
            'hypertension': p_htn,

            # Single disease (excluding others)
            'stroke_only': p_strk_adj * (1 - p_hd_adj * 0.5) * (1 - p_cirr_adj * 0.3),
            'heart_only': p_hd_adj * (1 - p_htn * 0.2) * (1 - p_cirr_adj * 0.3),
            'cirrhosis_only': p_cirr_adj * (1 - p_hd_adj * 0.2),

            # Two-disease combinations
            'stroke_heart': min(p_strk_adj, p_hd_adj) * 1.5,
            'stroke_cirrhosis': min(p_strk_adj, p_cirr_adj) * 1.3,
            'heart_cirrhosis': min(p_hd_adj, p_cirr_adj) * 1.2,

            # Three diseases
            'all_three': min(p_strk_adj, p_hd_adj, p_cirr_adj) * 1.8,

            # Healthy (no disease)
            'none': (1 - p_strk_adj) * (1 - p_hd_adj) * (1 - p_cirr_adj),
        }

        # Ensure all probabilities are within [0, 1]
        for key in results:
            results[key] = max(0, min(1, results[key]))

        # Normalize
        total = sum(results.values())
        if total > 0:
            for key in results:
                results[key] /= total

        return results

    # ------------------------------------------------------------------
    # 5. Relative Risk Calculation
    # ------------------------------------------------------------------
    def calculate_relative_risk(self, df, risk_factor_col, disease_col):
        """
        Calculate Relative Risk (RR) from data

        RR = P(disease | risk_factor=1) / P(disease | risk_factor=0)

        Parameters:
            df: Data DataFrame
            risk_factor_col: Risk factor column name
            disease_col: Disease column name

        Returns:
            rr: Relative Risk
        """
        if risk_factor_col not in df.columns or disease_col not in df.columns:
            return None

        p_disease_given_risk = df[df[risk_factor_col] == 1][disease_col].mean()
        p_disease_given_no_risk = df[df[risk_factor_col] == 0][disease_col].mean()

        if p_disease_given_no_risk == 0:
            return None

        rr = p_disease_given_risk / p_disease_given_no_risk
        return rr

    def calculate_conditional_prob_ratio(self, df, factor_col, disease_col):
        """
        Calculate Conditional Probability Ratio (CPR)

        CPR = P(disease | factor=1) / P(disease | factor=0)

        This is an effective indicator for measuring the impact of risk factors
        """
        if factor_col not in df.columns or disease_col not in df.columns:
            return None

        p_disease_given = df[df[factor_col] == 1][disease_col].mean()
        p_disease_given_not = df[df[factor_col] == 0][disease_col].mean()

        if p_disease_given_not == 0:
            return None

        cpr = p_disease_given / p_disease_given_not
        return cpr

    # ------------------------------------------------------------------
    # 6. Visualization
    # ------------------------------------------------------------------
    def plot_disease_network(self, save_path=None):
        """
        Draw Bayesian network structure diagram (using matplotlib arrows)
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 8)
        ax.axis('off')

        # Node positions
        nodes = {
            'hypertension': (2, 6),
            'heart_disease': (5, 4),
            'stroke': (8, 2),
            'cirrhosis': (5, 0),
        }

        # Node attributes
        node_colors = {
            'hypertension': '#FF6B6B',
            'heart_disease': '#E74C3C',
            'stroke': '#3498DB',
            'cirrhosis': '#F39C12',
        }

        node_labels = {
            'hypertension': 'Hypertension\nP=30%',
            'heart_disease': 'Heart Disease\nP=4.7%',
            'stroke': 'Stroke\nP=2.8%',
            'cirrhosis': 'Cirrhosis\nP=1.0%',
        }

        # Draw arrows (edges)
        edges = [
            ('hypertension', 'heart_disease', 'RR=2.8'),
            ('hypertension', 'stroke', 'RR=3.2'),
            ('heart_disease', 'stroke', 'RR=2.5'),
            ('heart_disease', 'cirrhosis', 'RR=1.5'),
        ]

        for src, dst, label in edges:
            x1, y1 = nodes[src]
            x2, y2 = nodes[dst]

            # Draw arrow
            ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color='#555',
                    lw=2, connectionstyle='arc3,rad=0.1'
                )
            )

            # Annotate RR value
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.3
            ax.text(mid_x, mid_y, label, fontsize=9,
                    ha='center', color='#555',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Draw nodes
        for node, (x, y) in nodes.items():
            circle = plt.Circle((x, y), 0.8, color=node_colors[node], alpha=0.85, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, node_labels[node], ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white',
                    multialignment='center', zorder=11)

        ax.set_title('Bayesian Network Multi-Disease Association Model Structure\n'
                     'Bayesian Network Multi-Disease Association Model',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_disease_correlation_heatmap(self, save_path=None):
        """
        Draw disease correlation heatmap (RR matrix)
        """
        diseases = ['stroke', 'heart_disease', 'cirrhosis', 'hypertension']
        rr_matrix = np.ones((4, 4))

        for i, d1 in enumerate(diseases):
            for j, d2 in enumerate(diseases):
                if i != j:
                    key = (d1, d2)
                    rev_key = (d2, d1)
                    rr_matrix[i, j] = self.relative_risk.get(key,
                        self.relative_risk.get(rev_key, 1.0))

        disease_labels = ['Stroke', 'Heart Disease', 'Cirrhosis', 'Hypertension']

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(rr_matrix, annot=True, fmt='.2f',
                    xticklabels=disease_labels, yticklabels=disease_labels,
                    cmap='YlOrRd', ax=ax, vmin=1, vmax=4,
                    linewidths=0.5, linecolor='white')

        ax.set_title('Relative Risk Ratio (RR) Heatmap Between Diseases\n'
                     'Relative Risk Ratio Between Diseases',
                     fontsize=13, fontweight='bold')

        # Add color legend
        fig.text(0.5, 0.01,
                 'RR > 2: Strong positive correlation | RR 1.5-2: Moderate positive correlation | RR < 1.5: Weak correlation',
                 ha='center', fontsize=9, style='italic', color='gray')

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_comorbidity_probability(self, save_path=None):
        """
        Draw comorbidity probability bar chart
        """
        scenarios = [
            ('No risk factors', 0, 0, 0),
            ('Hypertension only', 1, 0, 0),
            ('Heart disease only', 0, 1, 0),
            ('Hypertension + Heart disease', 1, 1, 0),
            ('All three diseases', 1, 1, 1),
        ]

        labels = []
        stroke_probs = []
        heart_probs = []
        cirrh_probs = []

        for label, htn, hd, cirr in scenarios:
            result = self.predict(hypertension=htn, heart_disease=hd, cirrhosis=cirr)
            labels.append(label)
            stroke_probs.append(result['stroke'] * 100)
            heart_probs.append(result['heart_disease'] * 100)
            cirrh_probs.append(result['cirrhosis'] * 100)

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width, stroke_probs, width, label='Stroke', color='#3498DB')
        bars2 = ax.bar(x, heart_probs, width, label='Heart Disease', color='#E74C3C')
        bars3 = ax.bar(x + width, cirrh_probs, width, label='Cirrhosis', color='#F39C12')

        ax.set_ylabel('Disease Probability (%)', fontsize=11)
        ax.set_xlabel('Risk Factor Combinations', fontsize=11)
        ax.set_title('Disease Prediction Probability Under Different Risk Factor Combinations\n'
                     'Disease Probability Under Different Risk Factor Combinations',
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Annotate values
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.5:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                            f'{height:.1f}', ha='center', fontsize=7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def run_full_analysis(self, stroke_df=None, heart_df=None, cirrhosis_df=None):
        """
        Run complete Bayesian network analysis
        """
        logger.info("=== Starting Bayesian Network Multi-Disease Association Analysis ===")

        # Estimate prior probabilities
        self.estimate_prior_from_data(stroke_df, heart_df, cirrhosis_df)

        # Estimate conditional probability table
        self.estimate_cpt()

        # Visualization
        logger.info("Generating visualization charts...")
        self.plot_disease_network(
            save_path='output/figures/bnmdap_network_structure.png'
        )
        self.plot_disease_correlation_heatmap(
            save_path='output/figures/bnmdap_correlation_heatmap.png'
        )
        self.plot_comorbidity_probability(
            save_path='output/figures/bnmdap_comorbidity_probability.png'
        )

        # Example predictions
        logger.info("\nExample prediction results:")
        scenarios = [
            ('Low-risk population (no known risk factors)', 0, 0, 0),
            ('Hypertension patient', 1, 0, 0),
            ('Heart disease patient', 0, 1, 0),
            ('Hypertension + Heart disease patient (comorbidity)', 1, 1, 0),
            ('All three diseases patient', 1, 1, 1),
        ]

        for label, htn, hd, cirr in scenarios:
            result = self.predict(hypertension=htn, heart_disease=hd, cirrhosis=cirr)
            logger.info(f"\n  [{label}]")
            logger.info(f"    Stroke probability: {result['stroke']*100:.2f}%")
            logger.info(f"    Heart disease probability: {result['heart_disease']*100:.2f}%")
            logger.info(f"    Cirrhosis probability: {result['cirrhosis']*100:.2f}%")
            logger.info(f"    Stroke + Heart disease comorbidity: {result['stroke_heart']*100:.2f}%")
            logger.info(f"    All three diseases comorbidity: {result['all_three']*100:.2f}%")

        logger.info("\n=== Bayesian Network Analysis Complete ===")
        return self


def run_all_datasets():
    """Run complete Bayesian network analysis on three datasets"""
    dfs = {}

    for csv_file, name in [
        ('heart.csv', 'heart'),
        ('stroke.csv', 'stroke'),
        ('cirrhosis.csv', 'cirrhosis')
    ]:
        if os.path.exists(csv_file):
            dfs[name] = pd.read_csv(csv_file)
            logger.info(f"Loaded {csv_file}")

    stroke_df = dfs.get('stroke')
    heart_df = dfs.get('heart')
    cirrhosis_df = dfs.get('cirrhosis')

    # Run analysis
    network = BayesianDiseaseNetwork()
    network.run_full_analysis(
        stroke_df=stroke_df,
        heart_df=heart_df,
        cirrhosis_df=cirrhosis_df
    )

    return network


if __name__ == '__main__':
    network = run_all_datasets()
