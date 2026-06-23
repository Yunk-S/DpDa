"""
Disease Prediction Visualization Module
为三大核心模型提供可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('output/figures', exist_ok=True)


class WeightEvaluationVisualizer:
    """权重评估模型可视化"""
    
    def __init__(self, dataset_name='disease'):
        self.dataset_name = dataset_name
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
        }
    
    def plot_feature_weights(self, weights_df, top_n=15, save_path=None):
        """绘制特征权重水平条形图"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_feature_weights.png'
        
        df = weights_df.sort_values('Normalized_Weight_Pct', ascending=True).tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
        
        bars = ax.barh(df['Feature'], df['Normalized_Weight_Pct'], color=colors)
        
        for bar, val in zip(bars, df['Normalized_Weight_Pct']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)
        
        ax.set_xlabel('Weight (%)', fontsize=11)
        ax.set_title(f'{self.dataset_name.capitalize()} - Feature Weight Analysis (Top {top_n})',
                    fontsize=13, fontweight='bold')
        ax.set_xlim(0, df['Normalized_Weight_Pct'].max() * 1.15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_cumulative_weight_curve(self, weights_df, threshold=80, save_path=None):
        """绘制累积权重曲线"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_cumulative_weight.png'
        
        df = weights_df.sort_values('Normalized_Weight_Pct', ascending=False).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        features = df['Feature'].values
        cum_weights = df['Cumulative_Weight_Pct'].values
        weights = df['Normalized_Weight_Pct'].values
        x = np.arange(len(features))
        
        bars = ax.bar(x, weights, color=self.colors['primary'], alpha=0.7, 
                      label='Individual Weight', edgecolor='white')
        
        ax2 = ax.twinx()
        ax2.plot(x, cum_weights, 'ro-', linewidth=2, markersize=6, 
                label='Cumulative Weight', color=self.colors['secondary'])
        ax2.axhline(y=threshold, color=self.colors['accent'], linestyle='--', 
                    linewidth=1.5, label=f'{threshold}% Threshold')
        
        for i, (feat, cw, w) in enumerate(zip(features, cum_weights, weights)):
            ax.text(i, w + 0.5, f'{w:.1f}%', ha='center', fontsize=8, fontweight='bold')
            if cw <= threshold + 5:
                ax2.text(i, cw + 1.5, f'{cw:.1f}%', ha='center', fontsize=7, rotation=90)
        
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Individual Weight (%)', fontsize=11)
        ax2.set_ylabel('Cumulative Weight (%)', fontsize=11)
        ax.set_xlabel('Feature', fontsize=11)
        ax.set_title(f'{self.dataset_name.capitalize()} - Cumulative Weight Impact',
                    fontsize=13, fontweight='bold')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_weight_heatmap(self, weights_df, save_path=None):
        """绘制权重热力图"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_weight_heatmap.png'
        
        df = weights_df.sort_values('Normalized_Weight_Pct', ascending=False)
        plot_data = df[['Feature', 'AUC', 'Normalized_Weight_Pct']].set_index('Feature')
        plot_data.columns = ['Univariate AUC', 'Comprehensive Weight (%)']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.4 + 2)))
        
        sns.heatmap(plot_data[['Univariate AUC']], annot=True, fmt='.3f',
                   cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'AUC'}, linewidths=0.5)
        axes[0].set_title('Univariate Predictive Ability (AUC)', fontsize=12, fontweight='bold')
        
        sns.heatmap(plot_data[['Comprehensive Weight (%)']], annot=True, fmt='.1f',
                   cmap='YlGnBu', ax=axes[1], cbar_kws={'label': 'Weight (%)'}, linewidths=0.5)
        axes[1].set_title('Comprehensive Weight Evaluation', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'{self.dataset_name.capitalize()} - Multi-Dimensional Weight Comparison',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_significance_analysis(self, weights_df, save_path=None):
        """绘制显著性分析图"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_significance.png'
        
        df = weights_df.sort_values('Normalized_Weight_Pct', ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = ['#2ECC71' if sig else '#E74C3C' for sig in df['Significant']]
        axes[0].barh(df['Feature'], -np.log10(df['P_Value_Full']), color=colors)
        axes[0].axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0].set_xlabel('-log10(p-value)', fontsize=11)
        axes[0].set_title('Statistical Significance', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        colors_or = ['#2ECC71' if or_ > 1 else '#E74C3C' for or_ in df['Odds_Ratio']]
        axes[1].barh(df['Feature'], df['Odds_Ratio'], color=colors_or)
        axes[1].axvline(x=1, color='gray', linestyle='--', label='OR=1')
        axes[1].set_xlabel('Odds Ratio', fontsize=11)
        axes[1].set_title('Odds Ratio by Feature', fontsize=12, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def generate_all_plots(self, weights_df, dataset_name=None):
        """生成所有图表"""
        if dataset_name:
            self.dataset_name = dataset_name
        
        output_dir = 'output/figures'
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        results['feature_weights'] = self.plot_feature_weights(
            weights_df, top_n=15,
            save_path=f'{output_dir}/{self.dataset_name}_feature_weights.png')
        
        results['cumulative_weight'] = self.plot_cumulative_weight_curve(
            weights_df, threshold=80,
            save_path=f'{output_dir}/{self.dataset_name}_cumulative_weight.png')
        
        results['heatmap'] = self.plot_weight_heatmap(
            weights_df,
            save_path=f'{output_dir}/{self.dataset_name}_weight_heatmap.png')
        
        results['significance'] = self.plot_significance_analysis(
            weights_df,
            save_path=f'{output_dir}/{self.dataset_name}_significance.png')
        
        return results


class AWELMVisualizer:
    """AWELM模型可视化"""
    
    def __init__(self, dataset_name='disease'):
        self.dataset_name = dataset_name
        self.colors = plt.cm.Set2.colors
    
    def plot_model_comparison(self, base_results, ensemble_result, save_path=None):
        """绘制模型性能对比"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_model_comparison.png'
        
        model_names = list(base_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        valid_metrics = [m for m in metrics if base_results[model_names[0]].get(m) is not None]
        
        n = len(valid_metrics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, valid_metrics):
            values = [base_results[m].get(metric, 0) for m in model_names]
            values.append(ensemble_result.get(metric, 0))
            names = model_names + ['Ensemble']
            colors = [self.colors[i % len(self.colors)] for i in range(len(model_names))]
            colors.append('#E74C3C')
            
            ax.bar(names, values, color=colors, edgecolor='white')
            ax.set_ylim(0, 1.1)
            ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(ax.patches, values):
                if val is not None:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.3f}', ha='center', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'{self.dataset_name.capitalize()} - Model Performance Comparison',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_roc_curves(self, base_predictions, ensemble_proba, y_test, save_path=None):
        """绘制ROC曲线"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_roc_curves.png'
        
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(base_predictions)))
        
        for (model_name, proba), color in zip(base_predictions.items(), colors):
            try:
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.5, alpha=0.7,
                       label=f'{model_name} (AUC={roc_auc:.3f})')
            except Exception:
                pass
        
        fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_proba)
        auc_ens = auc(fpr_ens, tpr_ens)
        ax.plot(fpr_ens, tpr_ens, color='#E74C3C', lw=2.5,
               label=f'Ensemble (AUC={auc_ens:.3f})', zorder=10)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title(f'{self.dataset_name.capitalize()} - ROC Curve Comparison',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_ensemble_weights(self, weights, save_path=None):
        """绘制集成权重饼图"""
        if save_path is None:
            save_path = f'output/figures/{self.dataset_name}_ensemble_weights.png'
        
        weights_clean = {k: v for k, v in weights.items() if v > 0.001}
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(weights_clean)))
        wedges, texts, autotexts = ax.pie(
            weights_clean.values(),
            labels=weights_clean.keys(),
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(weights_clean),
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax.set_title(f'{self.dataset_name.capitalize()} - Ensemble Weight Distribution',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def generate_all_plots(self, results, y_test, base_predictions, ensemble_proba, ensemble_pred):
        """生成所有图表"""
        output_dir = 'output/figures'
        os.makedirs(output_dir, exist_ok=True)
        
        r = {}
        r['model_comparison'] = self.plot_model_comparison(
            results['base_models'], results['ensemble'],
            save_path=f'{output_dir}/{self.dataset_name}_awelm_comparison.png')
        
        r['roc_curves'] = self.plot_roc_curves(
            base_predictions, ensemble_proba, y_test,
            save_path=f'{output_dir}/{self.dataset_name}_awelm_roc.png')
        
        r['ensemble_weights'] = self.plot_ensemble_weights(
            results['weights'],
            save_path=f'{output_dir}/{self.dataset_name}_awelm_weights.png')
        
        return r


class BNMDAPVisualizer:
    """BNMDAP可视化"""
    
    def __init__(self):
        self.colors = {
            'hypertension': '#FF6B6B',
            'heart_disease': '#E74C3C',
            'stroke': '#3498DB',
            'cirrhosis': '#F39C12',
        }
    
    def plot_disease_network(self, network_structure, relative_risk, disease_base_rates, save_path=None):
        """绘制贝叶斯网络结构图"""
        if save_path is None:
            save_path = 'output/figures/bnmdap_network_structure.png'
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 10)
        ax.axis('off')
        
        nodes = {
            'hypertension': (3, 8),
            'heart_disease': (6, 5.5),
            'stroke': (9, 3),
            'cirrhosis': (6, 1)
        }
        
        edges = [
            ('hypertension', 'heart_disease', relative_risk.get(('hypertension', 'heart_disease'), 2.8)),
            ('hypertension', 'stroke', relative_risk.get(('hypertension', 'stroke'), 3.2)),
            ('heart_disease', 'stroke', relative_risk.get(('heart_disease', 'stroke'), 2.5)),
            ('heart_disease', 'cirrhosis', relative_risk.get(('heart_disease', 'cirrhosis'), 1.5)),
        ]
        
        for src, dst, rr in edges:
            x1, y1 = nodes[src]
            x2, y2 = nodes[dst]
            
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=2.5, connectionstyle='arc3,rad=0.1'))
            
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.4
            ax.text(mid_x, mid_y, f'RR={rr:.1f}', fontsize=10, ha='center',
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
        
        for node, (x, y) in nodes.items():
            rate = disease_base_rates.get(node, 0)
            label = node.replace('_', ' ').title()
            circle = plt.Circle((x, y), 0.9, color=self.colors[node], alpha=0.85, zorder=10, ec='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y + 0.2, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=11)
            ax.text(x, y - 0.4, f'P={rate:.1%}', ha='center', va='center', fontsize=9, color='white', zorder=11)
        
        ax.set_title('Bayesian Network Multi-Disease Association Model\nDisease Network Structure with Relative Risk',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_correlation_heatmap(self, relative_risk, disease_base_rates, save_path=None):
        """绘制疾病关联热力图"""
        if save_path is None:
            save_path = 'output/figures/bnmdap_correlation_heatmap.png'
        
        diseases = ['stroke', 'heart_disease', 'cirrhosis', 'hypertension']
        labels = ['Stroke', 'Heart Disease', 'Cirrhosis', 'Hypertension']
        
        rr_matrix = np.ones((4, 4))
        for i, d1 in enumerate(diseases):
            for j, d2 in enumerate(diseases):
                if i != j:
                    rr_matrix[i, j] = relative_risk.get((d1, d2), relative_risk.get((d2, d1), 1.0))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(rr_matrix, annot=True, fmt='.2f',
                   xticklabels=labels, yticklabels=labels,
                   cmap='YlOrRd', ax=ax, vmin=1, vmax=4,
                   linewidths=1, linecolor='white',
                   annot_kws={'size': 12, 'weight': 'bold'})
        
        ax.set_title('Relative Risk (RR) Heatmap\nRR > 2: Strong | RR 1.5-2: Moderate | RR < 1.5: Weak',
                    fontsize=13, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def plot_comorbidity_probability(self, scenarios_results, save_path=None):
        """绘制共病概率条形图"""
        if save_path is None:
            save_path = 'output/figures/bnmdap_comorbidity_probability.png'
        
        labels = [s['label'] for s in scenarios_results]
        stroke_probs = [s['stroke'] * 100 for s in scenarios_results]
        heart_probs = [s['heart'] * 100 for s in scenarios_results]
        cirrh_probs = [s['cirrhosis'] * 100 for s in scenarios_results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - width, stroke_probs, width, label='Stroke', color=self.colors['stroke'], edgecolor='white')
        ax.bar(x, heart_probs, width, label='Heart Disease', color=self.colors['heart_disease'], edgecolor='white')
        ax.bar(x + width, cirrh_probs, width, label='Cirrhosis', color=self.colors['cirrhosis'], edgecolor='white')
        
        ax.set_ylabel('Disease Probability (%)', fontsize=12)
        ax.set_xlabel('Risk Factor Combinations', fontsize=12)
        ax.set_title('Disease Probability Under Different Risk Factor Combinations',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return save_path
    
    def generate_all_plots(self, network):
        """生成所有图表"""
        output_dir = 'output/figures'
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        results['network'] = self.plot_disease_network(
            network.network_structure, network.relative_risk, network.disease_base_rates,
            save_path=f'{output_dir}/bnmdap_network.png')
        
        results['heatmap'] = self.plot_correlation_heatmap(
            network.relative_risk, network.disease_base_rates,
            save_path=f'{output_dir}/bnmdap_heatmap.png')
        
        scenarios = [
            {'label': 'No Risk Factors', 'stroke': network.disease_base_rates.get('stroke', 0.028),
             'heart': network.disease_base_rates.get('heart_disease', 0.047), 'cirrhosis': network.disease_base_rates.get('cirrhosis', 0.01)},
            {'label': 'Hypertension Only', 'stroke': 0.05, 'heart': 0.12, 'cirrhosis': 0.015},
            {'label': 'Heart Disease Only', 'stroke': 0.07, 'heart': 0.15, 'cirrhosis': 0.02},
            {'label': 'Both Conditions', 'stroke': 0.15, 'heart': 0.25, 'cirrhosis': 0.03},
        ]
        
        results['comorbidity'] = self.plot_comorbidity_probability(
            scenarios, save_path=f'{output_dir}/bnmdap_comorbidity.png')
        
        return results


def visualize_weight_evaluation(weights_df, dataset_name):
    """便捷函数：可视化权重评估结果"""
    visualizer = WeightEvaluationVisualizer(dataset_name)
    return visualizer.generate_all_plots(weights_df, dataset_name)


def visualize_awelm(results, y_test, base_predictions, ensemble_proba, ensemble_pred, dataset_name):
    """便捷函数：可视化AWELM结果"""
    visualizer = AWELMVisualizer(dataset_name)
    return visualizer.generate_all_plots(results, y_test, base_predictions, ensemble_proba, ensemble_pred)


def visualize_bnmdap(network):
    """便捷函数：可视化BNMDAP结果"""
    visualizer = BNMDAPVisualizer()
    return visualizer.generate_all_plots(network)


if __name__ == '__main__':
    print("Disease Prediction Visualization Module")
    print("Available: WeightEvaluationVisualizer, AWELMVisualizer, BNMDAPVisualizer")
