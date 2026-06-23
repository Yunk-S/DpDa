"""
DpDa Model Checkpoint Generator
Pre-trains all models and saves results to JSON files to avoid runtime training.
Usage: python generate_checkpoints.py
"""
import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Force unbuffered output for progress display
sys.stdout.reconfigure(line_buffering=True)


class ProgressBar:
    """Simple step-based progress indicator (ASCII, cross-platform)"""
    def __init__(self, total, label=''):
        self.total = total
        self.label = label
        self.start = time.time()
        self.current = 0

    def inc(self, note=''):
        self.current += 1
        current = self.current
        elapsed = time.time() - self.start
        bar_len = 20
        filled = int(bar_len * current / self.total) if self.total > 0 else 0
        bar = '=' * filled + '-' * (bar_len - filled)
        pct = current / self.total * 100 if self.total > 0 else 0

        if current > 0 and current < self.total and elapsed > 0:
            eta = elapsed / current * (self.total - current)
            if eta >= 3600:
                eta_str = f'{eta/3600:.1f}h'
            elif eta >= 60:
                eta_str = f'{int(eta//60)}m{int(eta%60)}s'
            else:
                eta_str = f'{int(eta)}s'
        else:
            eta_str = '--'

        msg = f'\r  {self.label} [{bar}] {current}/{self.total} ({pct:.0f}%) ETA:{eta_str}  {note}'
        sys.stdout.write(msg)
        sys.stdout.flush()

    def done(self, note=''):
        elapsed = time.time() - self.start
        if elapsed >= 60:
            elapsed_str = f'{int(elapsed//60)}m{int(elapsed%60)}s'
        else:
            elapsed_str = f'{elapsed:.1f}s'
        bar = '=' * 20
        msg = f'\r  {self.label} [{bar}] DONE {self.total}/{self.total} (100%) T:{elapsed_str}  {note}'
        sys.stdout.write(msg)
        sys.stdout.write('\n')
        sys.stdout.flush()


def log(msg):
    msg = msg.replace('\u2713', '[OK]').replace('\u2717', '[X]')
    sys.stdout.write(f'  {msg}\n')
    sys.stdout.flush()


def save_checkpoint(name, data):
    path = os.path.join(CHECKPOINT_DIR, f'{name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    size_kb = os.path.getsize(path) / 1024
    log(f'  [OK] Saved {name}.json ({size_kb:.1f} KB)')


print('=' * 70)
print('  DpDa Checkpoint Generator'.center(70))
print('=' * 70)
print()

CHECKPOINT_DIR = 'output/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================================
# 1. WEIGHT EVALUATION CHECKPOINTS
# =====================================================================
print('─' * 70)
print('  [1/3] Weight Evaluation (单因素权重评估)')
print('─' * 70)

from weight_evaluation import FeatureWeightEvaluator

WE_DATASETS = [
    ('heart',    'heart.csv',    'HeartDisease'),
    ('stroke',   'stroke.csv',   'stroke'),
    ('cirrhosis','cirrhosis.csv','Stage', True),  # True = binarize: Stage >= 3
]

# Checkpoint 1: weight evaluation
pb = ProgressBar(len(WE_DATASETS), label='[WE]')
for entry in WE_DATASETS:
    dataset, csv_file, target_col = entry[0], entry[1], entry[2]
    binarize = entry[3] if len(entry) > 3 else False
    pb.inc(f'< {dataset}')
    try:
        df = pd.read_csv(csv_file)
        if binarize:
            df[target_col] = (df[target_col] >= 3).astype(int)
        evaluator = FeatureWeightEvaluator(dataset_name=dataset)
        exclude_cols = ['id', 'ID', 'N_Days']
        feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
        categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X, y, feature_names = evaluator.preprocess(df, target_col, categorical_cols, numeric_cols)
        evaluator.univariate_analysis(X, y, feature_names)
        evaluator.multivariate_analysis(X, y, feature_names)
        summary = evaluator.calculate_weights()

        result = {
            'dataset': dataset,
            'full_auc': float(evaluator.full_auc),
            'generated_at': datetime.now().isoformat(),
            'features': []
        }

        for _, row in summary.iterrows():
            result['features'].append({
                'name': str(row['Feature']),
                'coefficient': float(row['Coefficient_Full']),
                'p_value': float(row['P_Value_Full']),
                'odds_ratio': float(row['Odds_Ratio']),
                'univariate_auc': float(row['AUC']),
                'delta_auc': float(row['Delta_AUC']),
                'raw_weight': float(row['Raw_Weight']),
                'normalized_weight': float(row['Normalized_Weight_Pct']),
                'cumulative_weight': float(row['Cumulative_Weight_Pct']),
                'significant': bool(row['Significant']),
            })

        save_checkpoint(f'weight_eval_{dataset}', result)
    except Exception as e:
        import traceback as tb
        log(f'  [X] ERROR {dataset}: {e}')
        for line in traceback.format_exc().strip().split('\n'):
            if line.strip():
                log('    ' + line.strip())

pb.done()

# Checkpoint 2: AWELM ensemble
print('\n' + '─' * 70)
print('  [2/3] AWELM Ensemble (自适应加权集成)')
print('─' * 70)

from awelm import AdaptiveWeightedEnsemble
from sklearn.preprocessing import StandardScaler, LabelEncoder

pb = ProgressBar(len(WE_DATASETS), label='[AWELM]')
for entry in WE_DATASETS:
    dataset, csv_file, target_col = entry[0], entry[1], entry[2]
    binarize = entry[3] if len(entry) > 3 else False
    pb.inc(f'< {dataset}')
    try:
        df = pd.read_csv(csv_file)
        if binarize:
            df[target_col] = (df[target_col] >= 3).astype(int)
        exclude_cols = ['id', 'ID']
        feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        evaluator = AdaptiveWeightedEnsemble(dataset_name=dataset)
        results = evaluator.run_full_pipeline(X_scaled.values, y.values)

        output = {
            'dataset': dataset,
            'generated_at': datetime.now().isoformat(),
            'base_models': {},
            'ensemble': {},
            'weights': {},
            'best_model': results['best_model'],
        }

        for name, metrics in results['base_models'].items():
            output['base_models'][name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'auc': float(metrics['auc']) if metrics.get('auc') else None,
            }

        ens = results['ensemble']
        output['ensemble'] = {
            'accuracy': float(ens['accuracy']),
            'precision': float(ens['precision']),
            'recall': float(ens['recall']),
            'f1': float(ens['f1']),
            'auc': float(ens['auc']) if ens.get('auc') else None,
        }

        for name, w in results['weights'].items():
            output['weights'][name] = float(w)

        save_checkpoint(f'awelm_{dataset}', output)
    except Exception as e:
        log(f'  [X] ERROR {dataset}: {e}')

pb.done()

# =====================================================================
# 3. BNMDAP CHECKPOINTS
# =====================================================================
print('\n' + '─' * 70)
print('  [3/3] BNMDAP Bayesian Network (贝叶斯网络)')
print('─' * 70)

from bnmdap import BayesianDiseaseNetwork

BNMDAP_STEPS = [
    ('Load datasets',       'loading datasets'),
    ('Init network',        'init network'),
    ('Estimate priors',      'estimating priors'),
    ('Compute 8 scenarios', 'computing probabilities'),
    ('Save checkpoint',     'saving'),
]

try:
    pb = ProgressBar(len(BNMDAP_STEPS), label='[BNMDAP]')
    for i, (step, note) in enumerate(BNMDAP_STEPS):
        pb.inc(f'< {note}')

        if i == 0:
            dfs = {}
            for csv_file, name in [
                ('heart.csv', 'heart'),
                ('stroke.csv', 'stroke'),
                ('cirrhosis.csv', 'cirrhosis')
            ]:
                if os.path.exists(csv_file):
                    dfs[name] = pd.read_csv(csv_file)
        elif i == 1:
            network = BayesianDiseaseNetwork()
        elif i == 2:
            network.estimate_prior_from_data(
                stroke_df=dfs.get('stroke'),
                heart_df=dfs.get('heart'),
                cirrhosis_df=dfs.get('cirrhosis')
            )
        elif i == 3:
            scenarios = [
                (0,0,0,'none'), (1,0,0,'hypertension_only'), (0,1,0,'heart_only'),
                (0,0,1,'cirrhosis_only'), (1,1,0,'hypertension_heart'),
                (1,0,1,'hypertension_cirrhosis'), (0,1,1,'heart_cirrhosis'), (1,1,1,'all_three'),
            ]
            bnmdap_result = {
                'generated_at': datetime.now().isoformat(),
                'priors': {},
                'scenarios': {}
            }
            if hasattr(network, 'priors'):
                for k, v in network.priors.items():
                    bnmdap_result['priors'][k] = float(v.item()) if hasattr(v,'item') else (float(v) if v else 0.0)
            for htn, hd, cirr, name in scenarios:
                try:
                    result = network.predict(hypertension=htn, heart_disease=hd, cirrhosis=cirr)
                    sanitized = {}
                    for k, v in result.items():
                        if hasattr(v, 'item'): sanitized[k] = float(v.item())
                        elif v is None or (isinstance(v, float) and np.isnan(v)): sanitized[k] = 0.0
                        else: sanitized[k] = float(v)
                    bnmdap_result['scenarios'][name] = sanitized
                except Exception as ex:
                    log(f'    Scenario {name} failed: {ex}')
        elif i == 4:
            save_checkpoint('bnmdap_analysis', bnmdap_result)

    pb.done()
except Exception as e:
    log(f'  [X] ERROR BNMDAP: {e}')

# =====================================================================
# Summary
# =====================================================================
print()
print('=' * 70)
print('  Checkpoint Generation Complete!'.center(70))
print('=' * 70)
total_kb = 0
n = 0
for f in sorted(os.listdir(CHECKPOINT_DIR)):
    if f.endswith('.json'):
        size_kb = os.path.getsize(os.path.join(CHECKPOINT_DIR, f)) / 1024
        total_kb += size_kb
        print(f'  [OK] {f:<40} {size_kb:>8.1f} KB')
        n += 1
print(f'\n  Total: {n} files, {total_kb:.1f} KB')
print(f'  Location: {os.path.abspath(CHECKPOINT_DIR)}')
print('=' * 70)

