"""
DpDa Inference Model Trainer

Trains clean, inference-ready models for stroke/heart/cirrhosis that use the
EXACT same feature schema the prediction code emits.

Each saved bundle contains:
- model: trained classifier/regressor
- preprocessor: fitted ColumnTransformer/StandardScaler pipeline
- feature_columns: ordered list of input feature names
- metadata: dataset info, training date, metrics
"""
import os
import sys
import json
import joblib
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

INFERENCE_DIR = 'output/inference_models'
os.makedirs(INFERENCE_DIR, exist_ok=True)


# ============================================================
# Feature schema - matches what multi_disease_model.py emits
# ============================================================

# Stroke schema (used in _predict_stroke_probability)
STROKE_FEATURES = ['gender', 'age', 'hypertension', 'heart_disease',
                   'avg_glucose_level', 'bmi', 'smoking_status']
STROKE_CATEGORICAL = {'gender': {'Male': 0, 'Female': 1, 'Other': 2},
                      'smoking_status': {'never smoked': 0, 'formerly smoked': 1,
                                          'smokes': 2, 'Unknown': 3,
                                          'never_smoked': 0, 'formerly_smoked': 1}}

# Heart schema (used in _predict_heart_probability)
HEART_FEATURES = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS',
                  'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ChestPainType',
                  'RestingECG', 'ST_Slope']
HEART_CATEGORICAL = {'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
                     'RestingECG': ['LVH', 'Normal', 'ST'],
                     'ST_Slope': ['Down', 'Flat', 'Up']}

# Cirrhosis schema (used in _predict_cirrhosis_probability)
CIRRHOSIS_FEATURES = ['Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
                      'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
                      'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
CIRRHOSIS_CATEGORICAL = {}


# ============================================================
# Feature transformation utilities
# ============================================================

def build_X(df, features, categorical_spec):
    """
    Build a numeric feature matrix from a raw DataFrame using:
      - direct numeric columns (passed through)
      - categorical columns: one-hot encoded
      - everything scaled via StandardScaler
    Returns (X_matrix, fitted_scaler, fitted_one_hot_encoder)
    """
    from sklearn.preprocessing import OneHotEncoder

    df = df.copy()
    numeric_cols = []
    cat_cols = []

    for f in features:
        if f in categorical_spec:
            if isinstance(categorical_spec[f], dict):
                df[f] = df[f].astype(str).map(categorical_spec[f]).fillna(0).astype(int)
                numeric_cols.append(f)
            else:
                cat_cols.append(f)
        else:
            if df[f].dtype == 'object':
                df[f] = pd.to_numeric(df[f], errors='coerce')
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(df[f].median() if df[f].notna().any() else 0)
            numeric_cols.append(f)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if cat_cols:
        cat_arr = ohe.fit_transform(df[cat_cols].astype(str).fillna('Unknown'))
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        cat_arr = np.empty((len(df), 0))
        cat_feature_names = []

    num_arr = df[numeric_cols].values
    X = np.hstack([num_arr, cat_arr])
    feature_names = numeric_cols + cat_feature_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_names, scaler, ohe


def predict_X(input_dict, features, categorical_spec, scaler, ohe, feature_names):
    """Transform a single input dict into the scaled feature row matching training schema."""
    df = pd.DataFrame([input_dict])
    numeric_cols = []
    cat_cols = []

    for f in features:
        if f in categorical_spec:
            if isinstance(categorical_spec[f], dict):
                df[f] = df[f].astype(str).map(categorical_spec[f]).fillna(0).astype(int)
                numeric_cols.append(f)
            else:
                cat_cols.append(f)
        else:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
            numeric_cols.append(f)

    if cat_cols:
        cat_arr = ohe.transform(df[cat_cols].astype(str).fillna('Unknown'))
    else:
        cat_arr = np.empty((1, 0))

    num_arr = df[numeric_cols].values
    X = np.hstack([num_arr, cat_arr])
    X_scaled = scaler.transform(X)
    return X_scaled


# ============================================================
# Training routines
# ============================================================

def train_stroke():
    print('\n[Stroke] Loading data...')
    df = pd.read_csv('stroke.csv')
    # Build mapping: the prediction code expects columns matching original schema
    rename = {'avg_glucose_level': 'avg_glucose_level', 'bmi': 'bmi'}
    df['smoking_status'] = df['smoking_status'].fillna('Unknown')
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce').fillna(df['bmi'].median())

    y = df['stroke'].values
    X, feat_names, scaler, ohe = build_X(df, STROKE_FEATURES, STROKE_CATEGORICAL)
    print(f'[Stroke] Feature matrix shape: {X.shape}')

    # Use a simple train/test split (80/20)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.5)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)
    f1 = f1_score(y_te, y_pred)
    acc = accuracy_score(y_te, y_pred)
    print(f'[Stroke] ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f}')

    bundle = {
        'model': model,
        'scaler': scaler,
        'ohe': ohe,
        'feature_columns': feat_names,
        'features_input': STROKE_FEATURES,
        'categorical_spec': STROKE_CATEGORICAL,
        'task': 'classification',
        'metrics': {'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc)},
        'trained_at': datetime.now().isoformat(),
    }
    joblib.dump(bundle, os.path.join(INFERENCE_DIR, 'stroke_model.pkl'))
    print(f'[Stroke] Saved bundle -> {INFERENCE_DIR}/stroke_model.pkl')
    return bundle


def train_heart():
    print('\n[Heart] Loading data...')
    df = pd.read_csv('heart.csv')
    df = df.dropna()

    y = df['HeartDisease'].values
    X, feat_names, scaler, ohe = build_X(df, HEART_FEATURES, HEART_CATEGORICAL)
    print(f'[Heart] Feature matrix shape: {X.shape}')

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)
    f1 = f1_score(y_te, y_pred)
    acc = accuracy_score(y_te, y_pred)
    print(f'[Heart] ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f}')

    bundle = {
        'model': model,
        'scaler': scaler,
        'ohe': ohe,
        'feature_columns': feat_names,
        'features_input': HEART_FEATURES,
        'categorical_spec': HEART_CATEGORICAL,
        'task': 'classification',
        'metrics': {'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc)},
        'trained_at': datetime.now().isoformat(),
    }
    joblib.dump(bundle, os.path.join(INFERENCE_DIR, 'heart_model.pkl'))
    print(f'[Heart] Saved bundle -> {INFERENCE_DIR}/heart_model.pkl')
    return bundle


def train_cirrhosis():
    print('\n[Cirrhosis] Loading data...')
    df = pd.read_csv('cirrhosis.csv')
    # Convert Stage 1-4 to binary (>=3 = severe)
    df['Stage_binary'] = (df['Stage'] >= 3).astype(int)
    y = df['Stage_binary'].values

    X, feat_names, scaler, ohe = build_X(df, CIRRHOSIS_FEATURES, CIRRHOSIS_CATEGORICAL)
    print(f'[Cirrhosis] Feature matrix shape: {X.shape}')

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.5)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)
    f1 = f1_score(y_te, y_pred)
    acc = accuracy_score(y_te, y_pred)
    print(f'[Cirrhosis] ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f}')

    bundle = {
        'model': model,
        'scaler': scaler,
        'ohe': ohe,
        'feature_columns': feat_names,
        'features_input': CIRRHOSIS_FEATURES,
        'categorical_spec': CIRRHOSIS_CATEGORICAL,
        'task': 'classification',
        'metrics': {'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc)},
        'trained_at': datetime.now().isoformat(),
    }
    joblib.dump(bundle, os.path.join(INFERENCE_DIR, 'cirrhosis_model.pkl'))
    print(f'[Cirrhosis] Saved bundle -> {INFERENCE_DIR}/cirrhosis_model.pkl')
    return bundle


# ============================================================
# Inference helpers
# ============================================================

def load_bundle(name):
    return joblib.load(os.path.join(INFERENCE_DIR, f'{name}_model.pkl'))


def predict_one(bundle, input_dict):
    """Predict probability using the inference bundle."""
    X = predict_X(
        input_dict,
        bundle['features_input'],
        bundle['categorical_spec'],
        bundle['scaler'],
        bundle['ohe'],
        bundle['feature_columns'],
    )
    proba = bundle['model'].predict_proba(X)[0]
    return float(proba[1])  # positive class probability


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print('=' * 70)
    print('  DpDa Inference Model Trainer')
    print('=' * 70)

    start = time.time()
    train_stroke()
    train_heart()
    train_cirrhosis()
    elapsed = time.time() - start

    print('\n' + '=' * 70)
    print(f'  All inference models saved to: {INFERENCE_DIR}')
    print(f'  Training time: {elapsed:.1f}s')
    print('=' * 70)

    # Sanity check predictions
    print('\nSanity check predictions:')
    s_bundle = load_bundle('stroke')
    print(f'  stroke(age=80, hypertension=1, heart_disease=1):    prob={predict_one(s_bundle, {"age": 80, "hypertension": 1, "heart_disease": 1, "avg_glucose_level": 200, "bmi": 35, "gender": "Male", "smoking_status": "smokes"}):.4f}')
    print(f'  stroke(age=25, hypertension=0, heart_disease=0):    prob={predict_one(s_bundle, {"age": 25, "hypertension": 0, "heart_disease": 0, "avg_glucose_level": 90, "bmi": 22, "gender": "Female", "smoking_status": "never smoked"}):.4f}')

    h_bundle = load_bundle('heart')
    print(f'  heart(age=65, Male, ATA, BP=160, Chol=300, FBS=1):   prob={predict_one(h_bundle, {"Age": 65, "Sex": 1, "ChestPainType": "ASY", "RestingBP": 160, "Cholesterol": 300, "FastingBS": 1, "MaxHR": 120, "ExerciseAngina": 1, "Oldpeak": 2.5, "RestingECG": "Normal", "ST_Slope": "Down"}):.4f}')
    print(f'  heart(age=30, Female, ATA, BP=110, Chol=180, FBS=0): prob={predict_one(h_bundle, {"Age": 30, "Sex": 0, "ChestPainType": "ATA", "RestingBP": 110, "Cholesterol": 180, "FastingBS": 0, "MaxHR": 175, "ExerciseAngina": 0, "Oldpeak": 0.0, "RestingECG": "Normal", "ST_Slope": "Up"}):.4f}')

    c_bundle = load_bundle('cirrhosis')
    print(f'  cirrhosis(Bili=5.0, Albumin=2.5, Ascites=1, Edema=1): prob={predict_one(c_bundle, {"Age": 55*365.25, "Sex": 1, "Ascites": 1, "Hepatomegaly": 1, "Spiders": 1, "Edema": 1, "Bilirubin": 5.0, "Cholesterol": 250, "Albumin": 2.5, "Copper": 100, "Alk_Phos": 1500, "SGOT": 150, "Tryglicerides": 200, "Platelets": 150, "Prothrombin": 12.0}):.4f}')
    print(f'  cirrhosis(Bili=0.8, Albumin=4.0, Ascites=0, Edema=0): prob={predict_one(c_bundle, {"Age": 40*365.25, "Sex": 0, "Ascites": 0, "Hepatomegaly": 0, "Spiders": 0, "Edema": 0, "Bilirubin": 0.8, "Cholesterol": 200, "Albumin": 4.0, "Copper": 50, "Alk_Phos": 800, "SGOT": 40, "Tryglicerides": 120, "Platelets": 300, "Prothrombin": 10.0}):.4f}')
    print()