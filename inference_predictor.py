"""
Inference Predictor - loads pre-trained inference bundles and serves predictions.

Replaces the brittle multi_disease_model.predict_multi_disease_probability() pathway
which suffered from training-serving skew (mismatched feature schemas).
"""
import os
import logging
import joblib
import numpy as np

logger = logging.getLogger(__name__)


# Schema specifications for each disease, matching what the front-end sends

STROKE_SCHEMA = {
    'feature_map': {
        'gender': 'gender', 'age': 'age', 'hypertension': 'hypertension',
        'heart_disease': 'heart_disease', 'avg_glucose_level': 'avg_glucose_level',
        'bmi': 'bmi', 'smoking_status': 'smoking_status',
    },
    'type_map': {
        'gender': 'str', 'age': 'float', 'hypertension': 'int',
        'heart_disease': 'int', 'avg_glucose_level': 'float',
        'bmi': 'float', 'smoking_status': 'str',
    },
    'defaults': {
        'gender': 'Male', 'age': 50, 'hypertension': 0, 'heart_disease': 0,
        'avg_glucose_level': 100, 'bmi': 25, 'smoking_status': 'Unknown',
    }
}

HEART_SCHEMA = {
    'feature_map': {
        'Age': 'age', 'Sex': 'gender', 'RestingBP': 'resting_bp',
        'Cholesterol': 'cholesterol', 'FastingBS': 'fasting_bs',
        'MaxHR': 'max_hr', 'ExerciseAngina': 'exercise_angina',
        'Oldpeak': 'oldpeak', 'ChestPainType': 'chest_pain_type',
        'RestingECG': 'resting_ecg', 'ST_Slope': 'st_slope',
    },
    'type_map': {
        'Age': 'float', 'Sex': 'int', 'RestingBP': 'float',
        'Cholesterol': 'float', 'FastingBS': 'int', 'MaxHR': 'float',
        'ExerciseAngina': 'int', 'Oldpeak': 'float',
        'ChestPainType': 'str', 'RestingECG': 'str', 'ST_Slope': 'str',
    },
    'defaults': {
        'Age': 50, 'Sex': 1, 'RestingBP': 120, 'Cholesterol': 200,
        'FastingBS': 0, 'MaxHR': 150, 'ExerciseAngina': 0, 'Oldpeak': 0,
        'ChestPainType': 'ATA', 'RestingECG': 'Normal', 'ST_Slope': 'Flat',
    },
    'value_transforms': {
        'Sex': lambda v: 1 if str(v).lower() in ('male', 'm', '1') else 0,
        'ExerciseAngina': lambda v: 1 if str(v).lower() in ('y', 'yes', 'true', '1') else 0,
    }
}

CIRRHOSIS_SCHEMA = {
    'feature_map': {
        'Age': 'age', 'Sex': 'gender', 'Ascites': 'ascites',
        'Hepatomegaly': 'hepatomegaly', 'Spiders': 'spiders',
        'Edema': 'edema', 'Bilirubin': 'bilirubin',
        'Cholesterol': 'cholesterol', 'Albumin': 'albumin',
        'Copper': 'copper', 'Alk_Phos': 'alk_phos',
        'SGOT': 'sgot', 'Tryglicerides': 'tryglicerides',
        'Platelets': 'platelets', 'Prothrombin': 'prothrombin',
    },
    'type_map': {
        'Age': 'float', 'Sex': 'int', 'Ascites': 'int',
        'Hepatomegaly': 'int', 'Spiders': 'int', 'Edema': 'int',
        'Bilirubin': 'float', 'Cholesterol': 'float', 'Albumin': 'float',
        'Copper': 'float', 'Alk_Phos': 'float', 'SGOT': 'float',
        'Tryglicerides': 'float', 'Platelets': 'float', 'Prothrombin': 'float',
    },
    'defaults': {
        'Age': 50 * 365.25, 'Sex': 1, 'Ascites': 0, 'Hepatomegaly': 0,
        'Spiders': 0, 'Edema': 0, 'Bilirubin': 1.0, 'Cholesterol': 200,
        'Albumin': 3.5, 'Copper': 50, 'Alk_Phos': 100, 'SGOT': 40,
        'Tryglicerides': 150, 'Platelets': 300, 'Prothrombin': 10.0,
    },
    'value_transforms': {
        'Sex': lambda v: 1 if str(v).lower() in ('male', 'm', '1') else 0,
        # If Age was provided in years, multiply to days (training schema expects days)
        'Age': lambda v: float(v) * 365.25 if float(v) < 100 else float(v),
    }
}


def load_inference_models(model_dir):
    """Load all inference model bundles from a directory."""
    models = {}
    if not os.path.exists(model_dir):
        logger.warning(f"Inference model directory does not exist: {model_dir}")
        return models

    for disease in ('stroke', 'heart', 'cirrhosis'):
        path = os.path.join(model_dir, f'{disease}_model.pkl')
        if os.path.exists(path):
            try:
                models[disease] = joblib.load(path)
                logger.info(f"Loaded inference bundle: {disease}")
            except Exception as e:
                logger.error(f"Failed to load {disease} inference model: {e}")
    return models


def _map_input(form_data, schema):
    """Translate form data dict into a model-ready input dict."""
    feature_map = schema['feature_map']
    type_map = schema['type_map']
    defaults = schema['defaults']
    transforms = schema.get('value_transforms', {})

    mapped = {}
    for model_feature, form_key in feature_map.items():
        if form_key in form_data and form_data[form_key] not in ('', None):
            raw = form_data[form_key]
        else:
            raw = defaults.get(model_feature, 0)

        # Apply custom transform if any
        if model_feature in transforms:
            try:
                raw = transforms[model_feature](raw)
            except Exception:
                raw = defaults.get(model_feature, 0)

        # Type cast
        target_type = type_map.get(model_feature, 'float')
        try:
            if target_type == 'int':
                mapped[model_feature] = int(float(raw))
            elif target_type == 'float':
                mapped[model_feature] = float(raw)
            else:
                mapped[model_feature] = str(raw)
        except (ValueError, TypeError):
            mapped[model_feature] = defaults.get(model_feature, 0)

    return mapped


def _build_X_row(bundle, input_dict):
    """Apply bundle's scaler/ohe to a single input dict to produce one model row."""
    features_input = bundle['features_input']
    categorical_spec = bundle['categorical_spec']
    scaler = bundle['scaler']
    ohe = bundle['ohe']

    import pandas as pd
    df = pd.DataFrame([input_dict])
    numeric_cols = []
    cat_cols = []

    for f in features_input:
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


def predict_with_inference(inference_models, disease, form_data):
    """
    Run inference for a disease using the appropriate inference bundle.

    Returns: dict with probability (positive class), prediction (0/1), raw probability
    """
    if disease not in inference_models:
        return {'probability': None, 'prediction': None, 'error': f'No inference model for {disease}'}

    if disease == 'stroke':
        schema = STROKE_SCHEMA
    elif disease == 'heart':
        schema = HEART_SCHEMA
    elif disease == 'cirrhosis':
        schema = CIRRHOSIS_SCHEMA
    else:
        return {'probability': None, 'prediction': None, 'error': f'Unknown disease {disease}'}

    try:
        bundle = inference_models[disease]
        mapped = _map_input(form_data, schema)
        X = _build_X_row(bundle, mapped)
        proba = bundle['model'].predict_proba(X)[0]
        raw_prob = float(proba[1])
        prediction = int(raw_prob >= 0.5)
        return {
            'probability': raw_prob,
            'prediction': prediction,
            'raw_probability': raw_prob,
            'method': 'inference_bundle',
            'metrics': bundle.get('metrics', {}),
        }
    except Exception as e:
        logger.error(f"Inference prediction failed for {disease}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'probability': None, 'prediction': None, 'error': str(e)}


def predict_single_disease(inference_models, disease, form_data):
    """
    Convenience wrapper: returns just the raw probability (0-1) for one disease.
    Falls back to 0.05 on any error.
    """
    result = predict_with_inference(inference_models, disease, form_data)
    prob = result.get('probability')
    if prob is None:
        return 0.05
    return float(prob)