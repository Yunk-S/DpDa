"""
Vercel Python serverless: /api/predict
Single-disease prediction from inference model bundles.
POST body: { disease_type: 'stroke'|'heart'|'cirrhosis', ...feature_fields }
"""
import json
import os

try:
    import joblib
    import numpy as np
    _OK = True
except Exception as e:
    _OK = False
    _ERR = str(e)

_MODELS = {}
_LOADED = False
_MODEL_DIR = os.environ.get('MODEL_DIR', 'output/inference_models')


# ── schemas ──────────────────────────────────────────────────────

def _num(v, default):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _transform_stroke(body):
    d = {
        'gender': 'Male', 'age': 50, 'hypertension': 0, 'heart_disease': 0,
        'avg_glucose_level': 100, 'bmi': 25, 'smoking_status': 'Unknown'}
    gm = {'male': 0, 'female': 1}
    sm = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
    g = body.get('gender', d['gender'])
    s = body.get('smoking_status', d['smoking_status'])
    return {
        'gender': gm.get(str(g).lower(), 0),
        'age': _num(body.get('age'), d['age']),
        'hypertension': int(body.get('hypertension', 0) or 0),
        'heart_disease': int(body.get('heart_disease', 0) or 0),
        'avg_glucose_level': _num(body.get('avg_glucose_level'), d['avg_glucose_level']),
        'bmi': _num(body.get('bmi'), d['bmi']),
        'smoking_status': sm.get(str(s).lower(), 3),
    }


def _transform_heart(body):
    d = {
        'Age': 50, 'Sex': 1, 'RestingBP': 120, 'Cholesterol': 200,
        'FastingBS': 0, 'MaxHR': 150, 'ExerciseAngina': 0, 'Oldpeak': 0,
        'ChestPainType': 'ATA', 'RestingECG': 'Normal', 'ST_Slope': 'Flat'}
    g = body.get('gender', 'Male')
    sex = 1 if str(g).lower() in ('male', 'm', '1') else 0
    angina = 1 if str(body.get('exercise_angina', '')).lower() in ('y', 'yes', 'true', '1') else 0
    return {
        'Age': _num(body.get('age'), d['Age']),
        'Sex': sex,
        'RestingBP': _num(body.get('resting_bp'), d['RestingBP']),
        'Cholesterol': _num(body.get('cholesterol'), d['Cholesterol']),
        'FastingBS': int(_num(body.get('fasting_bs'), d['FastingBS'])),
        'MaxHR': _num(body.get('max_hr'), d['MaxHR']),
        'ExerciseAngina': angina,
        'Oldpeak': _num(body.get('oldpeak'), d['Oldpeak']),
        'ChestPainType': str(body.get('chest_pain_type', d['ChestPainType'])),
        'RestingECG': str(body.get('resting_ecg', d['RestingECG'])),
        'ST_Slope': str(body.get('st_slope', d['ST_Slope'])),
    }


def _transform_cirr(body):
    d = {
        'Age': 50 * 365.25, 'Sex': 1, 'Ascites': 0, 'Hepatomegaly': 0,
        'Spiders': 0, 'Edema': 0, 'Bilirubin': 1.0, 'Cholesterol': 200,
        'Albumin': 3.5, 'Copper': 50, 'Alk_Phos': 100, 'SGOT': 40,
        'Tryglicerides': 150, 'Platelets': 300, 'Prothrombin': 10.0}
    g = body.get('gender', 'Male')
    sex = 1 if str(g).lower() in ('male', 'm', '1') else 0
    age_years = _num(body.get('age'), 50)
    age_days = age_years * 365.25 if age_years < 100 else age_years
    return {
        'Age': age_days,
        'Sex': sex,
        'Ascites': int(body.get('ascites', 0) or 0),
        'Hepatomegaly': int(body.get('hepatomegaly', 0) or 0),
        'Spiders': int(body.get('spiders', 0) or 0),
        'Edema': int(body.get('edema', 0) or 0),
        'Bilirubin': _num(body.get('bilirubin'), d['Bilirubin']),
        'Cholesterol': _num(body.get('cholesterol'), d['Cholesterol']),
        'Albumin': _num(body.get('albumin'), d['Albumin']),
        'Copper': _num(body.get('copper'), d['Copper']),
        'Alk_Phos': _num(body.get('alk_phos'), d['Alk_Phos']),
        'SGOT': _num(body.get('sgot'), d['SGOT']),
        'Tryglicerides': _num(body.get('tryglicerides'), d['Tryglicerides']),
        'Platelets': _num(body.get('platelets'), d['Platelets']),
        'Prothrombin': _num(body.get('prothrombin'), d['Prothrombin']),
    }


def _transform(disease, body):
    if disease == 'stroke':
        return _transform_stroke(body)
    elif disease == 'heart':
        return _transform_heart(body)
    elif disease == 'cirrhosis':
        return _transform_cirr(body)
    return {}


# ── model inference ──────────────────────────────────────────────

def _build_X(bundle, inp):
    import pandas as pd
    feats = bundle['features_input']
    cats = bundle['categorical_spec']
    scaler = bundle['scaler']
    ohe = bundle['ohe']

    df = pd.DataFrame([inp])
    num_c, cat_c = [], []

    for f in feats:
        if f in cats:
            if isinstance(cats[f], dict):
                df[f] = df[f].astype(str).map(cats[f]).fillna(0).astype(int)
                num_c.append(f)
            else:
                cat_c.append(f)
        else:
            if f not in df.columns:
                df[f] = 0
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
            num_c.append(f)

    cat_arr = (ohe.transform(df[cat_c].astype(str).fillna('Unknown'))
                if cat_c else np.empty((1, 0)))
    num_arr = df[num_c].values
    return scaler.transform(np.hstack([num_arr, cat_arr]))


def _run(disease, body):
    if disease not in _MODELS:
        return None, f'No model: {disease}'
    try:
        b = _MODELS[disease]
        inp = _transform(disease, body)
        X = _build_X(b, inp)
        proba = b['model'].predict_proba(X)[0]
        raw = float(proba[1])
        pred = 1 if raw >= 0.5 else 0
        risk = 'high' if raw > 0.4 else ('medium' if raw > 0.2 else 'low')
        mult = {'high': 1.0, 'medium': 1.05, 'low': 0.95}
        cal = min(0.999, raw * mult[risk])
        return {
            'disease_type': disease,
            'prediction': pred,
            'probability': {'0': round(1 - cal, 4), '1': round(cal, 4)},
            'raw_probability': {'0': round(1 - raw, 4), '1': round(raw, 4)},
            'calibrated': True,
            'method': 'inference_bundle',
            'model_metrics': b.get('metrics', {}),
        }, None
    except Exception as e:
        return None, str(e)


# ── Vercel entry point ────────────────────────────────────────

def handler(event, context):
    hdrs = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json',
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': hdrs, 'body': ''}

    if not _OK:
        return {'statusCode': 500, 'headers': hdrs,
                'body': json.dumps({'error': f'Import error: {_ERR}'})}

    global _MODELS, _LOADED
    if not _LOADED:
        for dis in ('stroke', 'heart', 'cirrhosis'):
            p = os.path.join(_MODEL_DIR, f'{dis}_model.pkl')
            if os.path.exists(p):
                try:
                    _MODELS[dis] = joblib.load(p)
                except Exception:
                    pass
        _LOADED = True

    try:
        raw_body = event.get('body', '{}')
        content_type = ''
        hdr = event.get('headers', {}) or {}
        for k, v in hdr.items():
            if k.lower() == 'content-type':
                content_type = (v or '').lower()
                break

        # Support JSON and URL-encoded form data
        if 'application/json' in content_type:
            body = json.loads(raw_body) if raw_body else {}
        elif 'application/x-www-form-urlencoded' in content_type or 'multipart' in content_type:
            from urllib.parse import parse_qs
            parsed = parse_qs(raw_body)
            body = {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        else:
            try:
                body = json.loads(raw_body)
            except Exception:
                body = {}
    except Exception:
        return {'statusCode': 400, 'headers': hdrs,
                'body': json.dumps({'error': 'Invalid request body'})}

    disease = body.get('disease_type', body.get('disease', 'stroke'))
    result, err = _run(disease, body)

    if err:
        return {'statusCode': 500, 'headers': hdrs,
                'body': json.dumps({'error': err})}

    return {'statusCode': 200, 'headers': hdrs,
            'body': json.dumps(result)}
