from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, g
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import io
import base64
from datetime import datetime
import webbrowser
import threading
import time
import logging
import traceback
from model_calibration import calibrate_probability
from model_utilities import smooth_probability
from multi_disease_model import robust_model_predict
from inference_predictor import load_inference_models, predict_with_inference

# Limit OpenBLAS/threading to reduce memory usage
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'checkpoints')
INFERENCE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'inference_models')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Common processing before request handling
@app.before_request
def before_request():
    # Pass current datetime to all templates
    g.now = datetime.now()

# Load models and data
def load_models():
    models = {}
    model_dir = 'output/models'
    model_loaded = False
    model_types = {}

    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                model_path = os.path.join(model_dir, filename)
                model_name = filename.replace('.pkl', '')
                try:
                    model = joblib.load(model_path)
                    models[model_name] = model
                    # Check model type, record whether it is a LightGBM model
                    if 'lightgbm' in str(type(model)).lower():
                        model_types[model_name] = 'lightgbm'
                    else:
                        model_types[model_name] = 'other'
                    logger.info(f"Loaded model: {model_name}")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    # Create fallback model
                    if 'stroke' in model_name or 'heart' in model_name:
                        logger.info(f"Creating fallback classification model for {model_name}")
                        from sklearn.ensemble import RandomForestClassifier
                        models[model_name] = RandomForestClassifier(random_state=42)
                        model_types[model_name] = 'other'
                    elif 'cirrhosis' in model_name:
                        logger.info(f"Creating fallback regression model for {model_name}")
                        from sklearn.ensemble import RandomForestRegressor
                        models[model_name] = RandomForestRegressor(random_state=42)

    if not model_loaded:
        logger.warning("No models loaded successfully, all predictions will use simulated data")

    return models, model_loaded, model_types

def load_processed_data():
    data = {}
    data_dir = 'output/processed_data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                data_name = filename.replace('_processed.csv', '')
                try:
                    data[data_name] = pd.read_csv(data_path)
                    logger.info(f"Loaded data: {data_name}")
                except Exception as e:
                    logger.error(f"Failed to load data {data_name}: {e}")
    return data

# Global variables
try:
    MODELS, MODELS_LOADED, MODEL_TYPES = load_models()
    DATA = load_processed_data()
    INFERENCE_MODELS = load_inference_models(INFERENCE_MODEL_DIR)
    logger.info(f"Loaded inference models: {list(INFERENCE_MODELS.keys())}")
except Exception as e:
    logger.critical(f"Initialization failed: {e}")
    MODELS = {}
    MODELS_LOADED = False
    MODEL_TYPES = {}
    DATA = {}
    INFERENCE_MODELS = {}

# Home page route
@app.route('/')
def home():
    """Home page"""
    now = datetime.now()

    try:
        # Ensure data and models are loaded
        if 'DATA' not in globals() or len(DATA) == 0:
            load_processed_data()

        if 'MODELS' not in globals() or len(MODELS) == 0:
            load_models()

        # Generate home page statistics
        stats = {}
        for dataset_name, df in DATA.items():
            stats[dataset_name] = {
                "Sample Count": len(df),
                "Feature Count": len(df.columns) - 3 if 'ID' in df.columns and 'N_Days' in df.columns else len(df.columns) - 1
            }

            # Get target variable name and positive ratio
            if dataset_name == 'heart':
                target = 'HeartDisease'
                stats[dataset_name]["Positive Ratio"] = f"{df[target].mean():.1%}"
            elif dataset_name == 'stroke':
                target = 'stroke'
                stats[dataset_name]["Positive Ratio"] = f"{df[target].mean():.1%}"
            elif dataset_name == 'cirrhosis':
                target = 'Stage'
                # For regression task, show mean of target variable
                stats[dataset_name]["Mean Value"] = f"{df[target].mean():.2f}"

        # Get best model info
        model_info = {}
        for model_name, model in MODELS.items():
            dataset = model_name.split('_')[0]
            metrics_path = f"output/models/{dataset}_best_baseline_model_metrics.json"

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    model_info[dataset] = metrics
                except:
                    model_info[dataset] = {"accuracy": "N/A"}

        # Use standard template
        template = 'index.html'
        return render_template(template, stats=stats, model_info=model_info, now=now)
    except Exception as e:
        logger.error(f"Error loading home page: {e}")
        return render_error(500, "Home Page Load Failed", "Unable to load system home page, please try again later.")

# Static image route
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve image files directly, avoid base64 encoding"""
    try:
        return send_from_directory('static/images', filename)
    except Exception as e:
        logger.error(f"Image load failed {filename}: {e}")
        return '', 404

# Data analysis page route
@app.route('/data-analysis')
def data_analysis():
    """Data analysis page"""
    now = datetime.now()

    try:
        # Ensure data is loaded
        if 'DATA' not in globals() or len(DATA) == 0:
            load_processed_data()

        # Get chart list
        charts = {}

        # Define chart groups
        chart_groups = {
            'heart': {
                'Data Distribution': [
                    'heart_target_distribution.png',
                    'heart_numeric_distributions.png',
                    'heart_correlation_matrix.png'
                ],
                'Feature Analysis': [
                    'heart_RestingBP_outliers.png',
                    'heart_Cholesterol_outliers.png',
                    'heart_MaxHR_outliers.png',
                    'heart_Oldpeak_outliers.png',
                    'heart_FastingBS_outliers.png'
                ],
                'Data Visualization': [
                    'heart_pair_plot.png',
                    'gender_disease_comparison.png',
                    'disease_rate_by_age.png'
                ]
            },
            'stroke': {
                'Data Distribution': [
                    'stroke_target_distribution.png',
                    'stroke_numeric_distributions.png',
                    'stroke_correlation_matrix.png'
                ],
                'Missing Value Analysis': [
                    'stroke_missing_percent.png',
                    'stroke_missing_matrix.png'
                ],
                'Feature Analysis': [
                    'stroke_avg_glucose_level_outliers.png',
                    'stroke_bmi_outliers.png',
                    'stroke_hypertension_outliers.png',
                    'stroke_heart_disease_outliers.png'
                ],
                'Data Visualization': [
                    'stroke_pair_plot.png',
                    'disease_age_comparison.png'
                ]
            },
            'cirrhosis': {
                'Data Distribution': [
                    'cirrhosis_target_distribution.png',
                    'cirrhosis_numeric_distributions.png',
                    'cirrhosis_correlation_matrix.png'
                ],
                'Missing Value Analysis': [
                    'cirrhosis_missing_percent.png',
                    'cirrhosis_missing_matrix.png'
                ],
                'Feature Analysis': [
                    'cirrhosis_Bilirubin_outliers.png',
                    'cirrhosis_Cholesterol_outliers.png',
                    'cirrhosis_Albumin_outliers.png',
                    'cirrhosis_Copper_outliers.png',
                    'cirrhosis_Alk_Phos_outliers.png',
                    'cirrhosis_SGOT_outliers.png',
                    'cirrhosis_Tryglicerides_outliers.png',
                    'cirrhosis_Platelets_outliers.png',
                    'cirrhosis_Prothrombin_outliers.png'
                ],
                'Data Visualization': [
                    'cirrhosis_pair_plot.png'
                ]
            }
        }

        # Check if files exist, build chart info
        for dataset, groups in chart_groups.items():
            charts[dataset] = {}
            for group_name, chart_files in groups.items():
                charts[dataset][group_name] = []
                for chart_file in chart_files:
                    if os.path.exists(f'output/figures/{chart_file}'):
                        charts[dataset][group_name].append({
                            'file': chart_file,
                            'title': chart_file.replace('_', ' ').replace('.png', '').title()
                        })

        # Use standard template
        template = 'data_analysis.html'
        return render_template(template, charts=charts, now=now)
    except Exception as e:
        logger.error(f"Error loading data analysis page: {e}")
        return render_error(500, "Data Analysis Page Load Failed", "Unable to load data analysis page, please try again later.")

# Chart file route
@app.route('/figures/<path:filename>')
def serve_output_figure(filename):
    """Serve chart files from output/figures directory"""
    try:
        # If filename does not end with .png or .html, automatically add .png suffix
        if not (filename.endswith('.png') or filename.endswith('.html')):
            filename = f"{filename}.png"

        # Check if file exists
        file_path = os.path.join('output/figures', filename)
        if os.path.exists(file_path):
            return send_from_directory('output/figures', filename)
        else:
            # If image does not exist, return default image not found message
            return send_from_directory('static/images', 'image_not_found.png')
    except Exception as e:
        logger.error(f"Chart load failed {filename}: {e}")
        return send_from_directory('static/images', 'image_not_found.png')

# Modify this function to reuse serve_output_figure function
@app.route('/static/figures/<path:filename>')
def serve_static_figure(filename):
    """Serve static chart files"""
    return serve_output_figure(filename)

# Model performance page route
@app.route('/model-performance')
def model_performance():
    """Model performance page"""
    now = datetime.now()

    try:
        # Ensure models are loaded
        if 'MODELS' not in globals() or len(MODELS) == 0:
            load_models()

        # Build model performance info
        model_info = {}

        # Iterate through all datasets
        for dataset in ['heart', 'stroke', 'cirrhosis']:
            model_info[dataset] = {
                'performance_metrics': {},
                'charts': []
            }

            # Load model metrics
            metrics_path = f"output/models/{dataset}_best_baseline_model_metrics.json"
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    model_info[dataset]['performance_metrics'] = metrics
                except:
                    model_info[dataset]['performance_metrics'] = {"accuracy": "N/A"}

            # Define charts to display
            chart_files = [
                f'{dataset}_feature_importance.png',
                f'{dataset}_shap_importance.png',
                f'{dataset}_shap_summary.png',
            ]

            # For classification tasks, add ROC curve and confusion matrix
            if dataset in ['heart', 'stroke']:
                chart_files.extend([
                    f'{dataset}_roc_curve.png',
                    f'{dataset}_confusion_matrix.png',
                    f'{dataset}_prob_distribution.png',
                    f'{dataset}_calibration_curve.png',
                    f'{dataset}_calibration_performance.png'
                ])
            else:  # Regression task
                chart_files.extend([
                    f'{dataset}_pred_vs_actual.png',
                    f'{dataset}_residual_plot.png',
                    f'{dataset}_calibration_curve.png',
                    f'{dataset}_calibration_performance.png'
                ])

            # Check if files exist
            for chart_file in chart_files:
                if os.path.exists(f'output/figures/{chart_file}'):
                    model_info[dataset]['charts'].append({
                        'file': chart_file,
                        'title': chart_file.replace('_', ' ').replace('.png', '').title()
                    })

        # Use standard template
        template = 'model_performance.html'
        return render_template(template, model_info=model_info, now=now)
    except Exception as e:
        logger.error(f"Error loading model performance page: {e}")
        return render_error(500, "Model Performance Page Load Failed", "Unable to load model performance page, please try again later.")

# Single disease prediction page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Single disease prediction page"""
    now = datetime.now()

    # Use standard template
    template = 'predict.html'

    if request.method == 'GET':
        # Display prediction form
        try:
            disease = request.args.get('disease', 'stroke')
            return render_template(template, disease=disease, now=now)
        except Exception as e:
            logger.error(f"Prediction page load failed: {e}")
            return render_error(500, "Prediction Page Load Failed", "Unable to load prediction page, please try again later.")
    else:
        # Handle prediction request
        try:
            data = {}
            disease_type = request.form.get('disease_type', 'stroke')

            # Get data from form
            for key in request.form:
                if key != 'disease_type':
                    try:
                        # Try to convert to numeric type
                        val = request.form[key]
                        data[key] = float(val) if val.replace('.', '', 1).isdigit() else val
                    except ValueError:
                        data[key] = request.form[key]

            logger.info(f"Received prediction request: disease_type={disease_type}, data={data}")

            # Use new inference bundle pipeline (handles feature alignment properly)
            if disease_type in ('stroke', 'heart', 'cirrhosis'):
                inference_result = predict_with_inference(INFERENCE_MODELS, disease_type, data)

                if inference_result.get('probability') is not None:
                    raw_prob_sick = inference_result['probability']

                    # Risk-level based calibration
                    if raw_prob_sick > 0.4:
                        risk_level = 'high'
                    elif raw_prob_sick > 0.2:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'

                    if disease_type == 'stroke':
                        calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='spline', risk_level=risk_level)
                        if 0.3 <= raw_prob_sick < 0.4:
                            calibrated_prob_sick = min(0.95, calibrated_prob_sick * 1.2)
                    elif disease_type == 'heart':
                        calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='spline', risk_level=risk_level)
                        if 0.3 <= raw_prob_sick < 0.4:
                            calibrated_prob_sick = min(0.95, calibrated_prob_sick * 1.2)
                    else:
                        calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='power', risk_level=risk_level)

                    calibrated_probabilities = [1.0 - calibrated_prob_sick, calibrated_prob_sick]

                    prediction = int(calibrated_prob_sick > 0.5)

                    result = {
                        'disease_type': disease_type,
                        'prediction': int(prediction),
                        'probability': {str(i): float(prob) for i, prob in enumerate(calibrated_probabilities)},
                        'raw_probability': {"0": 1.0 - raw_prob_sick, "1": raw_prob_sick},
                        'calibrated': True,
                        'method': 'inference_bundle',
                        'model_metrics': inference_result.get('metrics', {}),
                    }
                    return jsonify(result)
                else:
                    logger.error(f"Inference prediction failed for {disease_type}: {inference_result.get('error')}")
                    return jsonify({
                        'disease_type': disease_type,
                        'prediction': 0,
                        'probability': {"0": 0.95, "1": 0.05},
                        'note': f'Prediction failed: {inference_result.get("error")}',
                    }), 500

            # Fall back to legacy models for backward compatibility
            if disease_type == 'stroke':
                model = MODELS.get('stroke_best_baseline_model')
                if not model or not MODELS_LOADED:
                    logger.warning("Stroke prediction model not loaded or using fallback model, generating simulated data")
                    # Generate simulated data
                    import random
                    prediction = 1 if random.random() < 0.15 else 0
                    prob_sick = random.uniform(0.1, 0.9) if prediction == 1 else random.uniform(0.01, 0.3)

                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': 'Model not loaded, using simulated data'
                    }
                    return jsonify(result)

                # Process data
                gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
                smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3,
                               'never_smoked': 0, 'formerly_smoked': 1}

                X = pd.DataFrame({
                    'gender': [gender_map.get(str(data.get('gender', '')), 0)],
                    'age': [data.get('age', 0)],
                    'hypertension': [data.get('hypertension', 0)],
                    'heart_disease': [data.get('heart_disease', 0)],
                    'avg_glucose_level': [data.get('avg_glucose_level', 0)],
                    'bmi': [data.get('bmi', 0)],
                    'smoking_status': [smoking_map.get(str(data.get('smoking_status', '')), 3)]
                })

                try:
                    # Use robust prediction function instead of direct model prediction
                    prediction, raw_probabilities, error_msg, methods_tried = robust_model_predict(
                        model, X, model_type='classification'
                    )

                    if error_msg:
                        logger.warning(f"Issues encountered during prediction, fallback methods tried: {methods_tried}. Error: {error_msg}")

                    # Calibrate probability
                    # Determine risk level
                    raw_prob_sick = raw_probabilities[1]

                    # More refined risk level determination
                    if raw_prob_sick > 0.4:
                        risk_level = 'high'
                    elif raw_prob_sick > 0.2:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'

                    # Calibrate probability - for stroke prediction, use more aggressive calibration
                    calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='spline', risk_level=risk_level)

                    # If raw probability is in medium range but close to high risk, further increase probability
                    if 0.3 <= raw_prob_sick < 0.4:
                        calibrated_prob_sick = min(0.95, calibrated_prob_sick * 1.2)

                    calibrated_probabilities = [1.0 - calibrated_prob_sick, calibrated_prob_sick]

                    # Update prediction result
                    if calibrated_prob_sick > 0.5 and prediction == 0:
                        prediction = 1

                    result = {
                        'disease_type': disease_type,
                        'prediction': int(prediction),
                        'probability': {str(i): float(prob) for i, prob in enumerate(calibrated_probabilities)},
                        'raw_probability': {str(i): float(prob) for i, prob in enumerate(raw_probabilities)},
                        'calibrated': True
                    }

                    # If fallback methods were used, add note
                    if len(methods_tried) > 1:
                        result['methods_note'] = f"Used {', '.join(methods_tried)} methods for prediction"

                except Exception as e:
                    logger.error(f"Stroke prediction calculation failed: {e}")
                    # Generate simulated data
                    import random
                    prediction = 1 if random.random() < 0.15 else 0
                    prob_sick = random.uniform(0.1, 0.9) if prediction == 1 else random.uniform(0.01, 0.3)

                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': 'Prediction calculation failed, using simulated data'
                }

            elif disease_type == 'heart':
                model = MODELS.get('heart_best_baseline_model')
                if not model or not MODELS_LOADED:
                    logger.warning("Heart disease prediction model not loaded or using fallback model, generating simulated data")
                    # Generate simulated data
                    import random
                    prediction = 1 if random.random() < 0.2 else 0
                    prob_sick = random.uniform(0.2, 0.9) if prediction == 1 else random.uniform(0.01, 0.4)

                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': 'Model not loaded, using simulated data'
                    }
                    return jsonify(result)

                # Process data
                X = pd.DataFrame({
                    'Age': [data.get('age', 0)],
                    'Sex': [1 if str(data.get('gender', '')) == 'Male' else 0],
                    'ChestPainType': [str(data.get('chest_pain_type', ''))],
                    'RestingBP': [data.get('resting_bp', 0)],
                    'Cholesterol': [data.get('cholesterol', 0)],
                    'FastingBS': [data.get('fasting_bs', 0)],
                    'RestingECG': [str(data.get('resting_ecg', ''))],
                    'MaxHR': [data.get('max_hr', 0)],
                    'ExerciseAngina': [1 if str(data.get('exercise_angina', '')) == 'Y' else 0],
                    'Oldpeak': [data.get('oldpeak', 0)],
                    'ST_Slope': [str(data.get('st_slope', ''))]
                })

                try:
                    # Use robust prediction function
                    prediction, raw_probabilities, error_msg, methods_tried = robust_model_predict(
                        model, X, model_type='classification'
                    )

                    if error_msg:
                        logger.warning(f"Issues encountered during prediction, fallback methods tried: {methods_tried}. Error: {error_msg}")

                    # Calibrate probability
                    # Determine risk level
                    raw_prob_sick = raw_probabilities[1]

                    # More refined risk level determination
                    if raw_prob_sick > 0.4:
                        risk_level = 'high'
                    elif raw_prob_sick > 0.2:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'

                    # Calibrate probability - for heart disease prediction, use more aggressive calibration
                    calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='spline', risk_level=risk_level)

                    # If raw probability is in medium range but close to high risk, further increase probability
                    if 0.3 <= raw_prob_sick < 0.4:
                        calibrated_prob_sick = min(0.95, calibrated_prob_sick * 1.2)

                    calibrated_probabilities = [1.0 - calibrated_prob_sick, calibrated_prob_sick]

                    # Update prediction result
                    if calibrated_prob_sick > 0.5 and prediction == 0:
                        prediction = 1

                    result = {
                        'disease_type': disease_type,
                        'prediction': int(prediction),
                        'probability': {str(i): float(prob) for i, prob in enumerate(calibrated_probabilities)},
                        'raw_probability': {str(i): float(prob) for i, prob in enumerate(raw_probabilities)},
                        'calibrated': True
                    }

                    # If fallback methods were used, add note
                    if len(methods_tried) > 1:
                        result['methods_note'] = f"Used {', '.join(methods_tried)} methods for prediction"

                except Exception as e:
                    logger.error(f"Heart disease prediction calculation failed: {e}")
                    # Generate simulated data
                    import random
                    prediction = 1 if random.random() < 0.2 else 0
                    prob_sick = random.uniform(0.2, 0.9) if prediction == 1 else random.uniform(0.01, 0.4)

                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': 'Prediction calculation failed, using simulated data'
                }

            elif disease_type == 'cirrhosis':
                model = MODELS.get('cirrhosis_best_baseline_model')
                if not model or not MODELS_LOADED:
                    logger.warning("Cirrhosis prediction model not loaded or using fallback model, generating simulated data")
                    # Generate simulated data
                    import random
                    prediction = random.randint(1, 4)

                    result = {
                        'disease_type': disease_type,
                        'prediction': float(prediction),
                        'probability': {str(i): (1.0 if i == prediction else 0.0) for i in range(1, 5)},
                        'note': 'Model not loaded, using simulated data'
                    }
                    return jsonify(result)

                # Process data
                X = pd.DataFrame({
                    'Age': [data.get('age', 0)],
                    'Sex': [1 if str(data.get('gender', '')) == 'Male' else 0],
                    'Ascites': [data.get('ascites', 0)],
                    'Hepatomegaly': [data.get('hepatomegaly', 0)],
                    'Spiders': [data.get('spiders', 0)],
                    'Edema': [data.get('edema', 0)],
                    'Bilirubin': [data.get('bilirubin', 0)],
                    'Cholesterol': [data.get('cholesterol', 0)],
                    'Albumin': [data.get('albumin', 0)],
                    'Copper': [data.get('copper', 0)],
                    'Alk_Phos': [data.get('alk_phos', 0)],
                    'SGOT': [data.get('sgot', 0)],
                    'Tryglicerides': [data.get('tryglicerides', 0)],
                    'Platelets': [data.get('platelets', 0)],
                    'Prothrombin': [data.get('prothrombin', 0)]
                })

                try:
                    # Use robust prediction function - cirrhosis is a regression problem
                    raw_prediction, _, error_msg, methods_tried = robust_model_predict(
                        model, X, model_type='regression'
                    )

                    if error_msg:
                        logger.warning(f"Issues encountered during prediction, fallback methods tried: {methods_tried}. Error: {error_msg}")

                    # For cirrhosis, map stage values to probability
                    raw_prob = min(raw_prediction / 4, 1.0)

                    # Calibrate probability
                    risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
                    calibrated_prob = calibrate_probability(raw_prob, method='power', risk_level=risk_level)

                    # Map calibrated probability back to stage value
                    calibrated_prediction = calibrated_prob * 4

                    result = {
                        'disease_type': disease_type,
                        'prediction': float(calibrated_prediction),
                        'raw_prediction': float(raw_prediction),
                        'probability': {str(i): (1.0 if round(calibrated_prediction) == i else 0.0) for i in range(1, 5)},
                        'calibrated': True
                    }

                    # If fallback methods were used, add note
                    if len(methods_tried) > 1:
                        result['methods_note'] = f"Used {', '.join(methods_tried)} methods for prediction"

                except Exception as e:
                    logger.error(f"Cirrhosis prediction calculation failed: {e}")
                    # Generate simulated data
                    import random
                    prediction = random.randint(1, 4)

                result = {
                    'disease_type': disease_type,
                    'prediction': float(prediction),
                        'probability': {str(i): (1.0 if i == prediction else 0.0) for i in range(1, 5)},
                        'note': 'Prediction calculation failed, using simulated data'
                }
            else:
                logger.error(f"Unknown disease type: {disease_type}")
                return jsonify({'error': 'Unsupported disease type'})

            logger.info(f"Prediction result: {result}")
            return jsonify(result)

        except Exception as e:
            error_msg = f"Prediction failed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return jsonify({'error': str(e)})

# Multi-disease joint prediction route
@app.route('/multi-predict', methods=['GET', 'POST'])
def multi_predict():
    """Multi-disease joint prediction page"""
    now = datetime.now()

    # Use standard template
    template = 'multi_predict.html'

    if request.method == 'GET':
        # Display prediction form page
        return render_template(template, now=now)

    # Handle POST request - API call
    try:
        data = {}

        # Get data from form
        for key in request.form:
            if key != 'prediction_type':
                try:
                    # Try to convert to numeric type
                    val = request.form[key]
                    data[key] = float(val) if val.replace('.', '', 1).isdigit() else val
                except ValueError:
                    data[key] = request.form[key]

        logger.info(f"Received multi-disease prediction request: data={data}")

        # Use multi-disease prediction model for prediction
        try:
            from multi_disease_model import MultiDiseasePredictor
            predictor = MultiDiseasePredictor()

            # Get prediction results (considering correlation between diseases)
            probabilities = predictor.predict_with_correlation(data)

            # Ensure all values are JSON serializable
            sanitized_probabilities = {}
            for key, value in probabilities.items():
                # Convert numpy types to Python native types
                if hasattr(value, 'item'):  # Check if numpy type
                    sanitized_probabilities[key] = float(value.item())
                elif value is None or np.isnan(value):
                    sanitized_probabilities[key] = 0.0  # Replace None and NaN values with 0
                else:
                    sanitized_probabilities[key] = float(value)

            result = {
                'status': 'success',
                'probabilities': sanitized_probabilities,
                'message': 'Multi-disease risk prediction completed'
            }

        except Exception as e:
            error_msg = f"Multi-disease model loading or prediction failed: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)

            # If model fails, use simulated data
            import random

            # Generate random but relatively reasonable single disease probabilities
            stroke_prob = random.uniform(0.01, 0.2)
            heart_prob = random.uniform(0.02, 0.25)
            cirrhosis_prob = random.uniform(0.01, 0.15)

            # Calculate disease combination probabilities
            stroke_heart = stroke_prob * heart_prob * 1.2  # Consider positive correlation
            stroke_cirrhosis = stroke_prob * cirrhosis_prob * 1.1
            heart_cirrhosis = heart_prob * cirrhosis_prob * 1.15
            all_three = stroke_heart * cirrhosis_prob * 0.9

            # Calculate probability of having a single disease
            stroke_only = stroke_prob - stroke_heart - stroke_cirrhosis + all_three
            heart_only = heart_prob - stroke_heart - heart_cirrhosis + all_three
            cirrhosis_only = cirrhosis_prob - stroke_cirrhosis - heart_cirrhosis + all_three

            # Calculate healthy probability
            none_prob = 1 - stroke_only - heart_only - cirrhosis_only - stroke_heart - stroke_cirrhosis - heart_cirrhosis - all_three

            # Ensure probabilities are non-negative
            probabilities = {
                'stroke': stroke_prob,
                'heart': heart_prob,
                'cirrhosis': cirrhosis_prob,
                'stroke_only': max(0, stroke_only),
                'heart_only': max(0, heart_only),
                'cirrhosis_only': max(0, cirrhosis_only),
                'stroke_heart': stroke_heart,
                'stroke_cirrhosis': stroke_cirrhosis,
                'heart_cirrhosis': heart_cirrhosis,
                'all_three': all_three,
                'none': max(0, none_prob)
            }

            result = {
                'status': 'success',
                'probabilities': probabilities,
                'message': 'Using fallback model for prediction',
                'note': 'Actual model loading failed, using simulated data'
            }

        logger.info(f"Multi-disease prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        error_msg = f"Multi-disease prediction failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': str(e)})

# Multi-disease correlation analysis route
@app.route('/multi-disease')
def multi_disease():
    """Multi-disease correlation analysis page"""
    now = datetime.now()

    try:
        # Ensure data is loaded
        if 'DATA' not in globals() or len(DATA) == 0:
            load_processed_data()

        # Use standard template
        template = 'multi_disease.html'
        return render_template(template, now=now)
    except Exception as e:
        logger.error(f"Error loading multi-disease correlation analysis page: {e}")
        return render_error(500, "Multi-Disease Correlation Analysis Page Load Failed", "Unable to load multi-disease correlation analysis data, please try again later.")

# Unified error handling
def render_error(code, title, message):
    """Render error page"""
    now = datetime.now()
    error_map = {
        404: "Page Not Found",
        500: "Internal Server Error"
    }
    error_title = title or error_map.get(code, "Unknown Error")
    return render_template('error.html',
                          error_code=code,
                          error_title=error_title,
                          error_message=message,
                          now=now), code

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"Page not found: {request.path}")
    return render_error(404, "Page Not Found", f"The page {request.path} you are looking for does not exist.")

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Server error: {e}")
    return render_error(500, "Internal Server Error", "An error occurred while processing your request, please try again later.")

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.error(f"Uncaught exception: {e}\n{traceback.format_exc()}")
    return render_error(500, "Internal Server Error", "An unexpected error occurred, please try again later.")

def generate_missing_charts():
    """Generate missing confusion matrices and residual plots"""
    try:
        logger.info("Checking and generating possibly missing charts...")

        # Ensure output directory exists
        os.makedirs('output/figures', exist_ok=True)

        # Use unified chart generation module
        try:
            from chart_generator import run_chart_generation

            # Check if confusion matrix is missing
            if not os.path.exists('output/figures/heart_confusion_matrix.png'):
                logger.info("Generating confusion matrix for heart dataset...")
                run_chart_generation('heart', ['confusion'])

            if not os.path.exists('output/figures/stroke_confusion_matrix.png'):
                logger.info("Generating confusion matrix for stroke dataset...")
                run_chart_generation('stroke', ['confusion'])

            # Check if residual plot is missing
            if not os.path.exists('output/figures/heart_residual_plot.png'):
                logger.info("Generating residual plot for heart dataset...")
                run_chart_generation('heart', ['residual'])

            if not os.path.exists('output/figures/stroke_residual_plot.png'):
                logger.info("Generating residual plot for stroke dataset...")
                run_chart_generation('stroke', ['residual'])

            if not os.path.exists('output/figures/cirrhosis_residual_plot.png'):
                logger.info("Generating residual plot for cirrhosis dataset...")
                run_chart_generation('cirrhosis', ['residual'])

        except ImportError:
            logger.warning("chart_generator module not found, trying to generate charts using old methods...")

            # Iterate through all loaded models
            for model_name, model in MODELS.items():
                dataset_name = model_name.split('_')[0]  # Extract dataset name from model name

                if dataset_name not in DATA:
                    logger.warning(f"Cannot find {dataset_name} dataset, skipping chart generation")
                    continue

                df = DATA[dataset_name]

                # Prepare data, exclude ID and N_Days columns
                if dataset_name == 'stroke':
                    X = df.drop(['stroke', 'ID', 'N_Days'], axis=1, errors='ignore')
                    y = df['stroke'] if 'stroke' in df.columns else None
                elif dataset_name == 'heart':
                    X = df.drop(['HeartDisease', 'ID', 'N_Days'], axis=1, errors='ignore')
                    y = df['HeartDisease'] if 'HeartDisease' in df.columns else None
                elif dataset_name == 'cirrhosis':
                    X = df.drop(['Stage', 'ID', 'N_Days'], axis=1, errors='ignore')
                    y = df['Stage'] if 'Stage' in df.columns else None
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue

                if y is None:
                    logger.warning(f"Cannot find target variable for {dataset_name} dataset, skipping chart generation")
                    continue

                # Generate confusion matrix (except for regression tasks)
                if dataset_name != 'cirrhosis' and not os.path.exists(f'output/figures/{dataset_name}_confusion_matrix.png'):
                    try:
                        from sklearn.metrics import confusion_matrix
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        # Use robust prediction function instead of direct model prediction
                        y_preds = []
                        for i in range(len(X)):
                            X_row = X.iloc[[i]]
                            pred, _, _, _ = robust_model_predict(model, X_row,
                                                          model_type='classification' if dataset_name != 'cirrhosis' else 'regression')
                            y_preds.append(pred)

                        # Calculate confusion matrix
                        cm = confusion_matrix(y, y_preds)

                        # Plot confusion matrix
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'{dataset_name.capitalize()} Model Confusion Matrix')
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.tight_layout()
                        plt.savefig(f'output/figures/{dataset_name}_confusion_matrix.png')
                        plt.close()

                        logger.info(f"Confusion matrix for {dataset_name} has been generated")
                    except Exception as e:
                        logger.error(f"Error generating confusion matrix for {dataset_name}: {e}")

                # Generate residual plot/accuracy plot
                if not os.path.exists(f'output/figures/{dataset_name}_residual_plot.png'):
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np

                        # Use robust prediction function instead of direct model prediction
                        y_preds = []
                        for i in range(len(X)):
                            X_row = X.iloc[[i]]
                            pred, _, _, _ = robust_model_predict(model, X_row,
                                                          model_type='classification' if dataset_name != 'cirrhosis' else 'regression')
                            y_preds.append(pred)

                        # For classification tasks, generate accuracy plot
                        if dataset_name in ['stroke', 'heart']:
                            from sklearn.metrics import accuracy_score
                            # Calculate accuracy
                            accuracy = accuracy_score(y, y_preds)

                            # Create bar chart
                            plt.figure(figsize=(10, 6))
                            plt.bar(['Accuracy'], [accuracy], color='blue')
                            plt.ylim(0, 1)
                            plt.title(f'{dataset_name.capitalize()} Model Accuracy')
                            plt.ylabel('Accuracy')
                            plt.tight_layout()
                            plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                            plt.close()

                            logger.info(f"Accuracy plot for {dataset_name} has been generated")
                        else:  # Regression task
                            # Calculate residuals
                            residuals = y.values - np.array(y_preds)

                            # Plot residual plot
                            plt.figure(figsize=(10, 6))
                            plt.scatter(y_preds, residuals, alpha=0.5)
                            plt.axhline(y=0, color='r', linestyle='-')
                            plt.xlabel('Predicted Value')
                            plt.ylabel('Residual')
                            plt.title(f'{dataset_name.capitalize()} Model Residual Plot')
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                            plt.close()

                            logger.info(f"Residual plot for {dataset_name} has been generated")
                    except Exception as e:
                        logger.error(f"Error generating residual plot for {dataset_name}: {e}")

    except Exception as e:
        logger.error(f"Error generating missing charts: {e}\n{traceback.format_exc()}")

# Generate missing charts in a background thread (never blocks server startup)
def _run_chart_gen():
    try:
        generate_missing_charts()
    except Exception:
        pass  # chart generation is non-critical for server startup

threading.Thread(target=_run_chart_gen, daemon=True).start()

# Open browser
def open_browser():
    # Wait for server to be ready, then open browser
    time.sleep(2)
    webbrowser.open('http://localhost:5000')


# ============================================================
# New route: Question 1 - Multi-factor Weight Evaluation
# ============================================================
@app.route('/weight-evaluation')
def weight_evaluation():
    """Multi-factor weight evaluation page"""
    now = datetime.now()
    try:
        template = 'weight_evaluation.html'
        return render_template(template, now=now)
    except Exception as e:
        logger.error(f"Error loading weight evaluation page: {e}")
        return render_error(500, "Page Load Failed", "Unable to load weight evaluation page.")


@app.route('/api/weight-evaluation/<dataset>', methods=['GET'])
def api_weight_evaluation(dataset):
    """
    API: Get feature weight evaluation results for specified dataset.
    Loads from pre-trained checkpoint to avoid runtime training.
    """
    try:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'weight_eval_{dataset}.json')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return jsonify(result)

        return jsonify({'error': f'Checkpoint not found for {dataset}. Run generate_checkpoints.py first.'}), 404

    except Exception as e:
        logger.error(f"Weight evaluation failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# New route: Question 2 - Adaptive Weighted Ensemble Learning (AWELM)
# ============================================================
@app.route('/awelm')
def awelm_page():
    """AWELM ensemble learning model page"""
    now = datetime.now()
    try:
        return render_template('awelm.html', now=now)
    except Exception as e:
        logger.error(f"Error loading AWELM page: {e}")
        return render_error(500, "Page Load Failed", "Unable to load AWELM page.")


@app.route('/api/awelm/<dataset>', methods=['GET', 'POST'])
def api_awelm(dataset):
    """
    API: Get AWELM ensemble model results.
    Loads from pre-trained checkpoint to avoid runtime training.
    """
    try:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'awelm_{dataset}.json')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return jsonify(result)

        return jsonify({'error': f'Checkpoint not found for {dataset}. Run generate_checkpoints.py first.'}), 404

    except Exception as e:
        logger.error(f"AWELM failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# New route: Question 3 - Bayesian Network Multi-Disease Association
# ============================================================
@app.route('/bnmdap')
def bnmdap_page():
    """Bayesian network multi-disease association analysis page"""
    now = datetime.now()
    try:
        return render_template('bnmdap.html', now=now)
    except Exception as e:
        logger.error(f"Error loading BNMDAP page: {e}")
        return render_error(500, "Page Load Failed", "Unable to load Bayesian network analysis page.")


@app.route('/api/bnmdap/predict', methods=['POST'])
def api_bnmdap_predict():
    """
    API: Predict multi-disease association probability based on Bayesian network.
    Uses pre-computed scenarios from checkpoint for instant response.
    """
    try:
        data = request.get_json() or {}

        hypertension = int(data.get('hypertension', 0))
        heart_disease = int(data.get('heart_disease', 0))
        cirrhosis = int(data.get('cirrhosis', 0))

        # Determine scenario key
        scenario_keys = {
            (0, 0, 0): 'none',
            (1, 0, 0): 'hypertension_only',
            (0, 1, 0): 'heart_only',
            (0, 0, 1): 'cirrhosis_only',
            (1, 1, 0): 'hypertension_heart',
            (1, 0, 1): 'hypertension_cirrhosis',
            (0, 1, 1): 'heart_cirrhosis',
            (1, 1, 1): 'all_three',
        }
        scenario = scenario_keys.get((hypertension, heart_disease, cirrhosis), 'none')

        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'bnmdap_analysis.json')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            probs = checkpoint.get('scenarios', {}).get(scenario, {})
            return jsonify({
                'status': 'success',
                'input': {
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'cirrhosis': cirrhosis,
                },
                'probabilities': probs
            })

        # Fallback: real-time computation
        from bnmdap import BayesianDiseaseNetwork
        import pandas as pd

        dfs = {}
        for csv_file, name in [
            ('heart.csv', 'heart'), ('stroke.csv', 'stroke'), ('cirrhosis.csv', 'cirrhosis')
        ]:
            if os.path.exists(csv_file):
                dfs[name] = pd.read_csv(csv_file)

        network = BayesianDiseaseNetwork()
        network.estimate_prior_from_data(
            stroke_df=dfs.get('stroke'),
            heart_df=dfs.get('heart'),
            cirrhosis_df=dfs.get('cirrhosis')
        )

        result = network.predict(
            hypertension=hypertension,
            heart_disease=heart_disease,
            cirrhosis=cirrhosis
        )

        sanitized = {}
        for key, value in result.items():
            if hasattr(value, 'item'):
                sanitized[key] = float(value.item())
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                sanitized[key] = 0.0
            else:
                sanitized[key] = float(value)

        return jsonify({
            'status': 'success',
            'input': {
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'cirrhosis': cirrhosis,
            },
            'probabilities': sanitized
        })

    except Exception as e:
        logger.error(f"BNMDAP prediction failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/bnmdap/analysis', methods=['GET'])
def api_bnmdap_analysis():
    """API: Get Bayesian network analysis results"""
    try:
        from bnmdap import BayesianDiseaseNetwork
        import pandas as pd

        dfs = {}
        for csv_file, name in [
            ('heart.csv', 'heart'), ('stroke.csv', 'stroke'), ('cirrhosis.csv', 'cirrhosis')
        ]:
            if os.path.exists(csv_file):
                dfs[name] = pd.read_csv(csv_file)

        network = BayesianDiseaseNetwork()
        network.estimate_prior_from_data(
            stroke_df=dfs.get('stroke'),
            heart_df=dfs.get('heart'),
            cirrhosis_df=dfs.get('cirrhosis')
        )

        # Return network structure info
        priors = {}
        for disease, rate in network.disease_base_rates.items():
            priors[disease] = float(rate)

        relative_risks = {}
        for (d1, d2), rr in network.relative_risk.items():
            relative_risks[f"{d1}_to_{d2}"] = float(rr)

        return jsonify({
            'status': 'success',
            'priors': priors,
            'relative_risks': relative_risks,
            'network_structure': {
                'nodes': list(network.network_structure.keys()),
                'edges': [
                    (p, d)
                    for d, info in network.network_structure.items()
                    for p in info['parents']
                ]
            }
        })

    except Exception as e:
        logger.error(f"BNMDAP analysis failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # If running this file directly, open browser and start Flask server
    threading.Thread(target=open_browser).start()
    app.run(debug=False)
