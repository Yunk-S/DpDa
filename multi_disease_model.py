import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model_calibration import calibrate_probability, ProbabilityCalibrator, AdaptiveCalibrator

try:
    from model_utilities import smooth_probability
except ImportError:
    def smooth_probability(prob, method='clip', min_prob=0.01, max_prob=0.99):
        """Simple probability smoothing function (fallback version)"""
        if np.isscalar(prob):
            return max(min_prob, min(max_prob, prob))
        else:
            return np.clip(prob, min_prob, max_prob)

def robust_model_predict(model, X, model_type='classification'):
    """
    Wrap model prediction with multiple fallback strategies to handle possible errors
    """
    prediction = None
    probabilities = None
    error_msg = None

    methods_tried = []

    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    # Method 1: Direct predict
    try:
        methods_tried.append("standard_predict")
        if isinstance(X, pd.DataFrame):
            prediction = model.predict(X_values)[0]
            try:
                prediction = float(prediction)
            except:
                prediction = 0
        else:
            prediction = model.predict(X)[0]
            try:
                prediction = float(prediction)
            except:
                prediction = 0

        if hasattr(model, 'predict_proba'):
            try:
                if isinstance(X, pd.DataFrame):
                    raw_probabilities = model.predict_proba(X_values)[0]
                    probabilities = []
                    for p in raw_probabilities:
                        try:
                            probabilities.append(float(p))
                        except:
                            probabilities.append(0.0)
                else:
                    raw_probabilities = model.predict_proba(X)[0]
                    probabilities = []
                    for p in raw_probabilities:
                        try:
                            probabilities.append(float(p))
                        except:
                            probabilities.append(0.0)
            except:
                probabilities = None
    except Exception as e:
        error_msg = f"Standard prediction method failed: {str(e)}"
        prediction = None

    if prediction is None:
        # Method 2: For LightGBM models, handle categorical_feature mismatch
        if hasattr(model, '_Booster') and error_msg and 'categorical_feature do not match' in str(error_msg):
            try:
                methods_tried.append("categorical_feature_fix")

                import lightgbm as lgb

                if hasattr(model, '_Booster') and model._Booster is not None:
                    if isinstance(X, pd.DataFrame):
                        raw_preds = model._Booster.predict(X_values)
                    else:
                        raw_preds = model._Booster.predict(X)

                    if model_type == 'classification':
                        if len(raw_preds.shape) == 1:
                            prediction = 1 if raw_preds[0] > 0.5 else 0
                            probabilities = [1-raw_preds[0], raw_preds[0]]
                        else:
                            prediction = np.argmax(raw_preds[0])
                            probabilities = raw_preds[0]
                    else:
                        prediction = raw_preds[0]
                        probabilities = None
                else:
                    raise ValueError("Model does not have an available Booster object")

            except Exception as e:
                error_msg = f"{error_msg}; categorical_feature fix failed: {str(e)}"

        # Method 3: For LightGBM models, try raw_score mode
        if prediction is None and hasattr(model, '_Booster'):
            try:
                methods_tried.append("lightgbm_raw_score")
                if isinstance(X, pd.DataFrame):
                    raw_scores = model.predict(X_values, raw_score=True)
                else:
                    raw_scores = model.predict(X, raw_score=True)

                if model_type == 'classification':
                    from scipy.special import expit
                    if isinstance(raw_scores, np.ndarray) and len(raw_scores.shape) == 1:
                        proba = expit(raw_scores)[0]
                        probabilities = [1-proba, proba]
                        prediction = 1 if proba > 0.5 else 0
                    else:
                        proba = expit(raw_scores[0])
                        probabilities = [1-proba, proba]
                        prediction = 1 if proba > 0.5 else 0
                else:
                    prediction = raw_scores[0]
                    probabilities = None
            except Exception as e:
                error_msg = f"{error_msg}; LightGBM raw_score method failed: {str(e)}"

        # Method 4: Use decision function (if available)
        if prediction is None and hasattr(model, 'decision_function'):
            try:
                methods_tried.append("decision_function")
                if isinstance(X, pd.DataFrame):
                    decision = model.decision_function(X_values)
                else:
                    decision = model.decision_function(X)

                from scipy.special import expit
                if len(decision.shape) == 1:
                    proba = expit(decision[0])
                    probabilities = [1-proba, proba]
                    prediction = 1 if proba > 0.5 else 0
                else:
                    proba = expit(decision[0])
                    probabilities = [1-proba, proba]
                    prediction = 1 if proba > 0.5 else 0
            except Exception as e:
                error_msg = f"{error_msg}; decision_function method failed: {str(e)}"

        # Method 5: For LightGBM, try direct Booster access
        if prediction is None and hasattr(model, '_Booster'):
            try:
                methods_tried.append("lightgbm_booster_direct")
                import lightgbm as lgb

                try:
                    if hasattr(model._Booster, 'model_file') and model._Booster.model_file:
                        predictor = lgb.Booster(model_file=model._Booster.model_file)
                    else:
                        model_str = model._Booster.model_str() if hasattr(model._Booster, 'model_str') else None
                        if model_str:
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                                f.write(model_str.encode())
                                temp_model_path = f.name
                            predictor = lgb.Booster(model_file=temp_model_path)
                            os.unlink(temp_model_path)
                        else:
                            raise ValueError("Cannot get LightGBM model file or model string")

                    if isinstance(X, pd.DataFrame):
                        raw_preds = predictor.predict(X_values)
                    else:
                        raw_preds = predictor.predict(X)

                    if model_type == 'classification':
                        if isinstance(raw_preds, np.ndarray) and len(raw_preds.shape) == 1:
                            prediction = 1 if raw_preds[0] > 0.5 else 0
                            probabilities = [1-raw_preds[0], raw_preds[0]]
                        else:
                            prediction = 1 if raw_preds[0][1] > 0.5 else 0
                            probabilities = raw_preds[0]
                    else:
                        prediction = raw_preds[0]
                except Exception as e:
                    raise ValueError(f"Cannot use Booster for prediction: {str(e)}")
            except Exception as e:
                error_msg = f"{error_msg}; LightGBM Booster direct access failed: {str(e)}"

    if prediction is None:
        if model_type == 'classification':
            import random
            prediction = random.randint(0, 1)
            probabilities = [1-prediction, prediction]
        else:
            prediction = 2.0

    try:
        prediction = float(prediction)
    except:
        prediction = 0.0

    if probabilities is not None and model_type == 'classification':
        if not isinstance(probabilities, list):
            try:
                probabilities = list(probabilities)
            except:
                probabilities = [0.5, 0.5]

        for i in range(len(probabilities)):
            try:
                probabilities[i] = float(probabilities[i])
            except:
                probabilities[i] = 0.0

        try:
            prob_value = float(probabilities[1])
        except:
            prob_value = 0.5

        if prob_value < 0.2:
            risk_level = 'low'
            min_prob, max_prob = 0.02, 0.90
        elif prob_value < 0.5:
            risk_level = 'medium'
            min_prob, max_prob = 0.05, 0.95
        else:
            risk_level = 'high'
            min_prob, max_prob = 0.10, 0.98

        smoothed_probs = []
        for i, prob in enumerate(probabilities):
            try:
                prob = float(prob)
            except:
                prob = 0.0

            if np.isnan(prob):
                prob = 0.0

            smoothed_prob = smooth_probability(prob, method='sigmoid_quantile',
                                              min_prob=min_prob, max_prob=max_prob)
            smoothed_probs.append(smoothed_prob)

        total = sum(smoothed_probs)
        if total > 0:
            probabilities = [float(p/total) for p in smoothed_probs]

    return prediction, probabilities, error_msg, methods_tried

class MultiDiseasePredictor:
    """Multi-disease joint prediction model for predicting the probability of having multiple diseases simultaneously"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}
        self.inference_models = {}
        self.common_features = ['age', 'gender', 'bmi', 'smoking', 'hypertension']
        self.feature_mapping = {
            'stroke': {
                'age': 'age',
                'gender': 'gender',
                'bmi': 'bmi',
                'smoking': 'smoking_status',
                'hypertension': 'hypertension'
            },
            'heart': {
                'age': 'Age',
                'gender': 'Sex',
                'bmi': None,
                'smoking': None,
                'hypertension': None
            },
            'cirrhosis': {
                'age': 'Age_years',
                'gender': 'Sex',
                'bmi': None,
                'smoking': None,
                'hypertension': None
            }
        }

        self.models_loaded = self.load_models()
        print(f"Model loading status: {'Success' if self.models_loaded else 'Using fallback models'}")

        self.init_calibrators()
        self._load_inference_bundles()

    def load_models(self):
        """Load pre-trained single-disease prediction models"""
        model_dir = 'output/models'
        models_loaded = False
        self.model_types = {}

        if os.path.exists(model_dir):
            try:
                model = joblib.load(os.path.join(model_dir, 'stroke_best_baseline_model.pkl'))
                self.models['stroke'] = model
                if 'lightgbm' in str(type(model)).lower():
                    self.model_types['stroke'] = 'lightgbm'
                else:
                    self.model_types['stroke'] = 'other'
                print("Successfully loaded stroke prediction model")
                models_loaded = True
            except Exception as e:
                print(f"Failed to load stroke model: {e}")
                from sklearn.ensemble import RandomForestClassifier
                self.models['stroke'] = RandomForestClassifier(n_estimators=10, random_state=42)
                self.model_types['stroke'] = 'other'
                print("Created stroke prediction fallback model")

            try:
                model = joblib.load(os.path.join(model_dir, 'heart_best_baseline_model.pkl'))
                self.models['heart'] = model
                if 'lightgbm' in str(type(model)).lower():
                    self.model_types['heart'] = 'lightgbm'
                else:
                    self.model_types['heart'] = 'other'
                print("Successfully loaded heart disease prediction model")
                models_loaded = True
            except Exception as e:
                print(f"Failed to load heart disease model: {e}")
                from sklearn.ensemble import RandomForestClassifier
                self.models['heart'] = RandomForestClassifier(n_estimators=10, random_state=42)
                self.model_types['heart'] = 'other'
                print("Created heart disease prediction fallback model")

            try:
                model = joblib.load(os.path.join(model_dir, 'cirrhosis_best_baseline_model.pkl'))
                self.models['cirrhosis'] = model
                if 'lightgbm' in str(type(model)).lower():
                    self.model_types['cirrhosis'] = 'lightgbm'
                else:
                    self.model_types['cirrhosis'] = 'other'
                print("Successfully loaded cirrhosis prediction model")
                models_loaded = True
            except Exception as e:
                print(f"Failed to load cirrhosis model: {e}")
                from sklearn.ensemble import RandomForestRegressor
                self.models['cirrhosis'] = RandomForestRegressor(n_estimators=10, random_state=42)
                self.model_types['cirrhosis'] = 'other'
                print("Created cirrhosis prediction fallback model")
        else:
            print(f"Model directory does not exist: {model_dir}")

        if not models_loaded:
            print("Warning: No models were successfully loaded, will use simulated data for prediction")

        return models_loaded

    def init_calibrators(self):
        """Initialize or load calibrators"""
        for disease_type in ['stroke', 'heart', 'cirrhosis']:
            self.calibrators[disease_type] = AdaptiveCalibrator()

            if not self.calibrators[disease_type].load_calibrators(disease_type):
                print(f"Calibrator for {disease_type} not found, will use direct calibration method")

    def _load_inference_bundles(self):
        """Load inference bundles from inference_predictor for correct feature transformation."""
        try:
            from inference_predictor import load_inference_models
            model_dir = os.path.join(os.path.dirname(__file__), 'output', 'inference_models')
            self.inference_models = load_inference_models(model_dir)
            if self.inference_models:
                print(f"Loaded inference bundles for multi-disease: {list(self.inference_models.keys())}")
            else:
                print("Warning: No inference bundles loaded in multi-disease model")
        except Exception as e:
            print(f"Failed to load inference bundles: {e}")
            self.inference_models = {}

    def predict_multi_disease_probability(self, data):
        """
        Predict the probability of multiple diseases occurring simultaneously

        Parameters:
            data: Dictionary containing user input data

        Returns:
            multi_probs: Dictionary containing probability combinations for multiple diseases
        """
        stroke_prob = self._predict_stroke_probability(data)
        heart_prob = self._predict_heart_probability(data)
        cirrhosis_prob = self._predict_cirrhosis_probability(data)

        multi_probs = {
            'stroke_only': stroke_prob * (1 - heart_prob) * (1 - cirrhosis_prob),
            'heart_only': (1 - stroke_prob) * heart_prob * (1 - cirrhosis_prob),
            'cirrhosis_only': (1 - stroke_prob) * (1 - heart_prob) * cirrhosis_prob,
            'stroke_heart': stroke_prob * heart_prob * (1 - cirrhosis_prob),
            'stroke_cirrhosis': stroke_prob * (1 - heart_prob) * cirrhosis_prob,
            'heart_cirrhosis': (1 - stroke_prob) * heart_prob * cirrhosis_prob,
            'all_three': stroke_prob * heart_prob * cirrhosis_prob,
            'none': (1 - stroke_prob) * (1 - heart_prob) * (1 - cirrhosis_prob)
        }

        multi_probs['stroke'] = stroke_prob
        multi_probs['heart'] = heart_prob
        multi_probs['cirrhosis'] = cirrhosis_prob

        return multi_probs

    def _predict_stroke_probability(self, data):
        """Predict stroke probability using inference bundle."""
        try:
            from inference_predictor import predict_single_disease
            if self.inference_models.get('stroke'):
                return predict_single_disease(self.inference_models, 'stroke', data)
        except Exception as e:
            print(f"Stroke inference failed: {e}")
        return 0.05

    def _predict_heart_probability(self, data):
        """Predict heart disease probability using inference bundle."""
        try:
            from inference_predictor import predict_single_disease
            if self.inference_models.get('heart'):
                return predict_single_disease(self.inference_models, 'heart', data)
        except Exception as e:
            print(f"Heart inference failed: {e}")
        return 0.05

    def _predict_cirrhosis_probability(self, data):
        """Predict cirrhosis severity and probability using inference bundle."""
        try:
            from inference_predictor import predict_single_disease
            if self.inference_models.get('cirrhosis'):
                return predict_single_disease(self.inference_models, 'cirrhosis', data)
        except Exception as e:
            print(f"Cirrhosis inference failed: {e}")
        return 0.05

    def predict_with_correlation(self, data):
        """
        Make predictions considering correlation between diseases

        Parameters:
            data: Dictionary containing user input data

        Returns:
            multi_probs_corr: Multi-disease probability dictionary after considering correlation
        """
        multi_probs = self.predict_multi_disease_probability(data)

        correlations = {
            'stroke_heart': 0.6,
            'stroke_cirrhosis': 0.2,
            'heart_cirrhosis': 0.3
        }

        stroke_prob = multi_probs['stroke']
        heart_prob = multi_probs['heart']
        cirrhosis_prob = multi_probs['cirrhosis']

        stroke_heart_prob = self._calculate_joint_prob(stroke_prob, heart_prob, correlations['stroke_heart'])
        stroke_cirrhosis_prob = self._calculate_joint_prob(stroke_prob, cirrhosis_prob, correlations['stroke_cirrhosis'])
        heart_cirrhosis_prob = self._calculate_joint_prob(heart_prob, cirrhosis_prob, correlations['heart_cirrhosis'])

        all_three_prob = stroke_heart_prob * cirrhosis_prob * 0.7

        multi_probs_corr = {
            'stroke_only': stroke_prob * (1 - stroke_heart_prob / stroke_prob) * (1 - stroke_cirrhosis_prob / stroke_prob),
            'heart_only': heart_prob * (1 - stroke_heart_prob / heart_prob) * (1 - heart_cirrhosis_prob / heart_prob),
            'cirrhosis_only': cirrhosis_prob * (1 - stroke_cirrhosis_prob / cirrhosis_prob) * (1 - heart_cirrhosis_prob / cirrhosis_prob),
            'stroke_heart': stroke_heart_prob * (1 - all_three_prob / stroke_heart_prob),
            'stroke_cirrhosis': stroke_cirrhosis_prob * (1 - all_three_prob / stroke_cirrhosis_prob),
            'heart_cirrhosis': heart_cirrhosis_prob * (1 - all_three_prob / heart_cirrhosis_prob),
            'all_three': all_three_prob,
            'none': max(0, 1 - stroke_prob - heart_prob - cirrhosis_prob + stroke_heart_prob + stroke_cirrhosis_prob + heart_cirrhosis_prob - all_three_prob)
        }

        multi_probs_corr['stroke'] = stroke_prob
        multi_probs_corr['heart'] = heart_prob
        multi_probs_corr['cirrhosis'] = cirrhosis_prob

        return multi_probs_corr

    def _calculate_joint_prob(self, p1, p2, corr):
        """
        Calculate joint probability considering correlation

        Parameters:
            p1, p2: Independent probabilities of two events
            corr: Correlation coefficient [-1, 1]

        Returns:
            Joint probability P(A,B)
        """
        if corr >= 0:
            return p1 * p2 + corr * min(p1, p2) * (1 - max(p1, p2))
        else:
            return p1 * p2 + corr * min(p1, 1-p2) * min(1-p1, p2)
