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
        """Predict stroke probability"""
        if 'stroke' not in self.models:
            return 0.05

        try:
            gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
            smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3,
                           'never_smoked': 0, 'formerly_smoked': 1}

            X = pd.DataFrame({
                'gender': [gender_map.get(data.get('gender'), 0)],
                'age': [float(data.get('age', 50))],
                'hypertension': [int(data.get('hypertension', 0))],
                'heart_disease': [int(data.get('heart_disease', 0))],
                'avg_glucose_level': [float(data.get('avg_glucose_level', 100))],
                'bmi': [float(data.get('bmi', 25))],
                'smoking_status': [smoking_map.get(data.get('smoking_status'), 3)]
            })

            prediction, probabilities, error_msg, methods_tried = robust_model_predict(
                self.models['stroke'], X, model_type='classification'
            )

            if error_msg:
                print(f"Issues encountered during stroke prediction, fallback methods tried: {methods_tried}. Error: {error_msg}")

            raw_prob = probabilities[1] if probabilities is not None else 0.05

            if 'stroke' in self.calibrators and self.calibrators['stroke'].is_fitted:
                calibrated_prob = self.calibrators['stroke'].calibrate_probability(X, self.models['stroke'], 'stroke')
                if calibrated_prob is not None:
                    return calibrated_prob[0]

            risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')

            if risk_level == 'high':
                return calibrate_probability(raw_prob, method='spline', risk_level='high')
            else:
                return calibrate_probability(raw_prob, method='logistic', risk_level=risk_level)

        except Exception as e:
            print(f"Stroke prediction error: {e}")
            return 0.05

    def _predict_heart_probability(self, data):
        """Predict heart disease probability"""
        if 'heart' not in self.models:
            return 0.05

        try:
            X = pd.DataFrame({
                'Age': [float(data.get('age', 50))],
                'Sex': [1 if data.get('gender') == 'Male' else 0],
                'ChestPainType': [str(data.get('chest_pain_type', 'ATA'))],
                'RestingBP': [float(data.get('resting_bp', 120))],
                'Cholesterol': [float(data.get('cholesterol', 200))],
                'FastingBS': [int(float(data.get('fasting_bs', 0)))],
                'RestingECG': [str(data.get('resting_ecg', 'Normal'))],
                'MaxHR': [float(data.get('max_hr', 150))],
                'ExerciseAngina': [1 if str(data.get('exercise_angina', '')) == 'Y' else 0],
                'Oldpeak': [float(data.get('oldpeak', 0))],
                'ST_Slope': [str(data.get('st_slope', 'Flat'))]
            })

            prediction, probabilities, error_msg, methods_tried = robust_model_predict(
                self.models['heart'], X, model_type='classification'
            )

            if error_msg:
                print(f"Issues encountered during heart disease prediction, fallback methods tried: {methods_tried}. Error: {error_msg}")

            raw_prob = probabilities[1] if probabilities is not None else 0.05

            if 'heart' in self.calibrators and self.calibrators['heart'].is_fitted:
                calibrated_prob = self.calibrators['heart'].calibrate_probability(X, self.models['heart'], 'heart')
                if calibrated_prob is not None:
                    return calibrated_prob[0]

            risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
            return calibrate_probability(raw_prob, method='sigmoid', risk_level=risk_level)

        except Exception as e:
            print(f"Heart disease prediction error: {e}")
            return 0.05

    def _predict_cirrhosis_probability(self, data):
        """Predict cirrhosis severity and probability"""
        if 'cirrhosis' not in self.models:
            return 0.05

        try:
            X = pd.DataFrame({
                'Age': [float(data.get('age', 50)) * 365.25],
                'Sex': [1 if data.get('gender') == 'Male' else 0],
                'Ascites': [int(data.get('ascites', 0))],
                'Hepatomegaly': [int(data.get('hepatomegaly', 0))],
                'Spiders': [int(data.get('spiders', 0))],
                'Edema': [int(data.get('edema', 0))],
                'Bilirubin': [float(data.get('bilirubin', 1.0))],
                'Cholesterol': [float(data.get('cholesterol', 200))],
                'Albumin': [float(data.get('albumin', 3.5))],
                'Copper': [float(data.get('copper', 50))],
                'Alk_Phos': [float(data.get('alk_phos', 100))],
                'SGOT': [float(data.get('sgot', 40))],
                'Tryglicerides': [float(data.get('tryglicerides', 150))],
                'Platelets': [float(data.get('platelets', 300))],
                'Prothrombin': [float(data.get('prothrombin', 10))]
            })

            prediction, _, error_msg, methods_tried = robust_model_predict(
                self.models['cirrhosis'], X, model_type='regression'
            )

            if error_msg:
                print(f"Issues encountered during cirrhosis prediction, fallback methods tried: {methods_tried}. Error: {error_msg}")

            raw_prob = min(max(prediction / 4, 0), 1.0)

            if 'cirrhosis' in self.calibrators and self.calibrators['cirrhosis'].is_fitted:
                calibrated_prob = self.calibrators['cirrhosis'].calibrate_probability(X, self.models['cirrhosis'], 'cirrhosis')
                if calibrated_prob is not None:
                    return calibrated_prob[0]

            risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
            return calibrate_probability(raw_prob, method='power', risk_level=risk_level)

        except Exception as e:
            print(f"Cirrhosis prediction error: {e}")
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
