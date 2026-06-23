import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import os
import joblib
import logging
from scipy.special import expit

os.makedirs('output', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from model_utilities import smooth_probability

class ProbabilityCalibrator:
    """
    Probability calibrator: Used to calibrate probabilities output by machine learning models
    Supports both Platt Scaling and Isotonic Regression calibration methods
    """

    def __init__(self, method='isotonic', cv=5):
        """
        Initialize probability calibrator

        Args:
            method: Calibration method, 'sigmoid' (Platt Scaling) or 'isotonic' (Isotonic Regression)
            cv: Number of cross-validation folds
        """
        self.method = method
        self.cv = cv
        self.calibrators = {}
        self.is_fitted = False

        os.makedirs('output/models/calibrators', exist_ok=True)

    def fit(self, X, y, model, disease_type):
        """
        Train calibrator

        Args:
            X: Feature data
            y: Label data
            model: Model to be calibrated
            disease_type: Disease type (stroke, heart, cirrhosis)
        """
        logger.info(f"Training {self.method} calibrator for {disease_type} disease model...")

        is_classifier = hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')

        if not is_classifier:
            logger.warning(f"{disease_type} model is not a classifier, cannot use CalibratedClassifierCV")
            logger.warning("Will create a simple probability mapping function as a fallback calibrator")

            try:
                predictions = model.predict(X)

                self.calibrators[disease_type] = {
                    'model': model,
                    'method': 'simple_mapping',
                    'mean': predictions.mean(),
                    'std': predictions.std() or 1.0
                }

                self.is_fitted = True

                joblib.dump(self.calibrators[disease_type],
                           f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl')
                logger.info(f"Simple mapping calibrator saved to output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl")

                return self

            except Exception as e:
                logger.error(f"Failed to create simple mapping calibrator: {e}")
                return self

        try:
            calibrator = CalibratedClassifierCV(
                estimator=model,
                method=self.method,
                cv=self.cv
            )

            calibrator.fit(X, y)

            self.calibrators[disease_type] = calibrator
            self.is_fitted = True

            joblib.dump(calibrator, f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl')
            logger.info(f"Calibrator saved to output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl")

        except Exception as e:
            logger.error(f"Failed to train calibrator: {e}")
            logger.warning("Will create a simple probability mapping function as a fallback calibrator")

            try:
                predictions = model.predict(X)

                self.calibrators[disease_type] = {
                    'model': model,
                    'method': 'simple_mapping',
                    'mean': predictions.mean(),
                    'std': predictions.std() or 1.0
                }

                self.is_fitted = True

                joblib.dump(self.calibrators[disease_type],
                           f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl')
                logger.info(f"Simple mapping calibrator saved")

            except Exception as e2:
                logger.error(f"Failed to create simple mapping calibrator: {e2}")

        return self

    def calibrate_probability(self, X, disease_type):
        """
        Calibrate predicted probabilities

        Args:
            X: Feature data
            disease_type: Disease type

        Returns:
            Calibrated probability
        """
        if not self.is_fitted or disease_type not in self.calibrators:
            logger.warning(f"Warning: Calibrator for {disease_type} not trained")
            return None

        calibrator = self.calibrators[disease_type]

        if isinstance(calibrator, dict) and calibrator.get('method') == 'simple_mapping':
            try:
                model = calibrator['model']
                predictions = model.predict(X)

                mean = calibrator['mean']
                std = calibrator['std']
                normalized = (predictions - mean) / std

                return expit(normalized)

            except Exception as e:
                logger.error(f"Failed to use simple mapping calibrator: {e}")
                return None

        try:
            calibrated_probs = calibrator.predict_proba(X)

            return calibrated_probs[:, 1]
        except Exception as e:
            logger.error(f"Failed to use standard calibrator: {e}")
            return None

    def load_calibrator(self, disease_type):
        """
        Load trained calibrator

        Args:
            disease_type: Disease type

        Returns:
            True if loaded successfully, otherwise False
        """
        try:
            calibrator_path = f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl'
            if os.path.exists(calibrator_path):
                self.calibrators[disease_type] = joblib.load(calibrator_path)
                self.is_fitted = True
                logger.info(f"Successfully loaded calibrator for {disease_type}")
                return True
            else:
                logger.warning(f"Calibrator file does not exist: {calibrator_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            return False

    def plot_calibration_curve(self, X, y, model, disease_type, n_bins=10):
        """
        Plot calibration curve

        Args:
            X: Feature data
            y: Label data
            model: Original model
            disease_type: Disease type
            n_bins: Number of bins
        """
        try:
            import numpy as np

            unique_values = np.unique(y)
            if len(unique_values) > 2 or not all(val in [0, 1] for val in unique_values):
                logger.warning(f"Target variable is not binary, contains values: {unique_values}")
                logger.warning("Attempting to convert target variable to binary form...")

                if len(unique_values) > 2:
                    threshold = np.median(y)
                    binary_y = (y > threshold).astype(int)
                    logger.info(f"Using threshold {threshold} to convert target variable to binary")
                else:
                    min_val = np.min(y)
                    binary_y = (y > min_val).astype(int)
                    logger.info(f"Converting target variable from {unique_values} to [0, 1]")
            else:
                binary_y = y

            plt.figure(figsize=(10, 8))

            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                else:
                    y_prob = model.predict(X)

                    if y_prob.ndim == 1:
                        min_val = y_prob.min()
                        max_val = y_prob.max()
                        if max_val > min_val:
                            y_prob = (y_prob - min_val) / (max_val - min_val)
                        else:
                            y_prob = np.ones_like(y_prob) * 0.5
            except ValueError as e:
                logger.warning(f"Failed to get original model predicted probabilities: {e}")
                logger.warning("Using random probabilities instead of original model predictions")
                y_prob = np.random.random(len(y))

            prob_true, prob_pred = calibration_curve(binary_y, y_prob, n_bins=n_bins)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Original Model')

            if disease_type in self.calibrators:
                y_prob_calibrated = self.calibrate_probability(X, disease_type)
                if y_prob_calibrated is not None:
                    prob_true_calibrated, prob_pred_calibrated = calibration_curve(binary_y, y_prob_calibrated, n_bins=n_bins)
                    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='s', linewidth=1, label=f'After Calibration ({self.method})')

            plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal Calibration')

            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Probability')
            plt.title(f'Calibration Curve for {disease_type.capitalize()} Disease Prediction Model')
            plt.legend(loc='best')
            plt.grid(True)

            os.makedirs('output/figures', exist_ok=True)
            plt.savefig(f'output/figures/{disease_type}_calibration_curve.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting calibration curve: {e}")
            logger.error(f"Error details: {str(e)}")
            logger.error("Skipping calibration curve plotting")


class AdaptiveCalibrator:
    """
    Adaptive calibrator: Uses different calibration strategies based on different risk levels
    """

    def __init__(self):
        """Initialize adaptive calibrator"""
        self.low_risk_calibrator = ProbabilityCalibrator(method='sigmoid')
        self.high_risk_calibrator = ProbabilityCalibrator(method='isotonic')
        self.risk_threshold = 0.3
        self.is_fitted = False
        self.disease_types = []

    def fit(self, X, y, model, disease_type):
        """
        Train adaptive calibrator

        Args:
            X: Feature data
            y: Label data
            model: Model to be calibrated
            disease_type: Disease type
        """
        self.low_risk_calibrator.fit(X, y, model, f"{disease_type}_low_risk")
        self.high_risk_calibrator.fit(X, y, model, f"{disease_type}_high_risk")

        self.is_fitted = True
        if disease_type not in self.disease_types:
            self.disease_types.append(disease_type)

        return self

    def calibrate_probability(self, X, model, disease_type):
        """
        Calibrate predicted probabilities

        Args:
            X: Feature data
            model: Original model
            disease_type: Disease type

        Returns:
            Calibrated probability
        """
        if not self.is_fitted:
            logger.warning(f"Warning: Adaptive calibrator not trained")
            return None

        if hasattr(model, 'predict_proba'):
            raw_probs = model.predict_proba(X)[:, 1]
        else:
            raw_probs = model.predict(X)

        calibrated_probs = np.zeros_like(raw_probs)

        low_risk_mask = raw_probs < self.risk_threshold
        if np.any(low_risk_mask):
            low_risk_probs = self.low_risk_calibrator.calibrate_probability(
                X[low_risk_mask], f"{disease_type}_low_risk")
            if low_risk_probs is not None:
                calibrated_probs[low_risk_mask] = low_risk_probs
            else:
                calibrated_probs[low_risk_mask] = raw_probs[low_risk_mask]

        high_risk_mask = ~low_risk_mask
        if np.any(high_risk_mask):
            high_risk_probs = self.high_risk_calibrator.calibrate_probability(
                X[high_risk_mask], f"{disease_type}_high_risk")
            if high_risk_probs is not None:
                calibrated_probs[high_risk_mask] = high_risk_probs
            else:
                calibrated_probs[high_risk_mask] = raw_probs[high_risk_mask]

        return calibrated_probs

    def load_calibrators(self, disease_type):
        """
        Load trained calibrators

        Args:
            disease_type: Disease type

        Returns:
            True if loaded successfully, otherwise False
        """
        low_risk_loaded = self.low_risk_calibrator.load_calibrator(f"{disease_type}_low_risk")
        high_risk_loaded = self.high_risk_calibrator.load_calibrator(f"{disease_type}_high_risk")

        if low_risk_loaded and high_risk_loaded:
            self.is_fitted = True
            if disease_type not in self.disease_types:
                self.disease_types.append(disease_type)
            return True
        else:
            return False


class BetterCalibrator:
    """
    Enhanced calibrator: Specifically for calibrating low-risk patients
    """

    def __init__(self):
        """Initialize enhanced calibrator"""
        self.platt_calibrator = ProbabilityCalibrator(method='sigmoid')
        self.isotonic_calibrator = ProbabilityCalibrator(method='isotonic')
        self.risk_threshold = 0.25
        self.is_fitted = False
        self.disease_types = []

    def fit(self, X, y, model, disease_type):
        """
        Train enhanced calibrator

        Args:
            X: Feature data
            y: Label data
            model: Model to be calibrated
            disease_type: Disease type
        """
        self.platt_calibrator.fit(X, y, model, f"{disease_type}_better_low_risk")
        self.isotonic_calibrator.fit(X, y, model, f"{disease_type}_better_high_risk")

        self.is_fitted = True
        if disease_type not in self.disease_types:
            self.disease_types.append(disease_type)

        return self

    def calibrate_probability(self, X, model, disease_type):
        """
        Calibrate predicted probabilities

        Args:
            X: Feature data
            model: Original model
            disease_type: Disease type

        Returns:
            Calibrated probability
        """
        if not self.is_fitted:
            logger.warning(f"Warning: Enhanced calibrator not trained")
            return None

        if hasattr(model, 'predict_proba'):
            raw_probs = model.predict_proba(X)[:, 1]
        else:
            raw_probs = model.predict(X)

        calibrated_probs = np.zeros_like(raw_probs)

        low_risk_mask = raw_probs < self.risk_threshold
        if np.any(low_risk_mask):
            low_risk_probs = self.platt_calibrator.calibrate_probability(
                X[low_risk_mask], f"{disease_type}_better_low_risk")
            if low_risk_probs is not None:
                calibrated_probs[low_risk_mask] = low_risk_probs * 1.2
            else:
                calibrated_probs[low_risk_mask] = raw_probs[low_risk_mask]

        high_risk_mask = ~low_risk_mask
        if np.any(high_risk_mask):
            high_risk_probs = self.isotonic_calibrator.calibrate_probability(
                X[high_risk_mask], f"{disease_type}_better_high_risk")
            if high_risk_probs is not None:
                calibrated_probs[high_risk_mask] = high_risk_probs
            else:
                calibrated_probs[high_risk_mask] = raw_probs[high_risk_mask]

        return np.clip(calibrated_probs, 0, 1)

    def load_calibrators(self, disease_type):
        """
        Load trained calibrators

        Args:
            disease_type: Disease type

        Returns:
            True if loaded successfully, otherwise False
        """
        low_risk_loaded = self.platt_calibrator.load_calibrator(f"{disease_type}_better_low_risk")
        high_risk_loaded = self.isotonic_calibrator.load_calibrator(f"{disease_type}_better_high_risk")

        if low_risk_loaded and high_risk_loaded:
            self.is_fitted = True
            if disease_type not in self.disease_types:
                self.disease_types.append(disease_type)
            return True
        else:
            return False


def calibrate_probability(prob, method='spline', risk_level='high'):
    """
    Directly calibrate a single probability value

    Args:
        prob: Original probability value
        method: Calibration method ('spline', 'power', 'logistic')
        risk_level: Risk level ('low', 'medium', 'high')

    Returns:
        Calibrated probability value
    """
    if risk_level == 'low':
        power = 0.7
        logistic_a = 1.4
        logistic_b = -0.1
    elif risk_level == 'medium':
        power = 0.5
        logistic_a = 1.8
        logistic_b = 0
    else:
        power = 0.4
        logistic_a = 2.5
        logistic_b = 0.1

    calibrated_prob = None

    if method == 'power':
        calibrated_prob = np.power(prob, power)

    elif method == 'logistic':
        calibrated_prob = 1 / (1 + np.exp(-logistic_a * (prob - logistic_b)))

    elif method == 'spline':
        if prob < 0.1:
            calibrated_prob = prob * 2.0
        elif prob < 0.3:
            calibrated_prob = 0.2 + (prob - 0.1) * 1.8
        elif prob < 0.6:
            calibrated_prob = 0.56 + (prob - 0.3) * 1.4
        else:
            calibrated_prob = min(0.98, 0.98 + (prob - 0.6) * 0.05)

    else:
        calibrated_prob = prob

    if risk_level == 'low':
        min_prob, max_prob = 0.02, 0.90
        smooth_method = 'sigmoid_quantile'
    elif risk_level == 'medium':
        min_prob, max_prob = 0.05, 0.95
        smooth_method = 'sigmoid_quantile'
    else:
        min_prob, max_prob = 0.10, 0.98
        smooth_method = 'sigmoid_quantile'

    smoothed_prob = smooth_probability(calibrated_prob, method=smooth_method,
                                       min_prob=min_prob, max_prob=max_prob)

    return smoothed_prob


def load_data_and_models():
    """Load processed data and trained models"""
    data = {}
    models = {}

    data_dir = 'output/processed_data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                data_name = filename.replace('_processed.csv', '')
                try:
                    data[data_name] = pd.read_csv(data_path)
                    logger.info(f"Successfully loaded data: {data_name}")
                except Exception as e:
                    logger.error(f"Failed to load data {data_name}: {e}")

    model_dir = 'output/models'
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl') and 'best_baseline_model' in filename:
                model_path = os.path.join(model_dir, filename)
                model_name = filename.replace('.pkl', '')
                try:
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"Successfully loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")

    return data, models

def prepare_data_for_calibration(data):
    """Prepare data for calibration"""
    datasets = {}

    if 'stroke' in data:
        stroke_features = data['stroke'].drop(['id', 'stroke'], axis=1, errors='ignore')
        stroke_target = data['stroke']['stroke']
        datasets['stroke'] = (stroke_features, stroke_target)

    if 'heart' in data:
        heart_features = data['heart'].drop(['HeartDisease'], axis=1, errors='ignore')
        heart_target = data['heart']['HeartDisease']
        datasets['heart'] = (heart_features, heart_target)

    if 'cirrhosis' in data:
        cirrhosis_features = data['cirrhosis'].drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
        cirrhosis_target = data['cirrhosis']['Stage']
        datasets['cirrhosis'] = (cirrhosis_features, cirrhosis_target)

    return datasets

def train_calibrators(datasets, models):
    """Train probability calibrators"""
    calibrators = {}

    os.makedirs('output/models/calibrators', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)

    for disease_type, (features, target) in datasets.items():
        logger.info(f"\nStarting to train calibrator for {disease_type}...")

        model_key = f"{disease_type}_best_baseline_model"
        if model_key not in models:
            logger.warning(f"Model for {disease_type} not found, skipping calibrator training")
            continue

        model = models[model_key]

        try:
            X_train, X_cal, y_train, y_cal = train_test_split(
                features, target, test_size=0.3, random_state=42, stratify=target
            )
        except ValueError as e:
            logger.warning(f"Cannot use stratified sampling: {e}")
            logger.warning("Using random sampling instead (without stratify parameter)")
            X_train, X_cal, y_train, y_cal = train_test_split(
                features, target, test_size=0.3, random_state=42
            )

        if not isinstance(X_cal, pd.DataFrame):
            X_cal = pd.DataFrame(X_cal, columns=features.columns)

        logger.info(f"Training Platt Scaling calibrator for {disease_type}...")
        platt_calibrator = ProbabilityCalibrator(method='sigmoid')
        platt_calibrator.fit(X_cal, y_cal, model, disease_type)

        logger.info(f"Training Isotonic Regression calibrator for {disease_type}...")
        isotonic_calibrator = ProbabilityCalibrator(method='isotonic')
        isotonic_calibrator.fit(X_cal, y_cal, model, disease_type)

        logger.info(f"Training adaptive calibrator for {disease_type}...")
        adaptive_calibrator = AdaptiveCalibrator()
        adaptive_calibrator.fit(X_cal, y_cal, model, disease_type)

        logger.info(f"Training enhanced calibrator for {disease_type}...")
        better_calibrator = BetterCalibrator()
        better_calibrator.fit(X_cal, y_cal, model, disease_type)

        logger.info(f"Plotting calibration curve for {disease_type}...")
        platt_calibrator.plot_calibration_curve(X_cal, y_cal, model, disease_type)

        evaluate_calibrator_performance(X_cal, y_cal, model, platt_calibrator, isotonic_calibrator, disease_type)

        calibrators[disease_type] = {
            'platt': platt_calibrator,
            'isotonic': isotonic_calibrator,
            'adaptive': adaptive_calibrator,
            'better': better_calibrator
        }

    return calibrators

def evaluate_calibrator_performance(X, y, model, platt_calibrator, isotonic_calibrator, disease_type):
    """Evaluate calibrator performance"""
    logger.info(f"\nEvaluating calibrator performance for {disease_type}...")

    try:
        import numpy as np

        unique_values = np.unique(y)
        if len(unique_values) > 2 or not all(val in [0, 1] for val in unique_values):
            logger.warning(f"Target variable is not binary, contains values: {unique_values}")
            logger.warning("Attempting to convert target variable to binary form...")

            if len(unique_values) > 2:
                threshold = np.median(y)
                binary_y = (y > threshold).astype(int)
                logger.info(f"Using threshold {threshold} to convert target variable to binary")
            else:
                min_val = np.min(y)
                binary_y = (y > min_val).astype(int)
                logger.info(f"Converting target variable from {unique_values} to [0, 1]")
        else:
            binary_y = y

        try:
            if hasattr(model, 'predict_proba'):
                y_prob_orig = model.predict_proba(X)[:, 1]
            else:
                y_prob_orig = model.predict(X)

                if y_prob_orig.ndim == 1:
                    min_val = y_prob_orig.min()
                    max_val = y_prob_orig.max()
                    if max_val > min_val:
                        y_prob_orig = (y_prob_orig - min_val) / (max_val - min_val)
                    else:
                        y_prob_orig = np.ones_like(y_prob_orig) * 0.5
        except ValueError as e:
            logger.warning(f"Failed to get original model predicted probabilities: {e}")
            logger.warning("Using random probabilities instead of original model predictions")
            y_prob_orig = np.random.random(len(y))

        y_prob_platt = platt_calibrator.calibrate_probability(X, disease_type)
        y_prob_isotonic = isotonic_calibrator.calibrate_probability(X, disease_type)

        if y_prob_platt is None:
            logger.warning(f"Platt calibration failed, using original probabilities instead")
            y_prob_platt = y_prob_orig

        if y_prob_isotonic is None:
            logger.warning(f"Isotonic calibration failed, using original probabilities instead")
            y_prob_isotonic = y_prob_orig

        y_prob_orig = np.clip(y_prob_orig, 0, 1)
        y_prob_platt = np.clip(y_prob_platt, 0, 1)
        y_prob_isotonic = np.clip(y_prob_isotonic, 0, 1)

        brier_orig = brier_score_loss(binary_y, y_prob_orig)
        brier_platt = brier_score_loss(binary_y, y_prob_platt)
        brier_isotonic = brier_score_loss(binary_y, y_prob_isotonic)

        eps = 1e-15
        y_prob_orig = np.clip(y_prob_orig, eps, 1 - eps)
        y_prob_platt = np.clip(y_prob_platt, eps, 1 - eps)
        y_prob_isotonic = np.clip(y_prob_isotonic, eps, 1 - eps)

        log_loss_orig = log_loss(binary_y, y_prob_orig)
        log_loss_platt = log_loss(binary_y, y_prob_platt)
        log_loss_isotonic = log_loss(binary_y, y_prob_isotonic)

        logger.info(f"Brier Score (lower is better):")
        logger.info(f"  Original Model: {brier_orig:.4f}")
        logger.info(f"  Platt Calibration: {brier_platt:.4f}")
        logger.info(f"  Isotonic Calibration: {brier_isotonic:.4f}")

        logger.info(f"Log Loss (lower is better):")
        logger.info(f"  Original Model: {log_loss_orig:.4f}")
        logger.info(f"  Platt Calibration: {log_loss_platt:.4f}")
        logger.info(f"  Isotonic Calibration: {log_loss_isotonic:.4f}")

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(['Original Model', 'Platt Cal', 'Isotonic Cal'], [brier_orig, brier_platt, brier_isotonic])
        plt.title(f'{disease_type.capitalize()} Brier Score Comparison (Lower is Better)')
        plt.ylabel('Brier Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.bar(['Original Model', 'Platt Cal', 'Isotonic Cal'], [log_loss_orig, log_loss_platt, log_loss_isotonic])
        plt.title(f'{disease_type.capitalize()} Log Loss Comparison (Lower is Better)')
        plt.ylabel('Log Loss')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'output/figures/{disease_type}_calibration_performance.png')
        plt.close()

    except Exception as e:
        logger.error(f"Error evaluating calibrator performance: {e}")
        logger.error(f"Error details: {str(e)}")
        logger.error("Skipping calibrator performance evaluation")

def run_calibration_training():
    """Run calibrator training workflow"""
    logger.info("Starting probability calibrator training...")

    data, models = load_data_and_models()

    datasets = prepare_data_for_calibration(data)

    calibrators = train_calibrators(datasets, models)

    logger.info("\nCalibrator training completed!")
    return calibrators

if __name__ == "__main__":
    run_calibration_training()
