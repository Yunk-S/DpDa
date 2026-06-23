import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

class DNN(nn.Module):
    """Simple deep neural network model"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, dropout_rate=0.3):
        super(DNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.is_regression = False  # Default to classification task
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # If output_dim is not 1, this is a multi-class problem, do not use sigmoid
        if output_dim != 1:
            self.is_regression = True
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        # Only use sigmoid activation for binary classification
        if self.output_layer.out_features == 1 and not self.is_regression:
            x = self.sigmoid(x)
            
        return x

class AttentionDNN(nn.Module):
    """Deep neural network model with attention mechanism"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, dropout_rate=0.3):
        super(AttentionDNN, self).__init__()
        
        self.feature_layers = nn.ModuleList()
        self.is_regression = False  # Default to classification task
        
        # Feature extraction layers
        self.feature_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.feature_layers.append(nn.ReLU())
        self.feature_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.feature_layers.append(nn.Dropout(dropout_rate))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Sigmoid()
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.hidden_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # If output_dim is not 1, this is a multi-class problem, do not use sigmoid
        if output_dim != 1:
            self.is_regression = True
        
    def forward(self, x):
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
        
        # Attention weights
        attention_weights = self.attention(x)
        
        # Apply attention weights
        x = x * attention_weights
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Only use sigmoid activation for binary classification
        if self.output_layer.out_features == 1 and not self.is_regression:
            x = self.sigmoid(x)
            
        return x

class ModelTrainer:
    def __init__(self, data_splits=None):
        """Initialize the model training class"""
        self.models = {}
        self.best_models = {}
        self.model_results = {}
        self.feature_importances = {}
        self.shap_values = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store deep learning models
        self.dl_models = {}
        self.teacher_models = {}
        self.student_models = {}
        
        # Store ensemble models
        self.ensemble_models = {}
        
        # Store blended models
        self.blended_models = {}
        
        # Store class weights
        self.class_weights = {}
        
        # Store data splits
        self.data_splits = data_splits if data_splits else {}
        
        # Create output directories
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/figures', exist_ok=True)
        
    def load_processed_data(self, stroke_path='output/processed_data/stroke_processed.csv',
                           heart_path='output/processed_data/heart_processed.csv',
                           cirrhosis_path='output/processed_data/cirrhosis_processed.csv'):
        """Load processed data"""
        self.stroke_data = pd.read_csv(stroke_path)
        self.heart_data = pd.read_csv(heart_path)
        self.cirrhosis_data = pd.read_csv(cirrhosis_path)
        
        # Prepare training data
        # Stroke dataset
        stroke_features = self.stroke_data.drop(['id', 'stroke'], axis=1, errors='ignore')
        stroke_target = self.stroke_data['stroke']
        
        # Heart disease dataset
        heart_features = self.heart_data.drop(['HeartDisease'], axis=1, errors='ignore')
        heart_target = self.heart_data['HeartDisease']
        
        # Cirrhosis dataset
        cirrhosis_features = self.cirrhosis_data.drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
        cirrhosis_target = self.cirrhosis_data['Stage']
        
        self.datasets = {
            'stroke': (stroke_features, stroke_target),
            'heart': (heart_features, heart_target),
            'cirrhosis': (cirrhosis_features, cirrhosis_target)
        }
        
        print("Processed data loaded successfully!")
        return self.datasets
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.train_test_data = {}
        
        for name, (features, target) in self.datasets.items():
            # Remove non-numeric columns
            features = features.select_dtypes(include=['float64', 'int64'])
            
            # Check if stratified sampling can be used
            use_stratify = True
            if name == 'cirrhosis':  # Special handling for cirrhosis dataset
                # Check sample count for each class
                value_counts = pd.Series(target).value_counts()
                if value_counts.min() < 2:  # If minimum class count is less than 2, stratified sampling cannot be used
                    use_stratify = False
                    print(f"Warning: {name} dataset has a class with too few samples, not using stratified sampling")
            
            # Split data
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=test_size, random_state=random_state, stratify=target
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=test_size, random_state=random_state
                )
            
            self.train_test_data[name] = (X_train, X_test, y_train, y_test)
            print(f"{name} dataset split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return self.train_test_data
    
    def handle_imbalanced_data(self, method='smote'):
        """Handle imbalanced data"""
        self.class_weights = {}  # Initialize class weights dictionary
        
        for name, (X_train, X_test, y_train, y_test) in self.train_test_data.items():
            # Determine if it's a regression task
            is_regression = False
            if name == 'cirrhosis':  # Cirrhosis dataset's Stage column is continuous, should be treated as regression
                is_regression = True
                
            # For classification tasks, check for class imbalance
            unique_counts = np.unique(y_train, return_counts=True)
            class_ratio = min(unique_counts[1]) / max(unique_counts[1])
            
            if class_ratio < 0.5:  # If small class has less than half of large class samples, consider it imbalanced
                print(f"{name} dataset has class imbalance, applying {method} method...")
                
                if method == 'smote':
                    # Apply SMOTE oversampling
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    
                    # Update training data
                    self.train_test_data[name] = (X_train_resampled, X_test, y_train_resampled, y_test)
                    
                    # Print resampled class distribution
                    unique_counts_after = np.unique(y_train_resampled, return_counts=True)
                    print(f"  Before resampling: {dict(zip(unique_counts[0], unique_counts[1]))}")
                    print(f"  After resampling: {dict(zip(unique_counts_after[0], unique_counts_after[1]))}")
                    
                elif method == 'class_weight':
                    # Calculate class weights (used when training models later)
                    counts = np.bincount(y_train)
                    class_weight = {i: max(counts) / counts[i] for i in range(len(counts))}
                    self.class_weights[name] = class_weight
                    print(f"  Applied class weights: {class_weight}")
            else:
                print(f"{name} dataset class distribution is relatively balanced, no special treatment needed")
        
        return self.train_test_data
    
    def train_baseline_models(self):
        """Train and evaluate all models, finding the best performing ones"""
        # Baseline model list - Classification models
        self.baseline_models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
        }
        
        # Regression model list - For cirrhosis dataset
        self.regression_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'SVR': SVR(kernel='rbf'),
            'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=5),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, max_depth=5),
            'XGBRegressor': XGBRegressor(n_estimators=100, max_depth=5),
            'LGBMRegressor': LGBMRegressor(n_estimators=100, max_depth=5)
        }
        
        results = {}
        best_models = {}
        
        # For each dataset
        for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
            print(f"\nTraining baseline models for {dataset_name} dataset...")
            dataset_scores = {}
            
            # Determine if it's regression or classification task
            is_regression = False
            if dataset_name == 'cirrhosis':  # Cirrhosis dataset's Stage is continuous
                is_regression = True
            
            if not is_regression:
                # Classification task
                for model_name, model in self.baseline_models.items():
                    try:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        print(f"  Training {model_name}...")
                        print(f"    CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
                        print(f"    Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
                        
                        dataset_scores[model_name] = {
                            'model': model,
                            'cv_accuracy': cv_scores.mean(),
                            'test_accuracy': accuracy,
                            'test_f1': f1
                        }
                    except Exception as e:
                        print(f"  Error training {model_name}: {e}")
            else:
                # Regression task
                for model_name, model in self.regression_models.items():
                    try:
                        # Use KFold for cross-validation
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = []
                        
                        # Manually perform cross-validation to calculate MSE
                        for train_idx, val_idx in kf.split(X_train):
                            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                            
                            model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = model.predict(X_val_cv)
                            mse_cv = mean_squared_error(y_val_cv, y_pred_cv)
                            cv_scores.append(mse_cv)
                        
                        cv_mse = np.mean(cv_scores)
                        cv_std = np.std(cv_scores)
                        
                        # Train model on full training set
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        print(f"  Training {model_name}...")
                        print(f"    CV MSE: {cv_mse:.4f} +/- {cv_std:.4f}")
                        print(f"    Test MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                        
                        dataset_scores[model_name] = {
                            'model': model,
                            'cv_mse': cv_mse,
                            'test_mse': mse,
                            'test_mae': mae,
                            'test_r2': r2
                        }
                    except Exception as e:
                        print(f"  Error training {model_name}: {e}")
            
            # Select best model
            if dataset_scores:
                if not is_regression:
                    # Classification task uses F1 score
                    best_model_name = max(dataset_scores, key=lambda k: dataset_scores[k]['test_f1'])
                    print(f"  Best baseline model for {dataset_name} dataset: {best_model_name}")
                    best_models[dataset_name] = dataset_scores[best_model_name]['model']
                else:
                    # Regression task uses R2 score
                    best_model_name = max(dataset_scores, key=lambda k: dataset_scores[k]['test_r2'])
                    print(f"  Best baseline model for {dataset_name} dataset: {best_model_name}")
                    best_models[dataset_name] = dataset_scores[best_model_name]['model']
            else:
                print(f"  No models successfully trained for {dataset_name} dataset")
            
            results[dataset_name] = dataset_scores
        
        self.baseline_results = results
        self.best_baseline_models = best_models
        
        # Save best baseline models
        for dataset_name, model in best_models.items():
            joblib.dump(model, f'output/models/{dataset_name}_best_baseline_model.pkl')
        
        return best_models
    
    def optimize_best_models(self, cv=5):
        """Optimize hyperparameters of best models"""
        for name, best_model_info in self.best_models.items():
            print(f"\nOptimizing {best_model_info['name']} model for {name} dataset...")
            
            X_train, X_test, y_train, y_test = self.train_test_data[name]
            model_name = best_model_info['name']
            
            # Define parameter grid based on model type
            if model_name == 'LogisticRegression':
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['liblinear', 'saga']
                }
            elif model_name == 'DecisionTree':
                param_grid = {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            elif model_name == 'RandomForest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            elif model_name == 'GradientBoosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            elif model_name == 'LightGBM':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 70],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif model_name == 'CatBoost':
                param_grid = {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5, 7]
                }
            else:
                print(f"  No parameter grid defined for {model_name}, skipping optimization")
                continue
            
            # Create grid search object
            grid_search = GridSearchCV(
                estimator=best_model_info['model'],
                param_grid=param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Perform grid search
            try:
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                print(f"  Best parameters: {best_params}")
                print(f"  CV Score: {grid_search.best_score_:.4f}")
                
                # Evaluate best model on test set
                y_pred = best_model.predict(X_test)
                
                # Calculate evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                if len(np.unique(y_test)) == 2:  # Binary classification
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    # Calculate ROC AUC
                    try:
                        y_prob = best_model.predict_proba(X_test)[:, 1]
                        auc_score = roc_auc_score(y_test, y_prob)
                    except:
                        auc_score = None
                else:  # Multi-class classification
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    auc_score = None
                
                print(f"  Test Accuracy: {accuracy:.4f}")
                print(f"  Test Precision: {precision:.4f}")
                print(f"  Test Recall: {recall:.4f}")
                print(f"  Test F1 Score: {f1:.4f}")
                if auc_score:
                    print(f"  Test AUC: {auc_score:.4f}")
                
                # Update best model
                self.best_models[name]['model'] = best_model
                self.best_models[name]['params'] = best_params
                self.best_models[name]['performance'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc_score
                }
                
                # Save model
                joblib.dump(best_model, f'output/models/{name}_{model_name}_optimized.pkl')
                print(f"  Optimized model saved to output/models/{name}_{model_name}_optimized.pkl")
                
            except Exception as e:
                print(f"  Error optimizing {model_name}: {e}")
        
        return self.best_models
    
    def evaluate_and_visualize_models(self):
        """Evaluate best models and visualize results"""
        for name, best_model_info in self.best_models.items():
            print(f"\nEvaluating {best_model_info['name']} model for {name} dataset...")
            
            X_train, X_test, y_train, y_test = self.train_test_data[name]
            model = best_model_info['model']
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name.capitalize()} {best_model_info["name"]} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f'output/figures/{name}_confusion_matrix.png')
            plt.close()
            
            # Classification report
            report = classification_report(y_test, y_pred)
            print(f"  Classification Report:\n{report}")
            
            # ROC curve (only for binary classification)
            if len(np.unique(y_test)) == 2:
                try:
                    # Get prediction probabilities
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, 
                             label=f'ROC Curve (AUC = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{name.capitalize()} {best_model_info["name"]} ROC Curve')
                    plt.legend(loc='lower right')
                    plt.savefig(f'output/figures/{name}_roc_curve.png')
                    plt.close()
                    
                except Exception as e:
                    print(f"  Error plotting ROC curve: {e}")
            
            # Feature importance (if supported by model)
            try:
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X_train.columns
                    
                    # Sort
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 6))
                    plt.title(f'{name.capitalize()} {best_model_info["name"]} Feature Importance')
                    plt.bar(range(len(indices)), importances[indices], align='center')
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{name}_feature_importance.png')
                    plt.close()
                    
                    # Save feature importance
                    self.feature_importances[name] = {
                        'importance': importances,
                        'names': feature_names
                    }
                    
                    # SHAP value analysis
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)
                        
                        # Save SHAP values
                        self.shap_values[name] = {
                            'values': shap_values,
                            'data': X_test
                        }
                        
                        # Plot SHAP summary
                        plt.figure(figsize=(10, 8))
                        if isinstance(shap_values, list):  # Multi-class case
                            shap.summary_plot(shap_values[1], X_test, show=False)
                        else:  # Binary classification case
                            shap.summary_plot(shap_values, X_test, show=False)
                        plt.title(f'{name.capitalize()} {best_model_info["name"]} SHAP Values')
                        plt.tight_layout()
                        plt.savefig(f'output/figures/{name}_shap_summary.png')
                        plt.close()
                        
                    except Exception as e:
                        print(f"  Error in SHAP analysis: {e}")
                
            except Exception as e:
                print(f"  Error analyzing feature importance: {e}")
        
        return self.feature_importances, self.shap_values
    
    def train_deep_learning_models(self, epochs=100, batch_size=32, patience=10):
        """Train deep learning models"""
        print("\nStarting deep learning model training...")
        
        # Create model structures
        for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
            print(f"\nTraining deep learning models for {dataset_name} dataset...")
            
            # Determine task type
            is_regression = False
            if dataset_name == 'cirrhosis':  # Cirrhosis dataset is a regression task
                is_regression = True
            
            # Prepare data - ensure all data is numeric type
            try:
                # Try direct conversion
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            except (TypeError, ValueError):
                # If fails, try converting to numeric type first
                X_train = X_train.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_test = X_test.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            
            y_train_tensor = torch.FloatTensor(y_train.values)
            y_test_tensor = torch.FloatTensor(y_test.values)
            
            # Reshape labels to column vector
            y_train_tensor = y_train_tensor.reshape(-1, 1)
            y_test_tensor = y_test_tensor.reshape(-1, 1)
            
            # Create DataLoader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create model
            input_dim = X_train.shape[1]
            
            if not is_regression:
                # Classification model
                dnn_model = DNN(input_dim=input_dim).to(self.device)
                attention_model = AttentionDNN(input_dim=input_dim).to(self.device)
                criterion = nn.BCELoss()
            else:
                # Regression model - set is_regression=True
                dnn_model = DNN(input_dim=input_dim)
                dnn_model.is_regression = True
                dnn_model = dnn_model.to(self.device)
                
                attention_model = AttentionDNN(input_dim=input_dim)
                attention_model.is_regression = True
                attention_model = attention_model.to(self.device)
                
                criterion = nn.MSELoss()
            
            # Optimizer
            dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
            attention_optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
            
            # Train model
            best_dnn_loss = float('inf')
            best_att_loss = float('inf')
            dnn_patience_counter = 0
            att_patience_counter = 0
            
            for epoch in range(1, epochs + 1):
                dnn_model.train()
                attention_model.train()
                
                dnn_epoch_loss = 0
                att_epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Train DNN
                    dnn_optimizer.zero_grad()
                    dnn_outputs = dnn_model(batch_X)
                    dnn_loss = criterion(dnn_outputs, batch_y)
                    dnn_loss.backward()
                    dnn_optimizer.step()
                    dnn_epoch_loss += dnn_loss.item()
                    
                    # Train Attention DNN
                    attention_optimizer.zero_grad()
                    att_outputs = attention_model(batch_X)
                    att_loss = criterion(att_outputs, batch_y)
                    att_loss.backward()
                    attention_optimizer.step()
                    att_epoch_loss += att_loss.item()
                
                dnn_epoch_loss /= len(train_loader)
                att_epoch_loss /= len(train_loader)
                
                # Print training progress
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}/{epochs}, DNN Loss: {dnn_epoch_loss:.4f}, Attention Loss: {att_epoch_loss:.4f}")
                
                # Early stopping
                if dnn_epoch_loss < best_dnn_loss:
                    best_dnn_loss = dnn_epoch_loss
                    dnn_patience_counter = 0
                else:
                    dnn_patience_counter += 1
                
                if att_epoch_loss < best_att_loss:
                    best_att_loss = att_epoch_loss
                    att_patience_counter = 0
                else:
                    att_patience_counter += 1
                
                if dnn_patience_counter >= patience and att_patience_counter >= patience:
                    print(f"  Early stopping training, no improvement: {patience} epochs")
                    break
            
            # Evaluate model
            dnn_model.eval()
            attention_model.eval()
            
            with torch.no_grad():
                dnn_outputs = dnn_model(X_test_tensor.to(self.device))
                att_outputs = attention_model(X_test_tensor.to(self.device))
                
                dnn_preds = dnn_outputs.cpu().numpy()
                att_preds = att_outputs.cpu().numpy()
                y_test_np = y_test.values
                
                if not is_regression:
                    # Classification task evaluation
                    dnn_predictions = (dnn_preds > 0.5).astype(int).flatten()
                    att_predictions = (att_preds > 0.5).astype(int).flatten()
                    
                    dnn_accuracy = accuracy_score(y_test_np, dnn_predictions)
                    att_accuracy = accuracy_score(y_test_np, att_predictions)
                    
                    dnn_f1 = f1_score(y_test_np, dnn_predictions, average='weighted')
                    att_f1 = f1_score(y_test_np, att_predictions, average='weighted')
                    
                    print(f"  DNN Accuracy: {dnn_accuracy:.4f}, F1: {dnn_f1:.4f}")
                    print(f"  Attention DNN Accuracy: {att_accuracy:.4f}, F1: {att_f1:.4f}")
                    
                    # Try to calculate AUC (for binary classification)
                    try:
                        dnn_auc = roc_auc_score(y_test_np, dnn_preds)
                        att_auc = roc_auc_score(y_test_np, att_preds)
                        print(f"  DNN AUC: {dnn_auc:.4f}")
                        print(f"  Attention DNN AUC: {att_auc:.4f}")
                    except:
                        pass
                    
                    # Select better performing model as teacher
                    if att_f1 > dnn_f1:
                        self.teacher_models[dataset_name] = attention_model
                        print(f"  Selected Attention DNN as teacher model for {dataset_name} dataset")
                    else:
                        self.teacher_models[dataset_name] = dnn_model
                        print(f"  Selected DNN as teacher model for {dataset_name} dataset")
                else:
                    # Regression task evaluation
                    dnn_preds = dnn_preds.flatten()
                    att_preds = att_preds.flatten()
                    
                    dnn_mse = mean_squared_error(y_test_np, dnn_preds)
                    att_mse = mean_squared_error(y_test_np, att_preds)
                    
                    dnn_mae = mean_absolute_error(y_test_np, dnn_preds)
                    att_mae = mean_absolute_error(y_test_np, att_preds)
                    
                    dnn_r2 = r2_score(y_test_np, dnn_preds)
                    att_r2 = r2_score(y_test_np, att_preds)
                    
                    print(f"  DNN MSE: {dnn_mse:.4f}, MAE: {dnn_mae:.4f}, R2: {dnn_r2:.4f}")
                    print(f"  Attention DNN MSE: {att_mse:.4f}, MAE: {att_mae:.4f}, R2: {att_r2:.4f}")
                    
                    # Select better performing model as teacher (using R2)
                    if att_r2 > dnn_r2:
                        self.teacher_models[dataset_name] = attention_model
                        print(f"  Selected Attention DNN as teacher model for {dataset_name} dataset")
                    else:
                        self.teacher_models[dataset_name] = dnn_model
                        print(f"  Selected DNN as teacher model for {dataset_name} dataset")
            
            # Save both models
            self.dl_models[dataset_name] = {
                'dnn': dnn_model,
                'attention_dnn': attention_model
            }
        
        print("\nDeep learning model training complete!")
        return self.teacher_models
    
    def train_student_models_with_distillation(self, epochs=50, batch_size=32, temperature=3.0, alpha=0.5):
        """Train smaller student models using knowledge distillation"""
        for dataset_name, teacher_model in self.teacher_models.items():
            print(f"\nPerforming knowledge distillation for {dataset_name} dataset...")
            
            X_train, X_test, y_train, y_test = self.data_splits[dataset_name]
            
            # Determine task type
            is_regression = False
            if dataset_name == 'cirrhosis':  # Cirrhosis dataset is a regression task
                is_regression = True
            
            # Prepare data - ensure all data is numeric type
            try:
                # Try direct conversion
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            except (TypeError, ValueError):
                # If fails, try converting to numeric type first
                X_train = X_train.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_test = X_test.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            
            y_train_tensor = torch.FloatTensor(y_train.values)
            y_test_tensor = torch.FloatTensor(y_test.values)
            
            # Reshape labels to column vector
            y_train_tensor = y_train_tensor.reshape(-1, 1)
            y_test_tensor = y_test_tensor.reshape(-1, 1)
            
            # Create student model - smaller network
            input_dim = X_train.shape[1]
            
            if not is_regression:
                # Classification task
                y_train_tensor = y_train_tensor.reshape(-1, 1)
                student_model = DNN(input_dim=input_dim, hidden_dims=[64, 32], output_dim=1).to(self.device)
                
                # Get soft labels from teacher model
                teacher_model.eval()
                with torch.no_grad():
                    soft_targets = teacher_model(X_train_tensor.to(self.device))
                
                # Create DataLoader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, soft_targets.cpu())
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Optimizer and loss functions
                optimizer = optim.Adam(student_model.parameters(), lr=0.001)
                hard_loss_fn = nn.BCELoss()
                soft_loss_fn = nn.MSELoss()  # For soft labels
                
                # Train student model
                for epoch in range(1, epochs + 1):
                    student_model.train()
                    epoch_loss = 0.0
                    
                    for batch_X, batch_y_hard, batch_y_soft in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y_hard = batch_y_hard.to(self.device)
                        batch_y_soft = batch_y_soft.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = student_model(batch_X)
                        
                        # Calculate hard label loss
                        hard_loss = hard_loss_fn(outputs, batch_y_hard)
                        
                        # Calculate soft label loss
                        soft_loss = soft_loss_fn(outputs, batch_y_soft)
                        
                        # Total loss
                        loss = alpha * hard_loss + (1 - alpha) * soft_loss
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    epoch_loss /= len(train_loader)
                    if (epoch % 10 == 0) or (epoch == 1):
                        print(f"  Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
                
                # Evaluate student model
                student_model.eval()
                with torch.no_grad():
                    outputs = student_model(X_test_tensor.to(self.device))
                    probs = outputs.cpu().numpy()
                    predictions = (probs > 0.5).astype(int).flatten()
                    
                    y_test_np = y_test.values
                    
                    accuracy = accuracy_score(y_test_np, predictions)
                    f1 = f1_score(y_test_np, predictions, average='weighted')
                    
                    print(f"  Student model performance - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                    
                    try:
                        auc = roc_auc_score(y_test_np, probs)
                        print(f"  AUC: {auc:.4f}")
                    except:
                        pass
            else:
                # Regression task
                y_train_tensor = y_train_tensor.reshape(-1, 1)
                student_model = DNN(input_dim=input_dim, hidden_dims=[64, 32], output_dim=1)
                student_model.is_regression = True  # Set as regression model
                student_model = student_model.to(self.device)
                
                # Get soft labels from teacher model
                teacher_model.eval()
                with torch.no_grad():
                    soft_targets = teacher_model(X_train_tensor.to(self.device))
                
                # Create DataLoader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, soft_targets.cpu())
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Optimizer and loss functions
                optimizer = optim.Adam(student_model.parameters(), lr=0.001)
                hard_loss_fn = nn.MSELoss()  # Use MSE for regression
                soft_loss_fn = nn.MSELoss()  # Also use MSE for soft labels
                
                # Train student model
                for epoch in range(1, epochs + 1):
                    student_model.train()
                    epoch_loss = 0.0
                    
                    for batch_X, batch_y_hard, batch_y_soft in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y_hard = batch_y_hard.to(self.device)
                        batch_y_soft = batch_y_soft.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = student_model(batch_X)
                        
                        # Calculate hard label loss
                        hard_loss = hard_loss_fn(outputs, batch_y_hard)
                        
                        # Calculate soft label loss
                        soft_loss = soft_loss_fn(outputs, batch_y_soft)
                        
                        # Total loss
                        loss = alpha * hard_loss + (1 - alpha) * soft_loss
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    epoch_loss /= len(train_loader)
                    if (epoch % 10 == 0) or (epoch == 1):
                        print(f"  Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
                
                # Evaluate student model - using regression metrics
                student_model.eval()
                with torch.no_grad():
                    outputs = student_model(X_test_tensor.to(self.device))
                    predictions = outputs.cpu().numpy().flatten()
                    
                    y_test_np = y_test.values
                    
                    mse = mean_squared_error(y_test_np, predictions)
                    mae = mean_absolute_error(y_test_np, predictions)
                    r2 = r2_score(y_test_np, predictions)
                    
                    print(f"  Student model performance - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            # Save student model
            self.student_models[dataset_name] = student_model
        
        return self.student_models
    
    def train_ensemble_models(self):
        """Train ensemble models, integrating predictions from individual models"""
        for dataset_name in self.data_splits.keys():
            print(f"\nTraining ensemble models for {dataset_name} dataset...")
            
            X_train, X_test, y_train, y_test = self.data_splits[dataset_name]
            
            # Determine task type
            is_regression = False
            if dataset_name == 'cirrhosis':  # Cirrhosis dataset is a regression task
                is_regression = True
            
            # Get baseline models for this dataset
            if dataset_name not in self.baseline_results:
                print(f"  No baseline models available for {dataset_name} dataset, skipping ensemble model training")
                continue
            
            if not is_regression:
                # Classification task - Voting and Stacking classifiers
                model_results = self.baseline_results[dataset_name]
                
                # Select top 3 best performing models
                top_models = sorted(model_results.keys(), 
                                key=lambda k: model_results[k]['test_f1'] if 'test_f1' in model_results[k] else 0, 
                                reverse=True)[:3]
                
                estimators = [(name, model_results[name]['model']) for name in top_models]
                
                print("  Training voting ensemble model...")
                voting_clf = VotingClassifier(estimators=estimators, voting='soft')
                voting_clf.fit(X_train, y_train)
                
                print("  Training stacking ensemble model...")
                meta_clf = LGBMClassifier()
                stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_clf)
                stacking_clf.fit(X_train, y_train)
                
                # Evaluate voting classifier
                voting_preds = voting_clf.predict(X_test)
                voting_acc = accuracy_score(y_test, voting_preds)
                voting_f1 = f1_score(y_test, voting_preds, average='weighted')
                print(f"  Voting ensemble model - Accuracy: {voting_acc:.4f}, F1: {voting_f1:.4f}")
                
                # Evaluate stacking classifier
                stacking_preds = stacking_clf.predict(X_test)
                stacking_acc = accuracy_score(y_test, stacking_preds)
                stacking_f1 = f1_score(y_test, stacking_preds, average='weighted')
                print(f"  Stacking ensemble model - Accuracy: {stacking_acc:.4f}, F1: {stacking_f1:.4f}")
                
                # Try to calculate AUC (for binary classification)
                try:
                    voting_probs = voting_clf.predict_proba(X_test)[:, 1]
                    stacking_probs = stacking_clf.predict_proba(X_test)[:, 1]
                    
                    voting_auc = roc_auc_score(y_test, voting_probs)
                    stacking_auc = roc_auc_score(y_test, stacking_probs)
                    
                    print(f"  Voting ensemble model - AUC: {voting_auc:.4f}")
                    print(f"  Stacking ensemble model - AUC: {stacking_auc:.4f}")
                except:
                    pass
                
                # Select best ensemble model
                if stacking_f1 > voting_f1:
                    best_ensemble = stacking_clf
                    best_type = "stacking"
                else:
                    best_ensemble = voting_clf
                    best_type = "voting"
                    
                self.ensemble_models[dataset_name] = best_ensemble
                print(f"  Best model for {dataset_name} dataset: {best_type} ensemble")
            else:
                # Regression task - Using VotingRegressor and StackingRegressor
                model_results = self.baseline_results[dataset_name]
                
                # Select top 3 best performing regression models
                top_models = sorted(model_results.keys(), 
                                key=lambda k: model_results[k]['test_r2'] if 'test_r2' in model_results[k] else 0, 
                                reverse=True)[:3]
                
                estimators = [(name, model_results[name]['model']) for name in top_models]
                
                print("  Training voting regression ensemble model...")
                voting_reg = VotingRegressor(estimators=estimators)
                voting_reg.fit(X_train, y_train)
                
                print("  Training stacking regression ensemble model...")
                meta_reg = LGBMRegressor()
                stacking_reg = StackingRegressor(estimators=estimators, final_estimator=meta_reg)
                stacking_reg.fit(X_train, y_train)
                
                # Evaluate voting regressor
                voting_preds = voting_reg.predict(X_test)
                voting_mse = mean_squared_error(y_test, voting_preds)
                voting_mae = mean_absolute_error(y_test, voting_preds)
                voting_r2 = r2_score(y_test, voting_preds)
                print(f"  Voting regression ensemble model - MSE: {voting_mse:.4f}, MAE: {voting_mae:.4f}, R2: {voting_r2:.4f}")
                
                # Evaluate stacking regressor
                stacking_preds = stacking_reg.predict(X_test)
                stacking_mse = mean_squared_error(y_test, stacking_preds)
                stacking_mae = mean_absolute_error(y_test, stacking_preds)
                stacking_r2 = r2_score(y_test, stacking_preds)
                print(f"  Stacking regression ensemble model - MSE: {stacking_mse:.4f}, MAE: {stacking_mae:.4f}, R2: {stacking_r2:.4f}")
                
                # Select best ensemble model (using R2)
                if stacking_r2 > voting_r2:
                    best_ensemble = stacking_reg
                    best_type = "stacking"
                else:
                    best_ensemble = voting_reg
                    best_type = "voting"
                    
                self.ensemble_models[dataset_name] = best_ensemble
                print(f"  Best model for {dataset_name} dataset: {best_type} ensemble")
        
        return self.ensemble_models
    
    def build_multi_disease_model(self):
        """Build multi-disease association model to predict probability of having multiple diseases simultaneously"""
        print("\nBuilding multi-disease association model...")
        
        # Merge datasets
        # For simplification, we assume the feature data for multi-disease prediction is already available here
        # In practical applications, this part needs feature engineering based on specific data
        
        # Here we use a virtual method to illustrate the approach for multi-disease prediction
        print("Multi-disease prediction requires modeling based on actual patient data")
        print("Since the datasets do not contain multi-disease information for the same patient, we can use the following methods:")
        print("1. Build relationship models between disease pairs (e.g., stroke-heart disease, stroke-cirrhosis, heart disease-cirrhosis)")
        print("2. Use existing diseases as features to predict the likelihood of other diseases")
        print("3. Calculate conditional probabilities to estimate multi-disease comorbidity probability")
        
        # Example: Association between stroke and heart disease (stroke dataset contains heart_disease feature)
        if 'heart_disease' in self.stroke_data.columns and 'stroke' in self.stroke_data.columns:
            # Analyze relationship between stroke and heart disease
            cross_table = pd.crosstab(
                self.stroke_data['heart_disease'], 
                self.stroke_data['stroke'],
                normalize='index'
            )
            
            print("\nContingency table for heart disease and stroke relationship (row normalized):")
            print(cross_table)
            
            # Calculate conditional probability: P(stroke|heart_disease)
            prob_stroke_given_heart = cross_table.loc[1, 1]
            print(f"Probability of stroke given heart disease: {prob_stroke_given_heart:.4f}")
            
            # Visualization
            plt.figure(figsize=(8, 6))
            cross_table.plot(kind='bar', stacked=True)
            plt.title('Relationship between Heart Disease and Stroke')
            plt.xlabel('Has Heart Disease')
            plt.ylabel('Proportion')
            plt.xticks([0, 1], ['No Heart Disease', 'Has Heart Disease'])
            plt.legend(['No Stroke', 'Has Stroke'])
            plt.tight_layout()
            plt.savefig('output/figures/heart_stroke_relationship.png')
            plt.close()
            
            # Methods for building multi-disease risk assessment model
            print("\nMethods for building multi-disease risk assessment model:")
            print("1. Bayesian Network: Establish probabilistic dependencies between diseases")
            print("2. Multi-label Classification: Predict multiple disease labels simultaneously")
            print("3. Cascade Prediction: First predict one disease, then use that prediction as a feature to predict other diseases")
            print("4. Shared Representation Learning: Learn shared feature representations for multiple disease prediction tasks")
            
            print("\nDue to data limitations, we can estimate multi-disease comorbidity probability as follows:")
            print("- P(Disease A and Disease B) = P(Disease A) * P(Disease B|Disease A)")
            print("- P(Disease A, B, and C) = P(Disease A) * P(Disease B|Disease A) * P(Disease C|Disease A, Disease B)")
            
        else:
            print("Insufficient information in datasets to analyze multi-disease relationships")
        
        return None
    
    def run_full_model_pipeline(self):
        """Run complete model training pipeline"""
        # 1. Train baseline models
        self.train_baseline_models()
        
        # 2. Train deep learning models
        self.train_deep_learning_models()
        
        # 3. Train student models (knowledge distillation)
        self.train_student_models_with_distillation()
        
        # 4. Train ensemble models
        self.train_ensemble_models()
        
        # 5. Create blended models
        # self.create_blended_models()
        
        return self.ensemble_models
        
def create_blended_models(self):
    """Create blended models combining predictions from machine learning and deep learning models"""
    print("\nCreating blended models...")
    
    for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
        print(f"\nCreating blended models for {dataset_name} dataset...")
        
        # Determine task type
        is_regression = False
        if dataset_name == 'cirrhosis':  # Cirrhosis dataset is a regression task
            is_regression = True
        
        # Check if ensemble model is available
        if dataset_name not in self.ensemble_models:
            print(f"  No ensemble model available for {dataset_name} dataset, skipping blended model creation")
            continue
            
        # Check if deep learning model is available
        if dataset_name not in self.teacher_models:
            print(f"  No deep learning model available for {dataset_name} dataset, skipping blended model creation")
            continue
        
        ensemble_model = self.ensemble_models[dataset_name]
        dl_model = self.teacher_models[dataset_name]
        
        if not is_regression:
            # Classification task - Blended prediction
            # Machine learning model prediction
            ml_probs = ensemble_model.predict_proba(X_test)[:, 1]
            
            # Deep learning model prediction
            dl_model.eval()
            with torch.no_grad():
                dl_probs = dl_model(torch.FloatTensor(X_test.values).to(self.device)).cpu().numpy().flatten()
            
            # Blended prediction (simple average)
            blend_probs = (ml_probs + dl_probs) / 2
            blend_preds = (blend_probs > 0.5).astype(int)
            
            # Evaluate blended model
            blend_accuracy = accuracy_score(y_test, blend_preds)
            blend_f1 = f1_score(y_test, blend_preds, average='weighted')
            blend_auc = roc_auc_score(y_test, blend_probs)
            
            print(f"  Blended model performance - Accuracy: {blend_accuracy:.4f}, F1: {blend_f1:.4f}, AUC: {blend_auc:.4f}")
            
            # Compare with individual models
            ml_preds = (ml_probs > 0.5).astype(int)
            dl_preds = (dl_probs > 0.5).astype(int)
            
            ml_accuracy = accuracy_score(y_test, ml_preds)
            ml_f1 = f1_score(y_test, ml_preds, average='weighted')
            ml_auc = roc_auc_score(y_test, ml_probs)
            
            dl_accuracy = accuracy_score(y_test, dl_preds)
            dl_f1 = f1_score(y_test, dl_preds, average='weighted')
            dl_auc = roc_auc_score(y_test, dl_probs)
            
            print(f"  Machine learning model - Accuracy: {ml_accuracy:.4f}, F1: {ml_f1:.4f}, AUC: {ml_auc:.4f}")
            print(f"  Deep learning model - Accuracy: {dl_accuracy:.4f}, F1: {dl_f1:.4f}, AUC: {dl_auc:.4f}")
            
            # Select best model (based on F1 score)
            if blend_f1 > ml_f1 and blend_f1 > dl_f1:
                best_name = "Blended Model"
            elif ml_f1 > dl_f1:
                best_name = "Machine Learning Model"
            else:
                best_name = "Deep Learning Model"
            print(f"  Best model for {dataset_name} dataset: {best_name}")
            
            # Save blended model info
            self.blended_models[dataset_name] = {
                'ml_model': ensemble_model,
                'dl_model': dl_model,
                'performance': {
                    'accuracy': blend_accuracy,
                    'f1': blend_f1,
                    'auc': blend_auc
                }
            }
        else:
            # Regression task - Blended prediction
            # Machine learning model prediction
            ml_preds = ensemble_model.predict(X_test)
            
            # Deep learning model prediction
            dl_model.eval()
            with torch.no_grad():
                dl_preds = dl_model(torch.FloatTensor(X_test.values).to(self.device)).cpu().numpy().flatten()
            
            # Blended prediction (simple average)
            blend_preds = (ml_preds + dl_preds) / 2
            
            # Evaluate blended model
            blend_mse = mean_squared_error(y_test, blend_preds)
            blend_mae = mean_absolute_error(y_test, blend_preds)
            blend_r2 = r2_score(y_test, blend_preds)
            
            print(f"  Blended model performance - MSE: {blend_mse:.4f}, MAE: {blend_mae:.4f}, R2: {blend_r2:.4f}")
            
            # Compare with individual models
            ml_mse = mean_squared_error(y_test, ml_preds)
            ml_mae = mean_absolute_error(y_test, ml_preds)
            ml_r2 = r2_score(y_test, ml_preds)
            
            dl_mse = mean_squared_error(y_test, dl_preds)
            dl_mae = mean_absolute_error(y_test, dl_preds)
            dl_r2 = r2_score(y_test, dl_preds)
            
            print(f"  Machine learning model - MSE: {ml_mse:.4f}, MAE: {ml_mae:.4f}, R2: {ml_r2:.4f}")
            print(f"  Deep learning model - MSE: {dl_mse:.4f}, MAE: {dl_mae:.4f}, R2: {dl_r2:.4f}")
            
            # Select best model (based on R2 score)
            if blend_r2 > ml_r2 and blend_r2 > dl_r2:
                best_name = "Blended Model"
            elif ml_r2 > dl_r2:
                best_name = "Machine Learning Model"
            else:
                best_name = "Deep Learning Model"
            print(f"  Best model for {dataset_name} dataset: {best_name}")
            
            # Save blended model info
            self.blended_models[dataset_name] = {
                'ml_model': ensemble_model,
                'dl_model': dl_model,
                'performance': {
                    'mse': blend_mse,
                    'mae': blend_mae,
                    'r2': blend_r2
                }
            }
    
    return self.blended_models

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_full_model_pipeline()
