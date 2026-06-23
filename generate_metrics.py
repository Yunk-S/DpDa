import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_models_and_data(dataset_name=None):
    """Load models and test data
    
    Args:
        dataset_name (str, optional): Specify dataset name to load, e.g. 'stroke', 'heart', 'cirrhosis'.
                                     If None, load all datasets.
    """
    models = {}
    model_dir = 'output/models'
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                # If dataset is specified, only load models for that dataset
                if dataset_name and not filename.startswith(dataset_name):
                    continue
                    
                try:
                    model_path = os.path.join(model_dir, filename)
                    model_name = filename.replace('.pkl', '')
                    models[model_name] = joblib.load(model_path)
                    print(f"Loaded model: {model_name}")
                except Exception as e:
                    print(f"Failed to load model {model_name}: {e}")
    
    # Load processed data
    data = {}
    data_dir = 'output/processed_data'
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_name = filename.replace('_processed.csv', '')
                
                # If dataset is specified, only load that dataset
                if dataset_name and data_name != dataset_name:
                    continue
                    
                data_path = os.path.join(data_dir, filename)
                try:
                    data[data_name] = pd.read_csv(data_path)
                    print(f"Loaded data: {data_name}")
                except Exception as e:
                    print(f"Failed to load data {data_name}: {e}")
    
    return models, data

def prepare_test_data(data):
    """Prepare test data"""
    test_datasets = {}
    
    for name, df in data.items():
        if name == 'stroke':
            features = df.drop(['id', 'stroke'], axis=1, errors='ignore')
            target = df['stroke']
        elif name == 'heart':
            features = df.drop(['HeartDisease'], axis=1, errors='ignore')
            target = df['HeartDisease']
        elif name == 'cirrhosis':
            features = df.drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
            target = df['Stage']
        
        # Only keep numeric features
        features = features.select_dtypes(include=['float64', 'int64']).copy()
        
        # Split test set (for simplicity, use 20% of entire dataset as test set)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        test_datasets[name] = (X_test, y_test)
    
    return test_datasets

def generate_metrics_json(models, test_datasets):
    """Generate model evaluation metrics JSON file"""
    print("\nGenerating model evaluation metrics...")
    
    for dataset_name, (X_test, y_test) in test_datasets.items():
        # Find all models applicable to this dataset
        dataset_models = [m for m in models.keys() if m.startswith(dataset_name)]
        
        for model_name in dataset_models:
            model = models[model_name]
            metrics = {}
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                
                # Classification tasks
                if dataset_name in ['stroke', 'heart']:
                    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
                    
                    try:
                        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
                        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
                        metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
                    except:
                        metrics['precision'] = "N/A"
                        metrics['recall'] = "N/A"
                        metrics['f1'] = "N/A"
                    
                    # AUC (only applicable for binary classification)
                    if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                        try:
                            y_prob = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            metrics['auc'] = float(auc(fpr, tpr))
                        except:
                            metrics['auc'] = "N/A"
                    else:
                        metrics['auc'] = "N/A"
                    
                # Regression tasks
                else:
                    metrics['mse'] = float(mean_squared_error(y_test, y_pred))
                    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    metrics['r2'] = float(r2_score(y_test, y_pred))
                    
                # Save metrics to JSON file
                with open(f'output/models/{model_name}_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                print(f"Evaluation metrics generated for {model_name}")
                
            except Exception as e:
                print(f"Error generating evaluation metrics for {model_name}: {e}")

def main():
    """Main function"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate model evaluation metrics')
    parser.add_argument('--dataset', type=str, choices=['stroke', 'heart', 'cirrhosis'], 
                        help='Specify dataset to process, if not specified process all datasets')
    args = parser.parse_args()
    
    print(f"Starting to generate evaluation metrics for {'all' if args.dataset is None else args.dataset} models...")
    
    # Load models and data
    models, data = load_models_and_data(dataset_name=args.dataset)
    
    if not models:
        print(f"No {'any' if args.dataset is None else args.dataset} models found!")
        return
        
    if not data:
        print(f"No {'any' if args.dataset is None else args.dataset} data found!")
        return
    
    # Prepare test data
    test_datasets = prepare_test_data(data)
    
    # Generate model evaluation metrics JSON
    generate_metrics_json(models, test_datasets)
    
    print(f"\nEvaluation metrics generation for {'all' if args.dataset is None else args.dataset} models complete!")

if __name__ == "__main__":
    main()
