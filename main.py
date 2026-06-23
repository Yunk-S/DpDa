import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
from visualization import DataVisualizer
from model_training import ModelTrainer
from model_calibration import run_calibration_training
import time
import logging
import argparse
import subprocess

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Use app.log in root directory to avoid path issues
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directory structure"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/models/calibrators', exist_ok=True)
    os.makedirs('output/processed_data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    logger.info("Directory structure created")

def run_data_preprocessing():
    """Run data preprocessing pipeline"""
    logger.info("Starting data preprocessing...")
    start_time = time.time()
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()
    
    end_time = time.time()
    logger.info(f"Data preprocessing complete, time elapsed: {end_time - start_time:.2f} seconds")
    return processed_data

def run_data_visualization():
    """Run data visualization pipeline"""
    logger.info("Starting data visualization...")
    start_time = time.time()
    
    visualizer = DataVisualizer()
    visualizer.load_processed_data()
    visualizer.run_all_visualizations()
    
    end_time = time.time()
    logger.info(f"Data visualization complete, time elapsed: {end_time - start_time:.2f} seconds")

def run_generate_charts_and_metrics():
    """Run chart and metrics generation tool"""
    logger.info("Starting chart and evaluation metrics generation for each dataset...")
    start_time = time.time()
    
    # Ensure necessary directories exist
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    
    # Use unified chart generation module
    try:
        from chart_generator import run_chart_generation
        
        # Generate charts and metrics for heart disease dataset
        logger.info("Generating charts and evaluation metrics for heart dataset...")
        run_chart_generation('heart', ['all'])
        
        # Generate charts and metrics for cirrhosis dataset
        logger.info("Generating charts and evaluation metrics for cirrhosis dataset...")
        run_chart_generation('cirrhosis', ['all'])
        
        # Generate charts and metrics for stroke dataset
        logger.info("Generating charts and evaluation metrics for stroke dataset...")
        run_chart_generation('stroke', ['all'])
        
    except Exception as e:
        logger.error(f"Error generating charts and evaluation metrics: {e}")
    
    end_time = time.time()
    logger.info(f"Chart and evaluation metrics generation complete, time elapsed: {end_time - start_time:.2f} seconds")

def run_model_training():
    """Run model training pipeline"""
    logger.info("Starting model training...")
    start_time = time.time()
    
    # First get processed data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()  # Returns a dictionary, keys are dataset names, values are (features, target)
    
    # Split dataset, prepare for training
    data_splits = {}
    
    # Process each dataset separately
    for dataset_name, (features, target) in processed_data.items():
        # Split training and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        data_splits[dataset_name] = (X_train, X_test, y_train, y_test)
    
    # Initialize model trainer and pass in data
    trainer = ModelTrainer(data_splits)
    best_models = trainer.run_full_model_pipeline()
    
    end_time = time.time()
    logger.info(f"Model training complete, time elapsed: {end_time - start_time:.2f} seconds")
    return best_models

def run_model_calibration():
    """Run model calibration pipeline"""
    logger.info("Starting model calibration training...")
    start_time = time.time()
    
    # Call calibration training function from model_calibration module
    calibrators = run_calibration_training()
    
    end_time = time.time()
    logger.info(f"Model calibration complete, time elapsed: {end_time - start_time:.2f} seconds")
    return calibrators

def run_web_app():
    """Run web application"""
    logger.info("Starting web application...")
    try:
        from app import app
        app.run(debug=False, port=5000)
    except Exception as e:
        logger.error(f"Error starting web application: {e}")

def run_full_pipeline():
    """Run complete analysis and prediction pipeline"""
    logger.info("Starting complete analysis and prediction pipeline...")
    setup_directories()
    
    # Step 1: Data preprocessing
    processed_data = run_data_preprocessing()
    
    # Step 2: Data visualization
    run_data_visualization()
    
    # Step 3: Model training
    best_models = run_model_training()
    
    # Step 4: Model calibration
    calibrators = run_model_calibration()
    
    # Step 5: Generate charts and evaluation metrics
    run_generate_charts_and_metrics()
    
    # Step 6: Start web application
    run_web_app()
    
    # Print summary
    logger.info("Complete analysis and prediction pipeline finished!")
    logger.info(f"Best models: {list(best_models.keys())}")
    
    return processed_data, best_models, calibrators

def print_system_info():
    """Print system and dependency library information"""
    import sys
    import platform
    import sklearn
    import pandas as pd
    import numpy as np
    import matplotlib
    
    logger.info("System information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating system: {platform.system()} {platform.release()}")
    
    logger.info("Dependency library versions:")
    logger.info(f"scikit-learn: {sklearn.__version__}")
    logger.info(f"pandas: {pd.__version__}")
    logger.info(f"numpy: {np.__version__}")
    logger.info(f"matplotlib: {matplotlib.__version__}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disease Prediction and Big Data Analysis System')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing only')
    parser.add_argument('--visualize', action='store_true', help='Run data visualization only')
    parser.add_argument('--train', action='store_true', help='Run model training only')
    parser.add_argument('--calibrate', action='store_true', help='Run model calibration only')
    parser.add_argument('--generate', action='store_true', help='Run chart and metrics generation only')
    parser.add_argument('--webapp', action='store_true', help='Run web application only')
    parser.add_argument('--all', action='store_true', help='Run complete analysis and prediction pipeline')
    
    args = parser.parse_args()
    
    print_system_info()
    
    if args.preprocess:
        run_data_preprocessing()
    elif args.visualize:
        run_data_visualization()
    elif args.train:
        run_model_training()
    elif args.calibrate:
        run_model_calibration()
    elif args.generate:
        run_generate_charts_and_metrics()
    elif args.webapp:
        run_web_app()
    elif args.all:
        run_full_pipeline()
    else:
        # Run complete pipeline directly, no parameters needed
        run_full_pipeline()
