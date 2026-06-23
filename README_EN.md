# Multi-Disease Risk Prediction and Analysis System

A disease prediction system based on three core models: Multi-Factor Weight Evaluation, Adaptive Weighted Ensemble Learning, and Bayesian Network Multi-Disease Association Analysis.

---

## Project Overview

This system contains three core algorithmic models addressing different problems in disease prediction and risk assessment:

### Core Models

| Model | File | Problem Description |
|-------|------|---------------------|
| **Multi-Factor Weight Evaluation** | `weight_evaluation.py` | Quantify the impact of various factors on disease risk, identify key risk factors |
| **Adaptive Weighted Ensemble Learning (AWELM)** | `awelm.py` | Build high-precision disease prediction models, adapt to different disease characteristics |
| **Bayesian Network Multi-Disease Association (BNMDAP)** | `bnmdap.py` | Quantify disease comorbidity associations, predict probability of multiple diseases occurring simultaneously |

---

## Project Structure

```
DpDa/
├── core/                              # Core models directory
│   ├── weight_evaluation.py            # Problem 1: Multi-factor weight evaluation
│   ├── awelm.py                        # Problem 2: Adaptive weighted ensemble learning
│   └── bnmdap.py                       # Problem 3: Bayesian network multi-disease association
│
├── web/                               # Web application directory
│   ├── app.py                         # Flask application main program
│   ├── model_utilities.py             # Model utilities
│   ├── model_calibration.py           # Probability calibration
│   └── multi_disease_model.py        # Multi-disease prediction
│
├── data/                              # Dataset directory
│   ├── stroke.csv                     # Stroke dataset
│   ├── heart.csv                      # Heart disease dataset
│   └── cirrhosis.csv                  # Cirrhosis dataset
│
├── templates/                         # HTML templates
│   ├── weight_evaluation.html         # Weight evaluation page
│   ├── awelm.html                     # AWELM model page
│   └── bnmdap.html                    # BNMDAP page
│
├── static/                            # Static resources
│   ├── style.css                      # Stylesheet
│   ├── script.js                      # JavaScript
│   └── images/                        # Image resources
│
├── output/                            # Output directory
│   ├── figures/                       # Generated charts
│   ├── models/                        # Trained models
│   └── processed_data/               # Processed data
│
├── visualization.py                   # Visualization module
├── requirements.txt                   # Dependencies
├── start.bat                          # Windows startup script
├── start.sh                           # Linux/Mac startup script
└── README.md                          # This file
```

---

## Quick Start

### Requirements
- Python 3.8+
- See `requirements.txt` for details

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pre-generate Model Checkpoints (optional, skipped automatically if already exists)

```bash
python generate_checkpoints.py
```

This pre-trains all models and caches results to `output/checkpoints/` (9 JSON files), avoiding repeated training on each startup.

### Run Web Application

```bash
start.bat
```

The browser will automatically open `http://localhost:5000`

---

## Core Models Details

### 1. Multi-Factor Weight Evaluation Model (Problem 1)

**File**: `weight_evaluation.py`

**Objective**: Quantify the contribution of each factor to disease risk, identify key risk factors

**Comprehensive Weight Formula**:
```
W_j = |β_j_full| × (1 - p_j) × (1 + ΔAUC_j)
```

Where:
- `|β_j_full|`: Absolute value of multivariate regression coefficient
- `p_j`: p-value (statistical significance)
- `ΔAUC_j`: AUC decrease after removing feature j (independent contribution to model performance)

**Usage**:
```python
from weight_evaluation import FeatureWeightEvaluator

evaluator = FeatureWeightEvaluator(dataset_name='heart')
summary = evaluator.run_full_pipeline(
    df, 
    target_col='HeartDisease',
    categorical_cols=['Sex', 'ChestPainType'],
    numeric_cols=['Age', 'RestingBP', 'Cholesterol']
)
```

### 2. Adaptive Weighted Ensemble Learning Model (AWELM)

**File**: `awelm.py`

**Objective**: Build high-precision disease prediction models by optimizing ensemble weights to fuse multiple base models

**Base Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM

**Loss Function** (with balance factor and regularization):
```
L(w) = -mean[β·y·log(p) + (1-y)·log(1-p)] + λ·H(w)
```
Where `p = Σw_i·p_i`, `β = negatives/positives`

**Usage**:
```python
from awelm import AdaptiveWeightedEnsemble

ensemble = AdaptiveWeightedEnsemble(dataset_name='stroke')
results = ensemble.run_full_pipeline(X, y)
```

### 3. Bayesian Network Multi-Disease Association Probability Model (BNMDAP)

**File**: `bnmdap.py`

**Objective**: Quantify disease comorbidity associations, predict probability of multiple diseases occurring simultaneously

**Core Formulas**:
- Conditional probability: `P(A|B) = P(A,B) / P(B)`
- Bayes' theorem: `P(B|A) = P(A|B)·P(B) / P(A)`
- Relative risk: `RR = P(Disease|RiskFactor) / P(Disease)`

**Usage**:
```python
from bnmdap import BayesianDiseaseNetwork

network = BayesianDiseaseNetwork()
network.estimate_prior_from_data(stroke_df, heart_df, cirrhosis_df)
result = network.predict(hypertension=1, heart_disease=0, cirrhosis=0)
```

---

## Web Application Features

The system provides an interactive web interface with the following pages:

### 1. Weight Evaluation Page (`/weight-evaluation`)
- View feature weight rankings
- Visualize cumulative weight curve
- Heatmap comparing univariate AUC with comprehensive weights

### 2. AWELM Model Page (`/awelm`)
- Base model performance comparison
- Ensemble model weight distribution
- ROC curve comparison

### 3. BNMDAP Page (`/bnmdap`)
- Bayesian network structure diagram
- Disease association heatmap
- Comorbidity probability prediction

---

## API Endpoints

### Weight Evaluation API
```
GET /api/weight-evaluation/<dataset>
```
Parameters: `dataset` = heart | stroke | cirrhosis

### AWELM API
```
GET/POST /api/awelm/<dataset>
```
Parameters: `dataset` = heart | stroke | cirrhosis

### BNMDAP Prediction API
```
POST /api/bnmdap/predict
Content-Type: application/json

{
    "hypertension": 0,
    "heart_disease": 0,
    "cirrhosis": 0
}
```

## Training Results

### 1. Weight Evaluation

| Dataset | Features | Full AUC | Description |
|---------|---------|----------|-------------|
| heart | 11 | 0.9178 | Heart disease risk factor weight analysis |
| stroke | 10 | 0.8443 | Stroke risk factor weight analysis |
| cirrhosis | 17 | 0.7493 | Cirrhosis staging (>=3) weight analysis |

### 2. AWELM Ensemble Learning

| Dataset | Best Model | Best AUC | Ensemble AUC | Imbalance Ratio |
|---------|-----------|---------|-------------|-----------------|
| heart | Gradient Boosting | 0.9348 | 0.9298 | 1.24 |
| stroke | Logistic Regression | 0.8377 | 0.6114 | 19.54 |
| cirrhosis | SVM | 0.6833 | 0.6549 | 2.52 |

Note: The stroke dataset has severe class imbalance (~5% positive cases), which causes SVM's Platt Scaling to degrade.

### 3. BNMDAP Bayesian Network

| Prior Probability | Estimated Value | Source |
|-------------------|-----------------|--------|
| P(Stroke) | 4.87% | Data estimation |
| P(Heart Disease) | 55.34% | Data estimation |
| P(Advanced Cirrhosis) | 71.53% | Data estimation |
| P(Hypertension) | 9.75% | Data estimation |

---

## Dataset Description

| Dataset | Samples | Target Variable | Characteristics |
|---------|---------|----------------|-----------------|
| stroke.csv | ~5,110 | Stroke (binary) | Class imbalance |
| heart.csv | ~918 | Heart disease (binary) | Complex feature combinations |
| cirrhosis.csv | ~418 | Cirrhosis stage (1-4) | Regression task |

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
flask>=2.0.0
joblib>=1.1.0
```

---

## Authors

- Shi Yunkun
- Zeng Zihang
