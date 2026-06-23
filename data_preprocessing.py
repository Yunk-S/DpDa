import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import missingno as msno
import os

class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessing class"""
        self.stroke_data = None
        self.heart_data = None
        self.cirrhosis_data = None
        
        # Store encoders for inverse transformation
        self.encoders = {
            'stroke': {},
            'heart': {},
            'cirrhosis': {}
        }
        
        # Store scalers
        self.scalers = {
            'stroke': None,
            'heart': None,
            'cirrhosis': None
        }
    
    def load_data(self, stroke_path='stroke.csv', heart_path='heart.csv', cirrhosis_path='cirrhosis.csv'):
        """Load three datasets"""
        self.stroke_data = pd.read_csv(stroke_path)
        self.heart_data = pd.read_csv(heart_path)
        self.cirrhosis_data = pd.read_csv(cirrhosis_path)
        print("Data loading complete!")
        
        # Create output directories
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/figures', exist_ok=True)
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/processed_data', exist_ok=True)
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def explore_data(self, save_figures=True):
        """Basic data exploration"""
        print("Starting data exploration...")
        
        # Load data first if not already loaded
        if self.stroke_data is None or self.heart_data is None or self.cirrhosis_data is None:
            self.load_data()
        
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        for dataset_name, df in datasets.items():
            print(f"\nExploring {dataset_name} dataset:")
            print(f"Data shape: {df.shape}")
            print(f"Data types:\n{df.dtypes}")
            
            # Basic statistics
            print(f"Statistical summary:\n{df.describe().T}")
            
            # Missing value analysis
            missing_count = df.isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_data = pd.DataFrame({'Missing Count': missing_count, 'Missing Rate': missing_percent})
            missing_data = missing_data[missing_data['Missing Count'] > 0]
            
            if not missing_data.empty:
                print(f"Missing value analysis:\n{missing_data}")
                
                if save_figures:
                    # Visualize missing values using matplotlib
                    plt.figure(figsize=(10, 6))
                    
                    # Replace non-numeric data with NaN
                    data_for_viz = df.copy()
                    data_for_viz = data_for_viz.replace(['--', 'N/A', 'NA'], np.nan)
                    
                    # Calculate missing percentage for each column
                    missing_percent = data_for_viz.isnull().mean() * 100
                    
                    # Plot missing value bar chart
                    missing_percent.sort_values(ascending=False).plot(kind='bar')
                    plt.title(f'{dataset_name} - Missing Value Percentage')
                    plt.ylabel('Missing Value Percentage (%)')
                    plt.xlabel('Features')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_missing_percent.png')
                    plt.close()
                    
                    # Try using missingno library matrix plot (without heatmap)
                    try:
                        plt.figure(figsize=(12, 6))
                        msno.matrix(data_for_viz)
                        plt.title(f'{dataset_name} - Missing Value Matrix', fontsize=16)
                        plt.tight_layout()
                        plt.savefig(f"output/figures/{dataset_name}_missing_matrix.png")
                        plt.close()
                    except Exception as e:
                        print(f"Cannot generate missing value matrix: {e}")
    
        print("Data exploration complete!")
        return datasets
    
    def handle_missing_values(self, method='iterative'):
        """Handle missing values
        
        Parameters:
            method: Imputation method, options 'mean', 'median', 'knn', 'iterative'
        """
        # Stroke dataset missing value handling
        print("Handling stroke dataset missing values...")
        # Check bmi column for missing values
        self.stroke_data['bmi'] = self.stroke_data['bmi'].replace('N/A', np.nan).astype(float)
        
        # Handle missing values based on different methods
        if method == 'mean':
            self.stroke_data['bmi'] = self.stroke_data['bmi'].fillna(self.stroke_data['bmi'].mean())
        elif method == 'median':
            self.stroke_data['bmi'] = self.stroke_data['bmi'].fillna(self.stroke_data['bmi'].median())
        elif method == 'knn':
            # Use KNN imputation for continuous variable missing values
            numeric_cols = self.stroke_data.select_dtypes(include=['float64', 'int64']).columns
            knn_imputer = KNNImputer(n_neighbors=5)
            self.stroke_data[numeric_cols] = pd.DataFrame(
                knn_imputer.fit_transform(self.stroke_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.stroke_data.index
            )
        elif method == 'iterative':
            # Use iterative imputation for continuous variable missing values
            numeric_cols = self.stroke_data.select_dtypes(include=['float64', 'int64']).columns
            iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            self.stroke_data[numeric_cols] = pd.DataFrame(
                iter_imputer.fit_transform(self.stroke_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.stroke_data.index
            )
            
        # Use mode to impute categorical variable missing values
        self.stroke_data['smoking_status'] = self.stroke_data['smoking_status'].fillna(
            self.stroke_data['smoking_status'].mode()[0]
        )
        
        # Heart disease dataset missing value handling
        print("Handling heart disease dataset missing values...")
        # Check and fill missing values
        if self.heart_data.isnull().sum().sum() > 0:
            if method in ['mean', 'median']:
                for col in self.heart_data.select_dtypes(include=['float64', 'int64']).columns:
                    if method == 'mean':
                        self.heart_data[col] = self.heart_data[col].fillna(self.heart_data[col].mean())
                    else:
                        self.heart_data[col] = self.heart_data[col].fillna(self.heart_data[col].median())
            elif method == 'knn':
                numeric_cols = self.heart_data.select_dtypes(include=['float64', 'int64']).columns
                knn_imputer = KNNImputer(n_neighbors=5)
                self.heart_data[numeric_cols] = pd.DataFrame(
                    knn_imputer.fit_transform(self.heart_data[numeric_cols]), 
                    columns=numeric_cols,
                    index=self.heart_data.index
                )
            elif method == 'iterative':
                numeric_cols = self.heart_data.select_dtypes(include=['float64', 'int64']).columns
                iter_imputer = IterativeImputer(max_iter=10, random_state=42)
                self.heart_data[numeric_cols] = pd.DataFrame(
                    iter_imputer.fit_transform(self.heart_data[numeric_cols]), 
                    columns=numeric_cols,
                    index=self.heart_data.index
                )
                
            # Use mode to impute categorical variables
            for col in self.heart_data.select_dtypes(include=['object']).columns:
                self.heart_data[col] = self.heart_data[col].fillna(self.heart_data[col].mode()[0])
        
        # Cirrhosis dataset missing value handling
        print("Handling cirrhosis dataset missing values...")
        # Convert 'NA' to np.nan
        self.cirrhosis_data = self.cirrhosis_data.replace('NA', np.nan)
        
        # Handle missing values for numeric features
        if method in ['mean', 'median']:
            for col in self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns:
                if self.cirrhosis_data[col].isnull().sum() > 0:
                    if method == 'mean':
                        self.cirrhosis_data[col] = self.cirrhosis_data[col].fillna(self.cirrhosis_data[col].mean())
                    else:
                        self.cirrhosis_data[col] = self.cirrhosis_data[col].fillna(self.cirrhosis_data[col].median())
        elif method == 'knn':
            numeric_cols = self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns
            knn_imputer = KNNImputer(n_neighbors=5)
            self.cirrhosis_data[numeric_cols] = pd.DataFrame(
                knn_imputer.fit_transform(self.cirrhosis_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.cirrhosis_data.index
            )
        elif method == 'iterative':
            numeric_cols = self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns
            iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            self.cirrhosis_data[numeric_cols] = pd.DataFrame(
                iter_imputer.fit_transform(self.cirrhosis_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.cirrhosis_data.index
            )
            
        # Use mode to impute categorical variables
        for col in self.cirrhosis_data.select_dtypes(include=['object']).columns:
            if self.cirrhosis_data[col].isnull().sum() > 0:
                self.cirrhosis_data[col] = self.cirrhosis_data[col].fillna(self.cirrhosis_data[col].mode()[0])
                
        print("Missing value handling for all datasets complete!")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def detect_outliers(self, method='iqr', visualize=True):
        """Detect and handle outliers
        
        Parameters:
            method: Outlier detection method, options 'iqr' or 'zscore'
            visualize: Whether to visualize outliers
        """
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        outlier_summary = {}
        
        for name, data in datasets.items():
            print(f"\n{name.upper()} Dataset Outlier Detection:")
            
            # Only detect outliers for numeric features
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            
            # Filter out ID columns, label columns, and date columns
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['id', 'stroke', 
                                                                               'heartdisease', 'n_days', 
                                                                               'status', 'stage']]
            
            outliers_dict = {}
            
            for col in numeric_cols:
                outliers_indices = []
                
                if method == 'iqr':
                    # IQR method
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Find outlier indices
                    outliers_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
                    
                elif method == 'zscore':
                    # Z-score method
                    mean = data[col].mean()
                    std = data[col].std()
                    z_scores = abs((data[col] - mean) / std)
                    outliers_indices = data[z_scores > 3].index
                
                # Save outlier information
                outliers_dict[col] = {
                    'count': len(outliers_indices),
                    'percentage': (len(outliers_indices) / len(data)) * 100,
                    'indices': outliers_indices.tolist()
                }
                
                print(f"Feature '{col}' has {len(outliers_indices)} outliers ({(len(outliers_indices)/len(data)*100):.2f}%)")
                
                # Visualization
                if visualize and len(outliers_indices) > 0:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot boxplot
                    plt.subplot(1, 2, 1)
                    sns.boxplot(y=data[col])
                    plt.title(f'{name.capitalize()} Dataset - {col} Boxplot')
                    
                    # Plot histogram
                    plt.subplot(1, 2, 2)
                    sns.histplot(data[col], kde=True)
                    plt.title(f'{name.capitalize()} Dataset - {col} Histogram')
                    
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{name}_{col}_outliers.png')
                    plt.close()
            
            outlier_summary[name] = outliers_dict
        
        return outlier_summary
    
    def handle_outliers(self, method='cap'):
        """Handle outliers
        
        Parameters:
            method: Handling method, options 'cap' (capping), 'remove' (removal), 'mean' (mean replacement)
        """
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        for name, data in datasets.items():
            print(f"\nHandling {name.upper()} Dataset Outliers:")
            
            # Only handle numeric features
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            
            # Filter out ID columns, label columns, and date columns
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['id', 'stroke', 
                                                                               'heartdisease', 'n_days', 
                                                                               'status', 'stage']]
            
            for col in numeric_cols:
                # Calculate upper and lower bounds
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Find outliers
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                
                if len(outliers) > 0:
                    print(f"Handling {len(outliers)} outliers for feature '{col}'")
                    
                    if method == 'cap':
                        # Capping method
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    elif method == 'remove':
                        # Remove outliers (only when outlier ratio is not high)
                        if len(outliers) / len(data) < 0.05:  # Less than 5% outliers
                            data.drop(outliers.index, inplace=True)
                            print(f"Removed {len(outliers)} outlier rows")
                        else:
                            print(f"Outlier ratio too high ({len(outliers)/len(data)*100:.2f}%), using capping method instead")
                            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    elif method == 'mean':
                        # Mean replacement
                        mean_value = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)][col].mean()
                        data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = mean_value
            
            # Update datasets
            if name == 'stroke':
                self.stroke_data = data
            elif name == 'heart':
                self.heart_data = data
            else:
                self.cirrhosis_data = data
        
        print("Outlier handling complete!")
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def encode_categorical_features(self):
        """Encode categorical features"""
        # Handle stroke dataset
        print("\nEncoding stroke dataset categorical features...")
        
        # Binary encoding
        binary_cols = ['gender', 'ever_married', 'Residence_type']
        for col in binary_cols:
            le = LabelEncoder()
            self.stroke_data[col] = le.fit_transform(self.stroke_data[col])
            self.encoders['stroke'][col] = le
        
        # One-hot encoding
        onehot_cols = ['work_type', 'smoking_status']
        for col in onehot_cols:
            dummies = pd.get_dummies(self.stroke_data[col], prefix=col, drop_first=False)
            self.stroke_data = pd.concat([self.stroke_data, dummies], axis=1)
            self.stroke_data.drop(col, axis=1, inplace=True)
        
        # Handle heart disease dataset
        print("Encoding heart disease dataset categorical features...")
        # Binary encoding
        binary_cols = ['Sex', 'ExerciseAngina']
        for col in binary_cols:
            le = LabelEncoder()
            self.heart_data[col] = le.fit_transform(self.heart_data[col])
            self.encoders['heart'][col] = le
            
        # One-hot encoding
        onehot_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
        for col in onehot_cols:
            dummies = pd.get_dummies(self.heart_data[col], prefix=col, drop_first=False)
            self.heart_data = pd.concat([self.heart_data, dummies], axis=1)
            self.heart_data.drop(col, axis=1, inplace=True)
        
        # Handle cirrhosis dataset
        print("Encoding cirrhosis dataset categorical features...")
        # Binary encoding
        binary_cols = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders']
        for col in binary_cols:
            le = LabelEncoder()
            self.cirrhosis_data[col] = le.fit_transform(self.cirrhosis_data[col])
            self.encoders['cirrhosis'][col] = le
            
        # One-hot encoding
        onehot_cols = ['Drug', 'Status', 'Edema']
        for col in onehot_cols:
            dummies = pd.get_dummies(self.cirrhosis_data[col], prefix=col, drop_first=False)
            self.cirrhosis_data = pd.concat([self.cirrhosis_data, dummies], axis=1)
            self.cirrhosis_data.drop(col, axis=1, inplace=True)
            
        print("Categorical feature encoding complete!")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def scale_features(self):
        """Standardize features"""
        # Stroke dataset
        print("\nStandardizing stroke dataset features...")
        stroke_numeric = self.stroke_data.select_dtypes(include=['float64', 'int64']).columns
        stroke_numeric = [col for col in stroke_numeric if col not in ['id', 'stroke']]
        
        scaler = StandardScaler()
        self.stroke_data[stroke_numeric] = scaler.fit_transform(self.stroke_data[stroke_numeric])
        self.scalers['stroke'] = scaler
        
        # Heart disease dataset
        print("Standardizing heart disease dataset features...")
        heart_numeric = self.heart_data.select_dtypes(include=['float64', 'int64']).columns
        heart_numeric = [col for col in heart_numeric if col != 'HeartDisease']
        
        scaler = StandardScaler()
        self.heart_data[heart_numeric] = scaler.fit_transform(self.heart_data[heart_numeric])
        self.scalers['heart'] = scaler
        
        # Cirrhosis dataset
        print("Standardizing cirrhosis dataset features...")
        cirrhosis_numeric = self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns
        cirrhosis_numeric = [col for col in cirrhosis_numeric if col not in ['ID', 'N_Days', 'Stage']]
        
        scaler = StandardScaler()
        self.cirrhosis_data[cirrhosis_numeric] = scaler.fit_transform(self.cirrhosis_data[cirrhosis_numeric])
        self.scalers['cirrhosis'] = scaler
        
        print("Feature standardization complete!")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def feature_engineering(self):
        """Feature engineering, create new features"""
        # Stroke dataset feature engineering
        print("\nPerforming feature engineering on stroke dataset...")
        
        # Age grouping
        self.stroke_data['age_group'] = pd.cut(
            self.stroke_data['age'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # BMI classification
        def bmi_category(bmi):
            if bmi < 18.5:
                return 0  # Underweight
            elif bmi < 24:
                return 1  # Normal
            elif bmi < 28:
                return 2  # Overweight
            else:
                return 3  # Obese
                
        self.stroke_data['bmi_category'] = self.stroke_data['bmi'].apply(bmi_category)
        
        # High blood glucose risk
        self.stroke_data['glucose_risk'] = (self.stroke_data['avg_glucose_level'] > 140).astype(int)
        
        # Multiple risk factors (hypertension, heart disease, high blood glucose)
        self.stroke_data['multiple_risks'] = (
            self.stroke_data['hypertension'] + 
            self.stroke_data['heart_disease'] + 
            self.stroke_data['glucose_risk']
        )
        
        # Heart disease dataset feature engineering
        print("Performing feature engineering on heart disease dataset...")
        
        # Age grouping
        self.heart_data['age_group'] = pd.cut(
            self.heart_data['Age'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # Cholesterol classification
        def chol_category(chol):
            if chol < 200:
                return 0  # Normal
            elif chol < 240:
                return 1  # Borderline high
            else:
                return 2  # High
                
        self.heart_data['chol_category'] = self.heart_data['Cholesterol'].apply(chol_category)
        
        # Hypertension risk
        self.heart_data['bp_risk'] = (self.heart_data['RestingBP'] > 140).astype(int)
        
        # Heart rate warning (low max heart rate)
        self.heart_data['hr_warning'] = (self.heart_data['MaxHR'] < 100).astype(int)
        
        # Cirrhosis dataset feature engineering
        print("Performing feature engineering on cirrhosis dataset...")
        
        # Convert age to years (original data is in days)
        self.cirrhosis_data['Age_years'] = self.cirrhosis_data['Age'] / 365.25
        
        # Age grouping
        self.cirrhosis_data['age_group'] = pd.cut(
            self.cirrhosis_data['Age_years'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # Bilirubin risk
        self.cirrhosis_data['bilirubin_risk'] = (self.cirrhosis_data['Bilirubin'] > 1.2).astype(int)
        
        # Cholesterol risk
        self.cirrhosis_data['cholesterol_risk'] = (self.cirrhosis_data['Cholesterol'] > 240).astype(int)
        
        # Liver function composite score (based on key indicators)
        self.cirrhosis_data['liver_score'] = (
            (self.cirrhosis_data['Bilirubin'] > 1.2).astype(int) +
            (self.cirrhosis_data['Albumin'] < 3.5).astype(int) +
            (self.cirrhosis_data['Prothrombin'] > 12).astype(int)
        )
        
        print("Feature engineering complete!")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def prepare_train_test_data(self):
        """Prepare training and test data"""
        # Stroke dataset
        stroke_features = self.stroke_data.drop(['id', 'stroke'], axis=1, errors='ignore')
        stroke_target = self.stroke_data['stroke']
        
        # Heart disease dataset
        heart_features = self.heart_data.drop(['HeartDisease'], axis=1, errors='ignore')
        heart_target = self.heart_data['HeartDisease']
        
        # Cirrhosis dataset (using Stage as target variable)
        cirrhosis_features = self.cirrhosis_data.drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
        cirrhosis_target = self.cirrhosis_data['Stage']
        
        # Save processed data
        self.stroke_data.to_csv('output/processed_data/stroke_processed.csv', index=False)
        self.heart_data.to_csv('output/processed_data/heart_processed.csv', index=False)
        self.cirrhosis_data.to_csv('output/processed_data/cirrhosis_processed.csv', index=False)
        
        print("Training and test data preparation complete!")
        
        return {
            'stroke': (stroke_features, stroke_target),
            'heart': (heart_features, heart_target),
            'cirrhosis': (cirrhosis_features, cirrhosis_target)
        }
        
    def run_full_preprocessing(self):
        """Run complete preprocessing pipeline"""
        self.load_data()
        self.explore_data()
        self.handle_missing_values(method='iterative')
        self.detect_outliers()
        self.handle_outliers(method='cap')
        self.encode_categorical_features()
        self.feature_engineering()
        self.scale_features()
        return self.prepare_train_test_data()

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()
    print("Preprocessing complete, data saved to output/processed_data/ directory")
