import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Use try-except to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly package not available, interactive visualization features will be skipped")
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP package not available, SHAP value visualization will be skipped")
    SHAP_AVAILABLE = False

class DataVisualizer:
    def __init__(self, stroke_data=None, heart_data=None, cirrhosis_data=None):
        """Initialize data visualization class
        
        Parameters:
            stroke_data: Stroke dataset
            heart_data: Heart disease dataset
            cirrhosis_data: Cirrhosis dataset
        """
        self.stroke_data = stroke_data
        self.heart_data = heart_data
        self.cirrhosis_data = cirrhosis_data
        
        # Create output directory
        os.makedirs('output/figures', exist_ok=True)
        
        # Set visualization style
        sns.set(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # For displaying labels
        plt.rcParams['axes.unicode_minus'] = False  # For displaying minus signs
    
    def load_processed_data(self, stroke_path='output/processed_data/stroke_processed.csv',
                           heart_path='output/processed_data/heart_processed.csv',
                           cirrhosis_path='output/processed_data/cirrhosis_processed.csv'):
        """Load processed data"""
        self.stroke_data = pd.read_csv(stroke_path)
        self.heart_data = pd.read_csv(heart_path)
        self.cirrhosis_data = pd.read_csv(cirrhosis_path)
        print("Processed data loaded successfully!")
        
    def plot_feature_distributions(self, save_plots=True):
        """Plot feature distributions"""
        datasets = {
            'stroke': (self.stroke_data, 'stroke'),
            'heart': (self.heart_data, 'HeartDisease'),
            'cirrhosis': (self.cirrhosis_data, 'Stage')
        }
        
        for name, (data, target) in datasets.items():
            print(f"Plotting {name} dataset feature distributions...")
            
            # Categorical feature visualization
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            categorical_cols = [col for col in categorical_cols if col.lower() not in ['id']]
            
            if len(categorical_cols) > 0:
                # Calculate number of charts needed for each categorical feature
                n_cat_cols = len(categorical_cols)
                n_rows = (n_cat_cols + 2) // 3  # Maximum 3 charts per row
                
                # Create canvas
                plt.figure(figsize=(18, n_rows * 5))
                
                # Plot count chart for each categorical feature
                for i, col in enumerate(categorical_cols):
                    plt.subplot(n_rows, 3, i+1)
                    
                    # Check if target variable is in data
                    if target in data.columns:
                        # Count chart grouped by target variable
                        sns.countplot(x=col, hue=target, data=data)
                        plt.title(f'{name.capitalize()} - {col} Distribution (grouped by {target})')
                    else:
                        # Regular count chart
                        sns.countplot(x=col, data=data)
                        plt.title(f'{name.capitalize()} - {col} Distribution')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_categorical_distributions.png')
                plt.close()
            
            # Numeric feature visualization
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['id'] and col != target]
            
            if len(numeric_cols) > 0:
                # Create canvas
                n_num_cols = len(numeric_cols)
                n_rows = (n_num_cols + 2) // 3  # Maximum 3 charts per row
                
                plt.figure(figsize=(18, n_rows * 5))
                
                # Plot distribution for each numeric feature
                for i, col in enumerate(numeric_cols):
                    plt.subplot(n_rows, 3, i+1)
                    
                    # Plot histogram and density plot
                    if target in data.columns and len(data[target].unique()) <= 5:
                        # Distribution plot grouped by target variable
                        for cls in sorted(data[target].unique()):
                            sns.histplot(data[data[target]==cls][col], kde=True, 
                                         label=f'{target}={cls}', alpha=0.5)
                        plt.legend()
                        plt.title(f'{name.capitalize()} - {col} Distribution (grouped by {target})')
                    else:
                        # Regular distribution plot
                        sns.histplot(data[col], kde=True)
                        plt.title(f'{name.capitalize()} - {col} Distribution')
                    
                    plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_numeric_distributions.png')
                plt.close()
            
            # Plot target variable distribution
            if target in data.columns:
                plt.figure(figsize=(8, 6))
                target_counts = data[target].value_counts()
                
                # Calculate percentages
                target_percents = target_counts / target_counts.sum() * 100
                
                # Plot pie chart
                plt.pie(target_counts, labels=[f'{i} ({p:.1f}%)' for i, p in zip(target_counts.index, target_percents)],
                        autopct='%1.1f%%', startangle=90, shadow=True)
                plt.title(f'{name.capitalize()} - {target} Distribution')
                plt.axis('equal')
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_target_distribution.png')
                plt.close()
                
        print("Feature distribution visualization complete!")
    
    def plot_correlation_matrices(self, save_plots=True):
        """Plot feature correlation matrices"""
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        for name, data in datasets.items():
            print(f"Plotting {name} dataset correlation matrix...")
            
            # Select numeric features
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            
            # If too many features, keep only important ones
            if numeric_data.shape[1] > 15:
                # Select important features based on domain knowledge
                if name == 'stroke':
                    important_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 
                                     'bmi', 'stroke', 'multiple_risks', 'glucose_risk']
                    numeric_data = numeric_data[important_cols]
                elif name == 'heart':
                    important_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 
                                     'MaxHR', 'Oldpeak', 'HeartDisease', 'bp_risk']
                    numeric_data = numeric_data[important_cols]
                elif name == 'cirrhosis':
                    important_cols = ['Age_years', 'Bilirubin', 'Albumin', 'Copper', 
                                     'Prothrombin', 'Stage', 'liver_score', 'bilirubin_risk']
                    numeric_data = numeric_data[important_cols]
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                        linewidths=0.5, vmin=-1, vmax=1)
            plt.title(f'{name.capitalize()} Dataset - Feature Correlation Matrix')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'output/figures/{name}_correlation_matrix.png')
            plt.close()
            
            # If Plotly is available, create interactive heatmap
            if PLOTLY_AVAILABLE and save_plots:
                try:
                    fig = px.imshow(corr_matrix,
                                 x=corr_matrix.columns,
                                 y=corr_matrix.columns,
                                 color_continuous_scale='RdBu_r',
                                 title=f'{name.capitalize()} Dataset - Feature Correlation Matrix (Interactive)')
                    
                    fig.write_html(f'output/figures/{name}_correlation_matrix_interactive.html')
                except Exception as e:
                    print(f"Error creating interactive heatmap: {e}")
        
        print("Correlation matrix visualization complete!")
    
    def plot_feature_importance(self, save_plots=True):
        """Plot feature importance"""
        datasets = {
            'stroke': (self.stroke_data, 'stroke'),
            'heart': (self.heart_data, 'HeartDisease'),
            'cirrhosis': (self.cirrhosis_data, 'Stage')
        }
        
        for name, (data, target) in datasets.items():
            print(f"Calculating {name} dataset feature importance...")
            
            if target in data.columns:
                # Prepare data
                X = data.drop([target], axis=1)
                X = X.select_dtypes(include=['float64', 'int64'])  # Keep only numeric features
                
                # Remove ID columns
                id_cols = [col for col in X.columns if col.lower() in ['id', 'n_days']]
                X = X.drop(id_cols, axis=1, errors='ignore')
                
                if len(X.columns) == 0:
                    print(f"{name} dataset has no suitable features for importance calculation")
                    continue
                    
                y = data[target]
                
                # Determine task type based on dataset name and target column
                is_classification = True
                if name == 'cirrhosis' and target == 'Stage':
                    is_classification = False
                elif len(y.unique()) > 10:  # If target variable has many unique values, likely a regression task
                    is_classification = False
                
                # Handle categorical target variables
                if is_classification and (y.dtype == 'object' or y.dtype == 'category'):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Train model
                if is_classification:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Fit model
                model.fit(X, y)
                
                # Get feature importance
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot feature importance bar chart
                plt.figure(figsize=(12, 8))
                plt.title(f'{name.capitalize()} Dataset - Feature Importance')
                plt.bar(range(X.shape[1]), importances[indices], align='center')
                plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_feature_importance.png')
                plt.close()
                
                # Use SHAP values to explain model
                if SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X)
                        
                        # Plot summary chart
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                        plt.title(f'{name.capitalize()} Dataset - SHAP Value Feature Importance')
                        plt.tight_layout()
                        
                        if save_plots:
                            plt.savefig(f'output/figures/{name}_shap_importance.png')
                        plt.close()
                        
                        # Plot detailed SHAP values chart
                        plt.figure(figsize=(12, 10))
                        shap.summary_plot(shap_values, X, show=False)
                        plt.title(f'{name.capitalize()} Dataset - SHAP Value Feature Impact')
                        plt.tight_layout()
                        
                        if save_plots:
                            plt.savefig(f'output/figures/{name}_shap_summary.png')
                        plt.close()
                        
                    except Exception as e:
                        print(f"SHAP value calculation failed: {e}")
                else:
                    print("SHAP not available, skipping SHAP value visualization")
            
            else:
                print(f"Target variable {target} not found in {name} dataset")
        
        print("Feature importance visualization complete!")
    
    def plot_pair_plots(self, save_plots=True):
        """Plot pair plots to analyze relationships between features"""
        datasets = {
            'stroke': (self.stroke_data, 'stroke'),
            'heart': (self.heart_data, 'HeartDisease'),
            'cirrhosis': (self.cirrhosis_data, 'Stage')
        }
        
        for name, (data, target) in datasets.items():
            print(f"Plotting {name} dataset pair plots...")
            
            if target in data.columns:
                # Select most important features
                if name == 'stroke':
                    selected_features = ['age', 'avg_glucose_level', 'bmi', 'stroke']
                elif name == 'heart':
                    selected_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'HeartDisease']
                elif name == 'cirrhosis':
                    selected_features = ['Age_years', 'Bilirubin', 'Albumin', 'Prothrombin', 'Stage']
                
                # Filter data
                plot_data = data[selected_features].copy()
                
                # Plot pair plot
                plt.figure(figsize=(12, 10))
                sns.pairplot(plot_data, hue=target, diag_kind='kde')
                plt.suptitle(f'{name.capitalize()} Dataset - Pair Plot', y=1.02)
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_pair_plot.png')
                plt.close()
            
            else:
                print(f"Target variable {target} not found in {name} dataset")
        
        print("Pair plot visualization complete!")
    
    def plot_disease_comparison(self, save_plots=True):
        """Compare common features across three diseases"""
        print("Comparing common features across three diseases...")
        
        if self.stroke_data is None or self.heart_data is None or self.cirrhosis_data is None:
            print("Warning: Data not loaded, please call load_processed_data method first")
            return
        
        # Extract common features
        # Age is a common feature across all three datasets
        stroke_age = self.stroke_data[['age', 'stroke']].copy()
        stroke_age['disease'] = 'Stroke'
        stroke_age.rename(columns={'stroke': 'has_disease', 'age': 'Age'}, inplace=True)
        
        heart_age = self.heart_data[['Age', 'HeartDisease']].copy()
        heart_age['disease'] = 'Heart Disease'
        heart_age.rename(columns={'HeartDisease': 'has_disease'}, inplace=True)
        
        cirrhosis_age = self.cirrhosis_data[['Age_years', 'Stage']].copy()
        cirrhosis_age['disease'] = 'Cirrhosis'
        cirrhosis_age['has_disease'] = (cirrhosis_age['Stage'] > 2).astype(int)  # Treat stages 3,4 as severe disease
        cirrhosis_age.rename(columns={'Age_years': 'Age'}, inplace=True)
        cirrhosis_age.drop('Stage', axis=1, inplace=True)
        
        # Combine data
        combined_age = pd.concat([stroke_age, heart_age, cirrhosis_age], ignore_index=True)
        
        # Plot age distribution comparison
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='disease', y='Age', hue='has_disease', 
                      data=combined_age, split=True, inner="quart")
        plt.title('Age Distribution Comparison Across Three Diseases')
        plt.xlabel('Disease Type')
        plt.ylabel('Age')
        plt.legend(title='Has Disease', loc='best')
        
        if save_plots:
            plt.savefig('output/figures/disease_age_comparison.png')
        plt.close()
        
        # Plot disease prevalence by age group
        plt.figure(figsize=(14, 8))
        
        # Load raw data to get mean and standard deviation of age
        try:
            # Load raw data
            stroke_orig = pd.read_csv('stroke.csv')
            heart_orig = pd.read_csv('heart.csv')
            cirrhosis_orig = pd.read_csv('cirrhosis.csv')
            
            # Calculate statistics for raw data
            stroke_mean = stroke_orig['age'].mean()
            stroke_std = stroke_orig['age'].std()
            
            heart_mean = heart_orig['Age'].mean()
            heart_std = heart_orig['Age'].std()
            
            cirrhosis_age_years = cirrhosis_orig['Age'] / 365.25  # Convert to years
            cirrhosis_mean = cirrhosis_age_years.mean()
            cirrhosis_std = cirrhosis_age_years.std()
            
            print(f"Raw data age statistics:")
            print(f"Stroke data: Mean = {stroke_mean:.2f}, Std = {stroke_std:.2f}")
            print(f"Heart disease data: Mean = {heart_mean:.2f}, Std = {heart_std:.2f}")
            print(f"Cirrhosis data: Mean = {cirrhosis_mean:.2f}, Std = {cirrhosis_std:.2f}")
            
            # Reverse standardization - apply different mean and std based on disease type
            combined_age['Age_Original'] = combined_age.apply(
                lambda row: row['Age'] * stroke_std + stroke_mean if row['disease'] == 'Stroke' else
                           (row['Age'] * heart_std + heart_mean if row['disease'] == 'Heart Disease' else
                            row['Age'] * cirrhosis_std + cirrhosis_mean),
                axis=1
            )
            
            # Print age distribution before and after processing
            print("\nAge distribution comparison:")
            print("Standardized age range:", combined_age['Age'].min(), "to", combined_age['Age'].max())
            print("Unstandardized age range:", combined_age['Age_Original'].min(), "to", combined_age['Age_Original'].max())
            
        except Exception as e:
            print(f"Unable to load raw data for unstandardization: {e}")
            # If raw data cannot be loaded, use estimated mean and std
            combined_age['Age_Original'] = combined_age['Age'] * 20 + 50  # Rough estimate
        
        # Define age groups - using unstandardized age
        age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
        
        # Group using unstandardized age values
        combined_age['age_group'] = pd.cut(combined_age['Age_Original'], bins=age_bins, labels=age_labels)
        
        # Check age group distribution
        print("\nUnstandardized age group distribution:")
        print(combined_age['age_group'].value_counts().sort_index())
        
        # Calculate disease prevalence for each age group
        disease_rates = combined_age.groupby(['disease', 'age_group'])['has_disease'].mean().reset_index()
        disease_rates['percentage'] = disease_rates['has_disease'] * 100
        
        # Print results
        print("\nDisease prevalence by age group:")
        print(disease_rates)
        
        # Plot grouped bar chart
        plt.figure(figsize=(16, 8))
        sns.barplot(x='age_group', y='percentage', hue='disease', data=disease_rates)
        plt.title('Disease Prevalence Comparison by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Prevalence (%)')
        plt.legend(title='Disease Type')
        plt.xticks(rotation=45)
        
        if save_plots:
            plt.savefig('output/figures/disease_rate_by_age.png')
        plt.close()
        
        # Use Plotly to create interactive charts (if available)
        if PLOTLY_AVAILABLE and save_plots:
            try:
                fig = px.line(disease_rates, x='age_group', y='percentage', color='disease',
                            title='Disease Prevalence by Age Group (Interactive)',
                            labels={'age_group': 'Age Group', 'percentage': 'Prevalence (%)', 'disease': 'Disease Type'},
                            markers=True)
                
                fig.write_html('output/figures/disease_rate_by_age_interactive.html')
            except Exception as e:
                print(f"Error creating interactive age-disease chart: {e}")
            
        # Gender comparison (both stroke and heart disease datasets have gender features)
        # First check if gender features exist in datasets
        if 'gender' in self.stroke_data.columns and 'Sex' in self.heart_data.columns:
            print("Processing gender data...")
            
            try:
                # Extract gender and disease status from stroke data
                stroke_gender = self.stroke_data[['gender', 'stroke']].copy()
                stroke_gender['disease'] = 'Stroke'
                stroke_gender.rename(columns={'stroke': 'has_disease', 'gender': 'Sex'}, inplace=True)
                
                # Extract gender and disease status from heart disease data
                heart_gender = self.heart_data[['Sex', 'HeartDisease']].copy()
                heart_gender['disease'] = 'Heart Disease'
                heart_gender.rename(columns={'HeartDisease': 'has_disease'}, inplace=True)
                
                # Print debug info
                print("Stroke data gender unique values:", stroke_gender['Sex'].unique())
                print("Heart disease data gender unique values:", heart_gender['Sex'].unique())
                print("Stroke data gender dtype:", stroke_gender['Sex'].dtype)
                print("Heart disease data gender dtype:", heart_gender['Sex'].dtype)
                
                # Standardize gender representation - simple method: values less than 0 are 0 (female), greater than 0 are 1 (male)
                # This is because standardized values may no longer be 0 and 1, but retain positive/negative relationship
                
                # Process stroke data - keep original numeric relationship
                stroke_gender['Sex_Numeric'] = np.where(stroke_gender['Sex'] < 0, 0, 1)
                
                # Process heart disease data - keep original numeric relationship
                heart_gender['Sex_Numeric'] = np.where(heart_gender['Sex'] < 0, 0, 1)
                
                # Combine the two datasets
                combined_gender = pd.concat([stroke_gender, heart_gender], ignore_index=True)
                
                # Map to displayable text
                combined_gender['Sex_Display'] = combined_gender['Sex_Numeric'].map({0: 'Female', 1: 'Male'})
                
                # Print debug info
                print("Combined gender unique values:", combined_gender['Sex_Numeric'].unique())
                print("Mapped display values:", combined_gender['Sex_Display'].unique())
                print("Total data records:", len(combined_gender))
                
                # Calculate disease prevalence by gender
                gender_rates = combined_gender.groupby(['disease', 'Sex_Display'])['has_disease'].mean().reset_index()
                gender_rates['percentage'] = gender_rates['has_disease'] * 100
                
                # Print prevalence rates
                print("Disease prevalence by gender:")
                print(gender_rates)
                
                # Plot grouped bar chart
                plt.figure(figsize=(10, 6))
                
                # Ensure data exists before plotting
                if not gender_rates.empty and len(gender_rates) >= 2:
                    sns.barplot(x='Sex_Display', y='percentage', hue='disease', data=gender_rates)
                    plt.title('Impact of Gender on Stroke and Heart Disease Prevalence')
                    plt.xlabel('Gender')
                    plt.ylabel('Prevalence (%)')
                    plt.legend(title='Disease Type')
                    
                    if save_plots:
                        plt.savefig('output/figures/gender_disease_comparison.png')
                    plt.close()
                else:
                    print("Warning: Insufficient gender prevalence data to plot chart")
                
            except Exception as e:
                print(f"Error processing gender data: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Datasets missing gender features, cannot perform gender comparison")
        
        print("Disease comparison visualization complete!")
    
    def run_all_visualizations(self):
        """Run all visualization functions"""
        self.plot_feature_distributions()
        self.plot_correlation_matrices()
        self.plot_feature_importance()
        self.plot_pair_plots()
        self.plot_disease_comparison()
        print("All visualization tasks complete!")

if __name__ == "__main__":
    # Create visualizer and load data
    visualizer = DataVisualizer()
    visualizer.load_processed_data()
    
    # Run all visualizations
    visualizer.run_all_visualizations()
