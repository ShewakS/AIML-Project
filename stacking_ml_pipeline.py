"""
Machine Learning Pipeline with Stacking Ensemble
Random Forest + Logistic Regression Stacking Implementation
"""

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import joblib

class MLPipeline:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.pca = None
        self.scaler = StandardScaler()
        
    def load_dataset(self, file_path, target_column):
        """
        Step 2: Load Dataset
        """
        print("Step 2: Loading Dataset...")
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
            self.target_column = target_column
            print(f"Dataset loaded successfully! Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def inspect_data(self):
        """
        Step 3: Inspect Data
        """
        print("\nStep 3: Data Inspection...")
        print("="*50)
        
        print("Dataset Info:")
        print(self.data.info())
        
        print("\nDataset Shape:", self.data.shape)
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nDataset Description:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\nDuplicate Rows:", self.data.duplicated().sum())
        
        print("\nTarget Variable Distribution:")
        print(self.data[self.target_column].value_counts())
        
        # Visualize target distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        self.data[self.target_column].value_counts().plot(kind='bar')
        plt.title('Target Variable Distribution')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        self.data[self.target_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Target Variable Proportion')
        plt.tight_layout()
        plt.show()
    
    def clean_text(self, text):
        """
        Helper function to clean text data
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_data(self, text_columns=None):
        """
        Step 4: Clean text, remove duplicates, handle missing values
        """
        print("\nStep 4: Data Preprocessing...")
        print("="*50)
        
        # Remove duplicates
        initial_shape = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_shape - self.data.shape[0]} duplicate rows")
        
        # Clean text columns if specified
        if text_columns:
            for col in text_columns:
                if col in self.data.columns:
                    print(f"Cleaning text in column: {col}")
                    self.data[col] = self.data[col].apply(self.clean_text)
        
        # Handle missing values
        print("\nHandling missing values...")
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if self.data[col].dtype == 'object':
                    # Fill categorical/text columns with mode or empty string
                    mode_val = self.data[col].mode()
                    if len(mode_val) > 0:
                        self.data[col].fillna(mode_val[0], inplace=True)
                    else:
                        self.data[col].fillna('', inplace=True)
                else:
                    # Fill numerical columns with median
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        
        print("Data preprocessing completed!")
        print(f"Final dataset shape: {self.data.shape}")
    
    def encode_categorical_features(self, categorical_columns=None):
        """
        Step 5: Encode categorical features
        """
        print("\nStep 5: Encoding Categorical Features...")
        print("="*50)
        
        if categorical_columns is None:
            # Auto-detect categorical columns
            categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()
            if self.target_column in categorical_columns:
                categorical_columns.remove(self.target_column)
        
        for col in categorical_columns:
            if col in self.data.columns and col != self.target_column:
                print(f"Encoding column: {col}")
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        # Encode target variable if it's categorical
        if self.data[self.target_column].dtype == 'object':
            print(f"Encoding target column: {self.target_column}")
            le_target = LabelEncoder()
            self.data[self.target_column] = le_target.fit_transform(self.data[self.target_column])
            self.label_encoders[self.target_column] = le_target
        
        print("Categorical encoding completed!")
    
    def apply_tfidf_vectorization(self, text_columns=None, max_features=5000):
        """
        Step 6: Apply TF-IDF vectorization
        """
        print("\nStep 6: Applying TF-IDF Vectorization...")
        print("="*50)
        
        if text_columns is None:
            # Try to identify text columns (columns with string data)
            text_columns = []
            for col in self.data.columns:
                if col != self.target_column and self.data[col].dtype == 'object':
                    # Check if column contains text (average length > 10)
                    avg_length = self.data[col].astype(str).str.len().mean()
                    if avg_length > 10:
                        text_columns.append(col)
        
        if text_columns:
            print(f"Applying TF-IDF to columns: {text_columns}")
            
            # Combine text columns
            combined_text = self.data[text_columns].astype(str).agg(' '.join, axis=1)
            
            # Apply TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            tfidf_features = self.tfidf_vectorizer.fit_transform(combined_text)
            
            # Convert to DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )
            
            # Remove original text columns and add TF-IDF features
            self.data = self.data.drop(columns=text_columns)
            self.data = pd.concat([self.data.reset_index(drop=True), tfidf_df], axis=1)
            
            print(f"TF-IDF vectorization completed! Added {tfidf_features.shape[1]} features")
        else:
            print("No text columns found for TF-IDF vectorization")
        
        # Prepare features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target vector shape: {self.y.shape}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Step 7: Split into train/test
        """
        print("\nStep 7: Splitting Data...")
        print("="*50)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training target distribution:\n{pd.Series(self.y_train).value_counts()}")
        print(f"Test target distribution:\n{pd.Series(self.y_test).value_counts()}")
    
    def apply_pca(self, n_components=0.95):
        """
        Step 8: Reduce dimensions with PCA
        """
        print("\nStep 8: Applying PCA for Dimensionality Reduction...")
        print("="*50)
        
        # Scale features before PCA
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        print(f"Original feature dimensions: {self.X_train.shape[1]}")
        print(f"PCA reduced dimensions: {self.X_train_pca.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(1, min(21, len(self.pca.explained_variance_ratio_) + 1)), 
                self.pca.explained_variance_ratio_[:20])
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Individual Component Variance')
        plt.tight_layout()
        plt.show()
    
    def train_stacking_model(self):
        """
        Step 9: Train Stacking ML Model (Random Forest + Logistic Regression)
        """
        print("\nStep 9: Training Stacking Ensemble Model...")
        print("="*50)
        
        # Define base estimators
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        lr_classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
        
        # Create base estimators list
        base_estimators = [
            ('random_forest', rf_classifier),
            ('logistic_regression', lr_classifier)
        ]
        
        # Create stacking classifier
        self.model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print("Training stacking ensemble...")
        self.model.fit(self.X_train_pca, self.y_train)
        
        print("Model training completed!")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, self.X_train_pca, self.y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_model(self):
        """
        Step 10: Evaluate and visualize results
        """
        print("\nStep 10: Model Evaluation and Visualization...")
        print("="*50)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_pca)
        y_pred_proba = self.model.predict_proba(self.X_test_pca)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print("Model Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Confusion Matrix
        plt.subplot(2, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curve (for binary classification)
        if len(np.unique(self.y)) == 2:
            plt.subplot(2, 3, 2)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            print(f"AUC Score: {auc_score:.4f}")
        
        # Feature Importance (from Random Forest base estimator)
        plt.subplot(2, 3, 3)
        rf_model = self.model.named_estimators_['random_forest']
        feature_importance = rf_model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:]
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Feature Importances (Random Forest)')
        plt.yticks(range(len(top_features)), [f'Feature_{i}' for i in top_features])
        
        # Prediction Distribution
        plt.subplot(2, 3, 4)
        plt.hist(y_pred, bins=len(np.unique(self.y)), alpha=0.7, label='Predictions')
        plt.hist(self.y_test, bins=len(np.unique(self.y)), alpha=0.7, label='True Values')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Prediction vs True Distribution')
        plt.legend()
        
        # Model Comparison (Individual vs Stacking)
        plt.subplot(2, 3, 5)
        models_comparison = []
        
        # Individual Random Forest
        rf_individual = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_individual.fit(self.X_train_pca, self.y_train)
        rf_pred = rf_individual.predict(self.X_test_pca)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        models_comparison.append(('Random Forest', rf_accuracy))
        
        # Individual Logistic Regression
        lr_individual = LogisticRegression(max_iter=1000, random_state=42)
        lr_individual.fit(self.X_train_pca, self.y_train)
        lr_pred = lr_individual.predict(self.X_test_pca)
        lr_accuracy = accuracy_score(self.y_test, lr_pred)
        models_comparison.append(('Logistic Regression', lr_accuracy))
        
        # Stacking Ensemble
        models_comparison.append(('Stacking Ensemble', accuracy))
        
        models, accuracies = zip(*models_comparison)
        plt.bar(models, accuracies)
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.xticks(rotation=45)
        
        # Learning Curve
        plt.subplot(2, 3, 6)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            n_samples = int(train_size * len(self.X_train_pca))
            temp_model = StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('lr', LogisticRegression(max_iter=500, random_state=42))
                ],
                final_estimator=LogisticRegression(random_state=42),
                cv=3
            )
            temp_model.fit(self.X_train_pca[:n_samples], self.y_train[:n_samples])
            
            train_pred = temp_model.predict(self.X_train_pca[:n_samples])
            val_pred = temp_model.predict(self.X_test_pca)
            
            train_scores.append(accuracy_score(self.y_train[:n_samples], train_pred))
            val_scores.append(accuracy_score(self.y_test, val_pred))
        
        plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
        plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
        plt.xlabel('Training Set Size Ratio')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'models_comparison': models_comparison
        }
    
    def save_model(self, filename='stacking_model.pkl'):
        """
        Save the trained model and preprocessors
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='stacking_model.pkl'):
        """
        Load a saved model and preprocessors
        """
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.label_encoders = model_data['label_encoders']
        print(f"Model loaded from {filename}")

# Example usage function
def run_complete_pipeline(file_path, target_column, text_columns=None, categorical_columns=None):
    """
    Run the complete ML pipeline
    """
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Step 1: Already done (imports)
    print("Step 1: Libraries imported successfully!")
    
    # Step 2: Load dataset
    data = pipeline.load_dataset(file_path, target_column)
    if data is None:
        return None
    
    # Step 3: Inspect data
    pipeline.inspect_data()
    
    # Step 4: Preprocess data
    pipeline.preprocess_data(text_columns=text_columns)
    
    # Step 5: Encode categorical features
    pipeline.encode_categorical_features(categorical_columns=categorical_columns)
    
    # Step 6: Apply TF-IDF vectorization
    pipeline.apply_tfidf_vectorization(text_columns=text_columns)
    
    # Step 7: Split data
    pipeline.split_data()
    
    # Step 8: Apply PCA
    pipeline.apply_pca()
    
    # Step 9: Train stacking model
    pipeline.train_stacking_model()
    
    # Step 10: Evaluate model
    results = pipeline.evaluate_model()
    
    # Save model
    pipeline.save_model()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return pipeline, results

if __name__ == "__main__":
    # Example usage - Replace with your dataset path and parameters
    print("ML Pipeline with Stacking Ensemble (Random Forest + Logistic Regression)")
    print("="*70)
    
    # You need to specify your dataset path and target column
    # Example:
    # file_path = "your_dataset.csv"
    # target_column = "target"
    # text_columns = ["text_column1", "text_column2"]  # Optional
    # categorical_columns = ["cat_column1", "cat_column2"]  # Optional
    
    # pipeline, results = run_complete_pipeline(
    #     file_path=file_path,
    #     target_column=target_column,
    #     text_columns=text_columns,
    #     categorical_columns=categorical_columns
    # )
    
    print("To use this pipeline:")
    print("1. Replace the file_path with your dataset path")
    print("2. Specify the target_column name")
    print("3. Optionally specify text_columns and categorical_columns")
    print("4. Run the script!")

