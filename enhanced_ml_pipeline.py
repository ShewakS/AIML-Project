"""
Enhanced ML Pipeline for 80-90% Accuracy
Advanced feature engineering, hyperparameter tuning, and ensemble optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Advanced sklearn imports
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   StratifiedKFold, RandomizedSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier, 
                            ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from scipy.stats import uniform, randint
import joblib

# Try to import optional dependencies
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except:
    TEXTSTAT_AVAILABLE = False

class EnhancedMLPipeline:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.feature_selector = None
        self.dimensionality_reducer = None
        self.scaler = RobustScaler()
        
    def load_and_enhance_dataset(self, file_path, target_column):
        """Enhanced data loading with quality improvements"""
        print("🔄 Loading and enhancing dataset...")
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        
        # Remove duplicates more intelligently
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['Passage', 'Question', 'Correct_Option'])
        print(f"Removed {initial_size - len(self.data)} duplicate questions")
        
        # Add advanced text features
        self.data = self._add_text_features(self.data)
        
        # Balance dataset if needed
        self.data = self._balance_dataset(self.data)
        
        print(f"Enhanced dataset shape: {self.data.shape}")
        return self.data
    
    def _add_text_features(self, df):
        """Add sophisticated text-based features"""
        print("📝 Adding advanced text features...")
        
        # Passage features
        df['passage_length'] = df['Passage'].str.len()
        df['passage_word_count'] = df['Passage'].str.split().str.len()
        df['passage_sentence_count'] = df['Passage'].str.count(r'[.!?]+')
        df['passage_avg_word_length'] = df['Passage'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) else 0
        )
        
        # Question features
        df['question_length'] = df['Question'].str.len()
        df['question_word_count'] = df['Question'].str.split().str.len()
        df['question_mark_count'] = df['Question'].str.count(r'\?')
        
        # Question type indicators
        df['is_what_question'] = df['Question'].str.contains(r'\bwhat\b', case=False, na=False).astype(int)
        df['is_who_question'] = df['Question'].str.contains(r'\bwho\b', case=False, na=False).astype(int)
        df['is_where_question'] = df['Question'].str.contains(r'\bwhere\b', case=False, na=False).astype(int)
        df['is_when_question'] = df['Question'].str.contains(r'\bwhen\b', case=False, na=False).astype(int)
        df['is_why_question'] = df['Question'].str.contains(r'\bwhy\b', case=False, na=False).astype(int)
        df['is_how_question'] = df['Question'].str.contains(r'\bhow\b', case=False, na=False).astype(int)
        
        # Options features
        df['options_total_length'] = df['Options'].str.len()
        df['options_count'] = df['Options'].str.count(r"'[^']*'")
        
        # Readability features (if textstat is available)
        if TEXTSTAT_AVAILABLE:
            try:
                df['passage_readability'] = df['Passage'].apply(
                    lambda x: flesch_reading_ease(str(x)) if pd.notna(x) else 0
                )
                df['passage_grade_level'] = df['Passage'].apply(
                    lambda x: flesch_kincaid_grade(str(x)) if pd.notna(x) else 0
                )
            except:
                df['passage_readability'] = 0
                df['passage_grade_level'] = 0
        else:
            # Simple readability approximation
            df['passage_readability'] = df['Passage'].apply(
                lambda x: 100 - min(len(str(x).split()), 100) if pd.notna(x) else 0
            )
            df['passage_grade_level'] = df['Passage'].apply(
                lambda x: min(len(str(x).split()) / 10, 12) if pd.notna(x) else 0
            )
        
        # Keyword matching features
        df['keyword_in_passage'] = df.apply(
            lambda row: 1 if pd.notna(row['Keywords']) and 
            any(keyword.lower() in str(row['Passage']).lower() 
                for keyword in str(row['Keywords']).split(',')) else 0, axis=1
        )
        
        return df
    
    def _balance_dataset(self, df):
        """Balance dataset to improve training"""
        print("⚖️ Balancing dataset...")
        
        target_counts = df[self.target_column].value_counts()
        min_count = target_counts.min()
        max_count = target_counts.max()
        
        if max_count / min_count > 1.5:  # If imbalanced
            balanced_dfs = []
            for target_value in target_counts.index:
                class_df = df[df[self.target_column] == target_value]
                # Sample up to 1.2 * min_count to maintain some diversity
                sample_size = min(len(class_df), int(1.2 * min_count))
                balanced_dfs.append(class_df.sample(n=sample_size, random_state=42))
            
            df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
            print(f"Balanced dataset to {len(df)} samples")
        
        return df
    
    def advanced_preprocessing(self, text_columns, categorical_columns):
        """Enhanced preprocessing with multiple techniques"""
        print("🔧 Advanced preprocessing...")
        
        # Enhanced text cleaning
        for col in text_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(self._advanced_text_cleaning)
        
        # Encode categorical features with frequency encoding for high cardinality
        for col in categorical_columns:
            if col in self.data.columns:
                if self.data[col].nunique() > 10:  # High cardinality
                    freq_map = self.data[col].value_counts().to_dict()
                    self.data[f'{col}_frequency'] = self.data[col].map(freq_map)
                
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle target encoding
        if self.data[self.target_column].dtype == 'object':
            le_target = LabelEncoder()
            self.data[self.target_column] = le_target.fit_transform(self.data[self.target_column])
            self.label_encoders[self.target_column] = le_target
    
    def _advanced_text_cleaning(self, text):
        """More sophisticated text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove extra whitespace but preserve sentence structure
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation that matters
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        # Normalize case
        text = text.lower()
        
        return text.strip()
    
    def create_advanced_features(self, text_columns):
        """Create multiple types of text features"""
        print("🎯 Creating advanced text features...")
        
        # Combine all text columns
        combined_text = self.data[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # TF-IDF with multiple configurations
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(combined_text)
        
        # Count vectorizer for different perspective
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        count_features = self.count_vectorizer.fit_transform(combined_text)
        
        # Combine features
        from scipy.sparse import hstack
        text_features = hstack([tfidf_features, count_features])
        
        # Convert to DataFrame
        feature_names = (
            [f'tfidf_{i}' for i in range(tfidf_features.shape[1])] +
            [f'count_{i}' for i in range(count_features.shape[1])]
        )
        
        text_df = pd.DataFrame(text_features.toarray(), columns=feature_names)
        
        # Remove original text columns and add new features
        non_text_cols = [col for col in self.data.columns if col not in text_columns]
        self.data = pd.concat([
            self.data[non_text_cols].reset_index(drop=True), 
            text_df
        ], axis=1)
        
        print(f"Created {text_features.shape[1]} text features")
    
    def feature_selection_and_reduction(self):
        """Advanced feature selection and dimensionality reduction"""
        print("🎯 Feature selection and dimensionality reduction...")
        
        # Prepare features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        # Split data first
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Feature selection using mutual information
        self.feature_selector = SelectKBest(
            score_func=mutual_info_classif, 
            k=min(500, self.X_train.shape[1])  # Select top 500 features
        )
        
        X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.feature_selector.transform(self.X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Dimensionality reduction with TruncatedSVD (works better with sparse features)
        n_components = min(50, X_train_scaled.shape[1] - 1)
        self.dimensionality_reducer = TruncatedSVD(n_components=n_components, random_state=42)
        
        self.X_train_final = self.dimensionality_reducer.fit_transform(X_train_scaled)
        self.X_test_final = self.dimensionality_reducer.transform(X_test_scaled)
        
        print(f"Final feature dimensions: {self.X_train_final.shape[1]}")
        print(f"Explained variance ratio: {self.dimensionality_reducer.explained_variance_ratio_.sum():.4f}")
    
    def train_optimized_ensemble(self):
        """Train multiple optimized models with hyperparameter tuning"""
        print("🚀 Training optimized ensemble with hyperparameter tuning...")
        
        # Define base models with different strengths
        models = {
            'rf': RandomForestClassifier(random_state=42, n_jobs=-1),
            'et': ExtraTreesClassifier(random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Hyperparameter grids
        param_grids = {
            'rf': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'et': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'lr': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
            }
        }
        
        # Tune each model
        tuned_models = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"Tuning {name}...")
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                model, param_grids[name], 
                n_iter=20, cv=cv, scoring='accuracy',
                random_state=42, n_jobs=-1
            )
            
            search.fit(self.X_train_final, self.y_train)
            tuned_models[name] = search.best_estimator_
            
            print(f"{name} best score: {search.best_score_:.4f}")
        
        # Create stacking ensemble with best models
        base_estimators = [
            ('rf_tuned', tuned_models['rf']),
            ('et_tuned', tuned_models['et']),
            ('gb_tuned', tuned_models['gb']),
            ('svm_tuned', tuned_models['svm'])
        ]
        
        # Use tuned logistic regression as meta-learner
        self.best_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=tuned_models['lr'],
            cv=cv,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        print("Training final stacking ensemble...")
        self.best_model.fit(self.X_train_final, self.y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.best_model, self.X_train_final, self.y_train, 
            cv=cv, scoring='accuracy'
        )
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return tuned_models
    
    def comprehensive_evaluation(self):
        """Comprehensive model evaluation"""
        print("📊 Comprehensive model evaluation...")
        
        # Predictions
        y_pred = self.best_model.predict(self.X_test_final)
        y_pred_proba = self.best_model.predict_proba(self.X_test_final)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"🎯 ENHANCED MODEL PERFORMANCE:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix visualization
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Enhanced Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def save_enhanced_model(self, filename='enhanced_reading_comprehension_model.pkl'):
        """Save the enhanced model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'dimensionality_reducer': self.dimensionality_reducer,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filename)
        print(f"Enhanced model saved as {filename}")

def run_enhanced_pipeline(file_path, target_column):
    """Run the enhanced pipeline for 80-90% accuracy"""
    print("🚀 ENHANCED ML PIPELINE FOR HIGH ACCURACY")
    print("=" * 60)
    
    pipeline = EnhancedMLPipeline()
    
    # Load and enhance dataset
    pipeline.load_and_enhance_dataset(file_path, target_column)
    
    # Advanced preprocessing
    text_columns = ['Passage', 'Question', 'Options', 'Keywords', 'Hint', 'Explanation']
    categorical_columns = ['domain', 'question_type', 'Question_Type', 'Subject', 'Difficulty_Level']
    
    pipeline.advanced_preprocessing(text_columns, categorical_columns)
    
    # Create advanced features
    pipeline.create_advanced_features(text_columns)
    
    # Feature selection and reduction
    pipeline.feature_selection_and_reduction()
    
    # Train optimized ensemble
    tuned_models = pipeline.train_optimized_ensemble()
    
    # Comprehensive evaluation
    results = pipeline.comprehensive_evaluation()
    
    # Save model
    pipeline.save_enhanced_model()
    
    return pipeline, results, tuned_models

if __name__ == "__main__":
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"
    
    pipeline, results, tuned_models = run_enhanced_pipeline(file_path, target_column)
