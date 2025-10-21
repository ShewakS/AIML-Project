"""
Data Leakage Analysis and Fixed Pipeline
Critical Issue: Current pipeline has data leakage problems
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

def analyze_data_leakage():
    """Analyze potential data leakage issues in the current pipeline"""
    print("🔍 DATA LEAKAGE ANALYSIS")
    print("=" * 50)
    
    print("❌ CRITICAL ISSUES FOUND IN CURRENT PIPELINE:")
    print()
    
    print("1. 🚨 TARGET ENCODING BEFORE SPLIT:")
    print("   - Target encoding is done on the ENTIRE dataset")
    print("   - This leaks information from test set into training")
    print("   - Lines 147-148 in final_optimized_pipeline.py")
    print()
    
    print("2. 🚨 TF-IDF FITTING ON ENTIRE DATASET:")
    print("   - TF-IDF vectorizer is fit on ALL data before split")
    print("   - Test set vocabulary leaks into training features")
    print("   - Lines 163-171 in final_optimized_pipeline.py")
    print()
    
    print("3. 🚨 FEATURE ENGINEERING BEFORE SPLIT:")
    print("   - All feature engineering happens before train/test split")
    print("   - Statistics from test set influence training features")
    print()
    
    print("4. 🚨 DATA BALANCING ON ENTIRE DATASET:")
    print("   - Balancing is done before splitting")
    print("   - Can create artificial similarities between train/test")
    print()
    
    print("✅ WHAT SHOULD HAPPEN:")
    print("1. Split data FIRST (before any preprocessing)")
    print("2. Fit all transformers ONLY on training data")
    print("3. Transform test data using fitted transformers")
    print("4. Never use test data statistics in training")
    print()

class LeakageFreeMLPipeline:
    """Data leakage-free ML pipeline with proper train/test separation"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.preprocessors = {}
        
    def load_dataset(self, file_path, target_column):
        """Load dataset with minimal preprocessing"""
        print("📁 Loading dataset...")
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        
        # Only basic cleaning that doesn't use target information
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['Passage', 'Question'])
        print(f"Removed {initial_size - len(self.data)} duplicates")
        
        # Basic feature engineering (no target leakage)
        self.data = self._add_basic_features(self.data)
        
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def _add_basic_features(self, df):
        """Add basic features that don't use target information"""
        print("🔧 Adding basic features (no target leakage)...")
        
        # Text length features
        df['passage_length'] = df['Passage'].str.len()
        df['question_length'] = df['Question'].str.len()
        df['passage_word_count'] = df['Passage'].str.split().str.len()
        df['question_word_count'] = df['Question'].str.split().str.len()
        
        # Question type features
        df['is_what_question'] = df['Question'].str.contains(r'\bwhat\b', case=False, na=False).astype(int)
        df['is_who_question'] = df['Question'].str.contains(r'\bwho\b', case=False, na=False).astype(int)
        df['is_where_question'] = df['Question'].str.contains(r'\bwhere\b', case=False, na=False).astype(int)
        df['is_when_question'] = df['Question'].str.contains(r'\bwhen\b', case=False, na=False).astype(int)
        df['is_why_question'] = df['Question'].str.contains(r'\bwhy\b', case=False, na=False).astype(int)
        df['is_how_question'] = df['Question'].str.contains(r'\bhow\b', case=False, na=False).astype(int)
        
        # Subject complexity (predefined mapping - no target leakage)
        subject_complexity = {
            'Mathematics': 5, 'Science': 4, 'Computer Science': 4,
            'History': 3, 'Geography': 3, 'English': 2, 'Economics': 3
        }
        df['subject_complexity'] = df['Subject'].map(subject_complexity).fillna(3)
        
        # Difficulty mapping (predefined - no target leakage)
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df['difficulty_numeric'] = df['Difficulty_Level'].map(difficulty_map)
        
        # Word overlap (no target information used)
        df['question_passage_overlap'] = df.apply(self._calculate_word_overlap, axis=1)
        
        return df
    
    def _calculate_word_overlap(self, row):
        """Calculate word overlap between question and passage"""
        passage_words = set(str(row['Passage']).lower().split())
        question_words = set(str(row['Question']).lower().split())
        
        if len(passage_words) == 0 or len(question_words) == 0:
            return 0
        
        overlap = len(passage_words.intersection(question_words))
        return overlap / len(question_words)
    
    def proper_train_test_split(self, test_size=0.2, random_state=42):
        """Proper train/test split BEFORE any preprocessing"""
        print("✂️ Proper train/test split (BEFORE preprocessing)...")
        
        # Prepare features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Split FIRST - this is crucial
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train_raw.shape[0]} samples")
        print(f"Test set: {self.X_test_raw.shape[0]} samples")
        print(f"Training target distribution:\n{pd.Series(self.y_train).value_counts().sort_index()}")
        print(f"Test target distribution:\n{pd.Series(self.y_test).value_counts().sort_index()}")
        
        return self.X_train_raw, self.X_test_raw, self.y_train, self.y_test
    
    def leakage_free_preprocessing(self, text_columns, categorical_columns):
        """Preprocessing that prevents data leakage"""
        print("🛡️ Leakage-free preprocessing...")
        
        # Text cleaning (no leakage)
        for col in text_columns:
            if col in self.X_train_raw.columns:
                self.X_train_raw[col] = self.X_train_raw[col].fillna('').astype(str)
                self.X_test_raw[col] = self.X_test_raw[col].fillna('').astype(str)
        
        # Categorical encoding - FIT ONLY ON TRAINING DATA
        for col in categorical_columns:
            if col in self.X_train_raw.columns:
                # NO TARGET ENCODING - this was causing major leakage
                
                # Label encoding - fit only on training data
                le = LabelEncoder()
                self.X_train_raw[col] = le.fit_transform(self.X_train_raw[col].astype(str))
                
                # Transform test data using training-fitted encoder
                # Handle unseen categories
                test_values = self.X_test_raw[col].astype(str)
                test_encoded = []
                for val in test_values:
                    if val in le.classes_:
                        test_encoded.append(le.transform([val])[0])
                    else:
                        # Assign most frequent class for unseen categories
                        test_encoded.append(le.transform([le.classes_[0]])[0])
                
                self.X_test_raw[col] = test_encoded
                self.preprocessors[f'{col}_encoder'] = le
        
        print("✅ Categorical encoding completed without leakage")
    
    def leakage_free_text_features(self, text_columns):
        """Create text features without data leakage"""
        print("📝 Creating text features (leakage-free)...")
        
        # Combine text - separately for train and test
        train_combined_text = self.X_train_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        test_combined_text = self.X_test_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # TF-IDF - FIT ONLY ON TRAINING DATA
        tfidf = TfidfVectorizer(
            max_features=2000,  # Reduced to prevent overfitting
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        
        # Fit on training data only
        train_tfidf_features = tfidf.fit_transform(train_combined_text)
        
        # Transform test data using training-fitted vectorizer
        test_tfidf_features = tfidf.transform(test_combined_text)
        
        self.preprocessors['tfidf'] = tfidf
        
        # Convert to DataFrames
        feature_names = [f'tfidf_{i}' for i in range(train_tfidf_features.shape[1])]
        
        train_tfidf_df = pd.DataFrame(
            train_tfidf_features.toarray(),
            columns=feature_names,
            index=self.X_train_raw.index
        )
        
        test_tfidf_df = pd.DataFrame(
            test_tfidf_features.toarray(),
            columns=feature_names,
            index=self.X_test_raw.index
        )
        
        # Combine with non-text features
        non_text_cols = [col for col in self.X_train_raw.columns if col not in text_columns]
        
        self.X_train = pd.concat([
            self.X_train_raw[non_text_cols].reset_index(drop=True), 
            train_tfidf_df.reset_index(drop=True)
        ], axis=1)
        
        self.X_test = pd.concat([
            self.X_test_raw[non_text_cols].reset_index(drop=True), 
            test_tfidf_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"✅ Created {train_tfidf_features.shape[1]} text features without leakage")
    
    def leakage_free_feature_engineering(self):
        """Feature engineering without data leakage"""
        print("🔬 Feature engineering (leakage-free)...")
        
        # Feature selection - FIT ONLY ON TRAINING DATA
        selector = SelectKBest(score_func=mutual_info_classif, k=min(300, self.X_train.shape[1]))
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)
        self.preprocessors['feature_selector'] = selector
        
        # Scaling - FIT ONLY ON TRAINING DATA
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.preprocessors['scaler'] = scaler
        
        # Dimensionality reduction - FIT ONLY ON TRAINING DATA
        svd = TruncatedSVD(n_components=min(100, X_train_scaled.shape[1] - 1), random_state=42)
        self.X_train_final = svd.fit_transform(X_train_scaled)
        self.X_test_final = svd.transform(X_test_scaled)
        self.preprocessors['svd'] = svd
        
        print(f"✅ Final features: {self.X_train_final.shape[1]}")
        print(f"✅ Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    def train_leakage_free_model(self):
        """Train model without any data leakage"""
        print("🚀 Training leakage-free model...")
        
        # Use proven effective models
        models = {
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                random_state=42
            ),
            'svm': SVC(
                C=1,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        }
        
        # Cross-validation on training data only
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_scores = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
            model_scores[name] = cv_scores.mean()
            print(f"{name} CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Select best models for ensemble
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        # Create ensemble
        base_estimators = [(name, models[name]) for name, score in top_models]
        
        self.best_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1, random_state=42),
            cv=cv,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train on training data only
        self.best_model.fit(self.X_train_final, self.y_train)
        
        # Cross-validation score
        cv_score = cross_val_score(self.best_model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
        print(f"Final model CV accuracy: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")
        
        return model_scores
    
    def honest_evaluation(self):
        """Honest evaluation on truly unseen test data"""
        print("📊 Honest evaluation on unseen test data...")
        
        # Predictions on truly unseen test data
        y_pred = self.best_model.predict(self.X_test_final)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🎯 HONEST MODEL PERFORMANCE (No Data Leakage):")
        print(f"🏆 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Honest Model Performance (No Data Leakage)\nTest Accuracy: {accuracy:.2%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'test_accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def save_leakage_free_model(self, filename='leakage_free_model.pkl'):
        """Save the leakage-free model"""
        model_data = {
            'model': self.best_model,
            'preprocessors': self.preprocessors
        }
        joblib.dump(model_data, filename)
        print(f"Leakage-free model saved as {filename}")

def run_leakage_free_pipeline(file_path, target_column):
    """Run the leakage-free pipeline"""
    print("🛡️ LEAKAGE-FREE ML PIPELINE")
    print("=" * 50)
    
    # First, analyze the leakage issues
    analyze_data_leakage()
    
    print("\n" + "="*50)
    print("🔧 RUNNING CORRECTED PIPELINE")
    print("="*50)
    
    pipeline = LeakageFreeMLPipeline()
    
    # Load dataset
    pipeline.load_dataset(file_path, target_column)
    
    # CRITICAL: Split FIRST before any preprocessing
    pipeline.proper_train_test_split(test_size=0.2, random_state=42)
    
    # Leakage-free preprocessing
    text_columns = ['Passage', 'Question', 'Options', 'Keywords', 'Hint', 'Explanation']
    categorical_columns = ['domain', 'question_type', 'Question_Type', 'Subject', 'Difficulty_Level']
    
    pipeline.leakage_free_preprocessing(text_columns, categorical_columns)
    
    # Leakage-free feature engineering
    pipeline.leakage_free_text_features(text_columns)
    pipeline.leakage_free_feature_engineering()
    
    # Train model
    model_scores = pipeline.train_leakage_free_model()
    
    # Honest evaluation
    results = pipeline.honest_evaluation()
    
    # Save model
    pipeline.save_leakage_free_model()
    
    return pipeline, results, model_scores

if __name__ == "__main__":
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"
    
    pipeline, results, model_scores = run_leakage_free_pipeline(file_path, target_column)
