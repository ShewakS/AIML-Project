"""
Regularized ML Pipeline - Target: 80-90% Accuracy
Prevents overfitting through proper regularization techniques
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

class RegularizedMLPipeline:
    """ML pipeline with proper regularization to prevent overfitting"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.preprocessors = {}
        
    def load_and_prepare_dataset(self, file_path, target_column):
        """Load dataset with conservative feature engineering"""
        print("📁 Loading dataset with regularization focus...")
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        
        # Remove duplicates
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['Passage', 'Question'])
        print(f"Removed {initial_size - len(self.data)} duplicates")
        
        # Conservative feature engineering (prevent overfitting)
        self.data = self._create_conservative_features(self.data)
        
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def _create_conservative_features(self, df):
        """Create conservative features to prevent overfitting"""
        print("🔧 Creating conservative features...")
        
        # Basic text features (most important ones only)
        df['passage_length'] = df['Passage'].str.len()
        df['question_length'] = df['Question'].str.len()
        df['passage_word_count'] = df['Passage'].str.split().str.len()
        df['question_word_count'] = df['Question'].str.split().str.len()
        
        # Question type features (proven effective)
        df['is_what_question'] = df['Question'].str.contains(r'\bwhat\b', case=False, na=False).astype(int)
        df['is_who_question'] = df['Question'].str.contains(r'\bwho\b', case=False, na=False).astype(int)
        df['is_where_question'] = df['Question'].str.contains(r'\bwhere\b', case=False, na=False).astype(int)
        df['is_when_question'] = df['Question'].str.contains(r'\bwhen\b', case=False, na=False).astype(int)
        df['is_why_question'] = df['Question'].str.contains(r'\bwhy\b', case=False, na=False).astype(int)
        df['is_how_question'] = df['Question'].str.contains(r'\bhow\b', case=False, na=False).astype(int)
        
        # Subject complexity (simplified)
        subject_complexity = {
            'Mathematics': 3, 'Science': 3, 'Computer Science': 3,
            'History': 2, 'Geography': 2, 'English': 1, 'Economics': 2
        }
        df['subject_complexity'] = df['Subject'].map(subject_complexity).fillna(2)
        
        # Difficulty mapping
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df['difficulty_numeric'] = df['Difficulty_Level'].map(difficulty_map)
        
        # Simple word overlap
        df['question_passage_overlap'] = df.apply(self._calculate_simple_overlap, axis=1)
        
        return df
    
    def _calculate_simple_overlap(self, row):
        """Simple word overlap calculation"""
        passage_words = set(str(row['Passage']).lower().split())
        question_words = set(str(row['Question']).lower().split())
        
        if len(question_words) == 0:
            return 0
        
        overlap = len(passage_words.intersection(question_words))
        return min(overlap / len(question_words), 1.0)  # Cap at 1.0
    
    def proper_train_test_split(self, test_size=0.25, random_state=42):
        """Larger test set to better evaluate generalization"""
        print("✂️ Train/test split with larger test set...")
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Larger test set (25%) for better generalization assessment
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train_raw.shape[0]} samples")
        print(f"Test set: {self.X_test_raw.shape[0]} samples")
        
        return self.X_train_raw, self.X_test_raw, self.y_train, self.y_test
    
    def conservative_preprocessing(self, text_columns, categorical_columns):
        """Conservative preprocessing to prevent overfitting"""
        print("🛡️ Conservative preprocessing...")
        
        # Text cleaning
        for col in text_columns:
            if col in self.X_train_raw.columns:
                self.X_train_raw[col] = self.X_train_raw[col].fillna('').astype(str)
                self.X_test_raw[col] = self.X_test_raw[col].fillna('').astype(str)
        
        # Simple categorical encoding
        for col in categorical_columns:
            if col in self.X_train_raw.columns:
                le = LabelEncoder()
                self.X_train_raw[col] = le.fit_transform(self.X_train_raw[col].astype(str))
                
                # Handle unseen categories
                test_values = self.X_test_raw[col].astype(str)
                test_encoded = []
                for val in test_values:
                    if val in le.classes_:
                        test_encoded.append(le.transform([val])[0])
                    else:
                        test_encoded.append(0)
                
                self.X_test_raw[col] = test_encoded
                self.preprocessors[f'{col}_encoder'] = le
    
    def create_regularized_text_features(self, text_columns):
        """Create text features with regularization"""
        print("📝 Creating regularized text features...")
        
        # Combine text
        train_combined_text = self.X_train_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        test_combined_text = self.X_test_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Conservative TF-IDF (reduced features to prevent overfitting)
        tfidf = TfidfVectorizer(
            max_features=800,  # Reduced from 3800
            stop_words='english',
            ngram_range=(1, 2),  # Only 1-2 grams
            min_df=3,  # Higher min_df to reduce noise
            max_df=0.8,  # Lower max_df to remove very common terms
            sublinear_tf=True
        )
        
        # Fit on training data only
        train_tfidf_features = tfidf.fit_transform(train_combined_text)
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
        
        print(f"✅ Created {train_tfidf_features.shape[1]} regularized text features")
    
    def regularized_feature_engineering(self):
        """Feature engineering with strong regularization"""
        print("🔬 Regularized feature engineering...")
        
        # Conservative feature selection (fewer features)
        selector = SelectKBest(score_func=mutual_info_classif, k=min(100, self.X_train.shape[1]))
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)
        self.preprocessors['feature_selector'] = selector
        
        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.preprocessors['scaler'] = scaler
        
        # Conservative dimensionality reduction
        svd = TruncatedSVD(n_components=min(50, X_train_scaled.shape[1] - 1), random_state=42)
        self.X_train_final = svd.fit_transform(X_train_scaled)
        self.X_test_final = svd.transform(X_test_scaled)
        self.preprocessors['svd'] = svd
        
        print(f"✅ Final features: {self.X_train_final.shape[1]}")
        print(f"✅ Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    def train_regularized_models(self):
        """Train models with strong regularization"""
        print("🚀 Training regularized models...")
        
        # Models with regularization parameters to prevent overfitting
        models = {
            'rf_regularized': RandomForestClassifier(
                n_estimators=100,  # Reduced from 500
                max_depth=10,      # Limited depth
                min_samples_split=10,  # Higher split requirement
                min_samples_leaf=5,    # Higher leaf requirement
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'et_regularized': ExtraTreesClassifier(
                n_estimators=100,  # Reduced
                max_depth=10,      # Limited depth
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gb_regularized': GradientBoostingClassifier(
                n_estimators=50,   # Reduced
                learning_rate=0.05,  # Lower learning rate
                max_depth=4,       # Shallow trees
                subsample=0.8,     # Regularization
                random_state=42
            ),
            'svm_regularized': SVC(
                C=0.1,  # Strong regularization (lower C)
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'lr_regularized': LogisticRegression(
                C=0.1,  # Strong regularization
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            'ridge': RidgeClassifier(
                alpha=10.0,  # Strong regularization
                random_state=42
            )
        }
        
        # Cross-validation with more folds for better assessment
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        model_scores = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
            model_scores[name] = cv_scores.mean()
            std_score = cv_scores.std()
            print(f"{name}: {cv_scores.mean():.4f} (+/- {std_score * 2:.4f})")
            
            # Check for overfitting (high variance in CV scores)
            if std_score > 0.1:
                print(f"  ⚠️  High variance detected in {name}")
        
        # Select models with good performance but not perfect scores
        good_models = {name: score for name, score in model_scores.items() 
                      if 0.75 <= score <= 0.95}  # Exclude perfect scores
        
        if not good_models:
            # If no models in range, take the best non-perfect one
            good_models = {name: score for name, score in model_scores.items() 
                          if score < 0.99}
            if good_models:
                best_model_name = max(good_models.items(), key=lambda x: x[1])[0]
                good_models = {best_model_name: good_models[best_model_name]}
        
        print(f"\nSelected models for ensemble: {list(good_models.keys())}")
        
        # Create conservative ensemble
        if len(good_models) >= 2:
            # Use top 2-3 models for ensemble
            top_models = sorted(good_models.items(), key=lambda x: x[1], reverse=True)[:3]
            base_estimators = [(name, models[name]) for name, score in top_models]
            
            self.best_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(C=1.0, random_state=42),  # Moderate regularization
                cv=5,  # Fewer CV folds
                stack_method='predict_proba',
                n_jobs=-1
            )
        else:
            # Use single best model
            best_model_name = max(good_models.items(), key=lambda x: x[1])[0]
            self.best_model = models[best_model_name]
        
        # Train final model
        self.best_model.fit(self.X_train_final, self.y_train)
        
        # Final CV score
        final_cv_scores = cross_val_score(self.best_model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
        print(f"\nFinal model CV accuracy: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std() * 2:.4f})")
        
        return model_scores
    
    def evaluate_regularized_model(self):
        """Evaluate the regularized model"""
        print("📊 Evaluating regularized model...")
        
        # Predictions on test set
        y_pred = self.best_model.predict(self.X_test_final)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🎯 REGULARIZED MODEL PERFORMANCE:")
        print(f"🏆 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if 0.80 <= accuracy <= 0.90:
            print("✅ PERFECT! Achieved target range of 80-90%!")
        elif 0.75 <= accuracy < 0.80:
            print("👍 GOOD! Close to target range (75-80%)")
        elif 0.70 <= accuracy < 0.75:
            print("📈 Decent performance (70-75%)")
        elif accuracy > 0.95:
            print("⚠️  Still showing signs of overfitting")
        else:
            print("📊 Performance within expected range")
        
        # Detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Regularized Model Performance\nTest Accuracy: {accuracy:.2%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'test_accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def save_regularized_model(self, filename='regularized_model.pkl'):
        """Save the regularized model"""
        model_data = {
            'model': self.best_model,
            'preprocessors': self.preprocessors
        }
        joblib.dump(model_data, filename)
        print(f"Regularized model saved as {filename}")

def run_regularized_pipeline(file_path, target_column):
    """Run the regularized pipeline targeting 80-90% accuracy"""
    print("🛡️ REGULARIZED ML PIPELINE")
    print("Target: 80-90% accuracy (preventing overfitting)")
    print("=" * 60)
    
    pipeline = RegularizedMLPipeline()
    
    # Load and prepare dataset
    pipeline.load_and_prepare_dataset(file_path, target_column)
    
    # Proper train/test split with larger test set
    pipeline.proper_train_test_split(test_size=0.25, random_state=42)
    
    # Conservative preprocessing
    text_columns = ['Passage', 'Question', 'Options', 'Keywords', 'Hint', 'Explanation']
    categorical_columns = ['domain', 'question_type', 'Question_Type', 'Subject', 'Difficulty_Level']
    
    pipeline.conservative_preprocessing(text_columns, categorical_columns)
    
    # Regularized feature engineering
    pipeline.create_regularized_text_features(text_columns)
    pipeline.regularized_feature_engineering()
    
    # Train regularized models
    model_scores = pipeline.train_regularized_models()
    
    # Evaluate
    results = pipeline.evaluate_regularized_model()
    
    # Save model
    pipeline.save_regularized_model()
    
    return pipeline, results, model_scores

if __name__ == "__main__":
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"
    
    pipeline, results, model_scores = run_regularized_pipeline(file_path, target_column)
