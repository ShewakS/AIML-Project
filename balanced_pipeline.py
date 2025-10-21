"""
Balanced ML Pipeline - Target: 80-90% Accuracy
Optimal balance between performance and preventing overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

class BalancedMLPipeline:
    """Balanced ML pipeline - good performance without overfitting"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.preprocessors = {}
        
    def load_and_enhance_dataset(self, file_path, target_column):
        """Load dataset with balanced feature engineering"""
        print("📁 Loading dataset with balanced approach...")
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        
        # Remove duplicates
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['Passage', 'Question'])
        print(f"Removed {initial_size - len(self.data)} duplicates")
        
        # Balanced feature engineering
        self.data = self._create_balanced_features(self.data)
        
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def _create_balanced_features(self, df):
        """Create balanced features - effective but not overfitting"""
        print("🔧 Creating balanced features...")
        
        # Essential text features
        df['passage_length'] = df['Passage'].str.len()
        df['question_length'] = df['Question'].str.len()
        df['passage_word_count'] = df['Passage'].str.split().str.len()
        df['question_word_count'] = df['Question'].str.split().str.len()
        df['passage_sentence_count'] = df['Passage'].str.count(r'[.!?]+')
        
        # Question type features (proven effective)
        df['is_what_question'] = df['Question'].str.contains(r'\bwhat\b', case=False, na=False).astype(int)
        df['is_who_question'] = df['Question'].str.contains(r'\bwho\b', case=False, na=False).astype(int)
        df['is_where_question'] = df['Question'].str.contains(r'\bwhere\b', case=False, na=False).astype(int)
        df['is_when_question'] = df['Question'].str.contains(r'\bwhen\b', case=False, na=False).astype(int)
        df['is_why_question'] = df['Question'].str.contains(r'\bwhy\b', case=False, na=False).astype(int)
        df['is_how_question'] = df['Question'].str.contains(r'\bhow\b', case=False, na=False).astype(int)
        
        # Advanced question types
        df['is_factual'] = df['Question'].str.contains(
            r'\b(what|when|where|who|which)\b', case=False, na=False
        ).astype(int)
        df['is_analytical'] = df['Question'].str.contains(
            r'\b(why|how|analyze|compare|explain)\b', case=False, na=False
        ).astype(int)
        
        # Subject and difficulty features
        subject_complexity = {
            'Mathematics': 4, 'Science': 3, 'Computer Science': 4,
            'History': 2, 'Geography': 2, 'English': 1, 'Economics': 3
        }
        df['subject_complexity'] = df['Subject'].map(subject_complexity).fillna(2)
        
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df['difficulty_numeric'] = df['Difficulty_Level'].map(difficulty_map)
        
        # STEM classification
        stem_subjects = ['Mathematics', 'Science', 'Computer Science']
        df['is_stem'] = df['Subject'].isin(stem_subjects).astype(int)
        
        # Semantic features
        df['question_passage_overlap'] = df.apply(self._calculate_word_overlap, axis=1)
        df['keyword_match'] = df.apply(self._keyword_passage_match, axis=1)
        
        # Interaction features (limited to prevent overfitting)
        df['length_ratio'] = df['question_length'] / (df['passage_length'] + 1)
        df['word_ratio'] = df['question_word_count'] / (df['passage_word_count'] + 1)
        
        return df
    
    def _calculate_word_overlap(self, row):
        """Calculate word overlap between question and passage"""
        passage_words = set(str(row['Passage']).lower().split())
        question_words = set(str(row['Question']).lower().split())
        
        if len(question_words) == 0:
            return 0
        
        overlap = len(passage_words.intersection(question_words))
        return overlap / len(question_words)
    
    def _keyword_passage_match(self, row):
        """Check if keywords appear in passage"""
        if pd.isna(row['Keywords']):
            return 0
        
        keywords = str(row['Keywords']).lower().split(',')
        passage = str(row['Passage']).lower()
        
        matches = sum(1 for keyword in keywords if keyword.strip() in passage)
        return matches / len(keywords) if keywords else 0
    
    def proper_train_test_split(self, test_size=0.2, random_state=42):
        """Standard train/test split"""
        print("✂️ Train/test split...")
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train_raw.shape[0]} samples")
        print(f"Test set: {self.X_test_raw.shape[0]} samples")
        
        return self.X_train_raw, self.X_test_raw, self.y_train, self.y_test
    
    def balanced_preprocessing(self, text_columns, categorical_columns):
        """Balanced preprocessing"""
        print("🔧 Balanced preprocessing...")
        
        # Text cleaning
        for col in text_columns:
            if col in self.X_train_raw.columns:
                self.X_train_raw[col] = self.X_train_raw[col].fillna('').astype(str)
                self.X_test_raw[col] = self.X_test_raw[col].fillna('').astype(str)
        
        # Categorical encoding
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
    
    def create_balanced_text_features(self, text_columns):
        """Create balanced text features"""
        print("📝 Creating balanced text features...")
        
        # Combine text
        train_combined_text = self.X_train_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        test_combined_text = self.X_test_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Balanced TF-IDF configuration
        tfidf = TfidfVectorizer(
            max_features=1500,  # Moderate number of features
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
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
        
        print(f"✅ Created {train_tfidf_features.shape[1]} balanced text features")
    
    def balanced_feature_engineering(self):
        """Balanced feature engineering"""
        print("🔬 Balanced feature engineering...")
        
        # Feature selection - moderate number
        selector = SelectKBest(score_func=mutual_info_classif, k=min(200, self.X_train.shape[1]))
        X_train_selected = selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = selector.transform(self.X_test)
        self.preprocessors['feature_selector'] = selector
        
        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.preprocessors['scaler'] = scaler
        
        # Moderate dimensionality reduction
        svd = TruncatedSVD(n_components=min(80, X_train_scaled.shape[1] - 1), random_state=42)
        self.X_train_final = svd.fit_transform(X_train_scaled)
        self.X_test_final = svd.transform(X_test_scaled)
        self.preprocessors['svd'] = svd
        
        print(f"✅ Final features: {self.X_train_final.shape[1]}")
        print(f"✅ Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    def train_balanced_ensemble(self):
        """Train balanced ensemble with moderate regularization"""
        print("🚀 Training balanced ensemble...")
        
        # Models with balanced regularization
        models = {
            'rf_balanced': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'et_balanced': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gb_balanced': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                random_state=42
            ),
            'svm_balanced': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'lr_balanced': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            )
        }
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_scores = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
            model_scores[name] = cv_scores.mean()
            std_score = cv_scores.std()
            print(f"{name}: {cv_scores.mean():.4f} (+/- {std_score * 2:.4f})")
        
        # Select top 3 models for ensemble
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:3]
        
        print(f"\nTop models selected for ensemble:")
        for name, score in top_models:
            print(f"  {name}: {score:.4f}")
        
        # Create ensemble
        base_estimators = [(name, models[name]) for name, score in top_models]
        
        self.best_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1.0, random_state=42),
            cv=cv,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train final model
        self.best_model.fit(self.X_train_final, self.y_train)
        
        # Final CV score
        final_cv_scores = cross_val_score(self.best_model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
        print(f"\nFinal ensemble CV accuracy: {final_cv_scores.mean():.4f} (+/- {final_cv_scores.std() * 2:.4f})")
        
        return model_scores
    
    def evaluate_balanced_model(self):
        """Evaluate the balanced model"""
        print("📊 Evaluating balanced model...")
        
        # Predictions on test set
        y_pred = self.best_model.predict(self.X_test_final)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🎯 BALANCED MODEL PERFORMANCE:")
        print(f"🏆 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if 0.80 <= accuracy <= 0.90:
            print("🎉 EXCELLENT! Achieved target range of 80-90%!")
        elif 0.75 <= accuracy < 0.80:
            print("👍 VERY GOOD! Close to target range (75-80%)")
        elif 0.70 <= accuracy < 0.75:
            print("📈 GOOD! Solid performance (70-75%)")
        elif accuracy > 0.95:
            print("⚠️  Potential overfitting detected")
        else:
            print("📊 Reasonable performance for this dataset")
        
        # Detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Balanced Model Performance\nTest Accuracy: {accuracy:.2%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'test_accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def save_balanced_model(self, filename='balanced_model.pkl'):
        """Save the balanced model"""
        model_data = {
            'model': self.best_model,
            'preprocessors': self.preprocessors
        }
        joblib.dump(model_data, filename)
        print(f"Balanced model saved as {filename}")

def run_balanced_pipeline(file_path, target_column):
    """Run the balanced pipeline targeting 80-90% accuracy"""
    print("⚖️ BALANCED ML PIPELINE")
    print("Target: 80-90% accuracy with balanced approach")
    print("=" * 60)
    
    pipeline = BalancedMLPipeline()
    
    # Load and enhance dataset
    pipeline.load_and_enhance_dataset(file_path, target_column)
    
    # Proper train/test split
    pipeline.proper_train_test_split(test_size=0.2, random_state=42)
    
    # Balanced preprocessing
    text_columns = ['Passage', 'Question', 'Options', 'Keywords', 'Hint', 'Explanation']
    categorical_columns = ['domain', 'question_type', 'Question_Type', 'Subject', 'Difficulty_Level']
    
    pipeline.balanced_preprocessing(text_columns, categorical_columns)
    
    # Balanced feature engineering
    pipeline.create_balanced_text_features(text_columns)
    pipeline.balanced_feature_engineering()
    
    # Train balanced ensemble
    model_scores = pipeline.train_balanced_ensemble()
    
    # Evaluate
    results = pipeline.evaluate_balanced_model()
    
    # Save model
    pipeline.save_balanced_model()
    
    return pipeline, results, model_scores

if __name__ == "__main__":
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"
    
    pipeline, results, model_scores = run_balanced_pipeline(file_path, target_column)
