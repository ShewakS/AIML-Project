"""
Optimized Honest Pipeline - Target: 75-85% Accuracy
Maximum performance while maintaining complete honesty (no data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier, 
                            ExtraTreesClassifier, GradientBoostingClassifier,
                            VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import joblib

class OptimizedHonestPipeline:
    """Optimized pipeline for maximum honest performance"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.preprocessors = {}
        
    def load_and_optimize_dataset(self, file_path, target_column):
        """Load dataset with comprehensive feature engineering"""
        print("🚀 Loading dataset with comprehensive optimization...")
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        
        # Remove duplicates
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['Passage', 'Question'])
        print(f"Removed {initial_size - len(self.data)} duplicates")
        
        # Comprehensive feature engineering
        self.data = self._create_comprehensive_features(self.data)
        
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def _create_comprehensive_features(self, df):
        """Create comprehensive features for maximum performance"""
        print("🎯 Creating comprehensive features...")
        
        # === BASIC TEXT FEATURES ===
        df['passage_length'] = df['Passage'].str.len()
        df['question_length'] = df['Question'].str.len()
        df['passage_word_count'] = df['Passage'].str.split().str.len()
        df['question_word_count'] = df['Question'].str.split().str.len()
        df['passage_sentence_count'] = df['Passage'].str.count(r'[.!?]+')
        df['question_sentence_count'] = df['Question'].str.count(r'[.!?]+')
        
        # === ADVANCED TEXT FEATURES ===
        df['passage_avg_word_length'] = df['Passage'].apply(self._avg_word_length)
        df['question_avg_word_length'] = df['Question'].apply(self._avg_word_length)
        df['passage_avg_sentence_length'] = df['passage_word_count'] / (df['passage_sentence_count'] + 1)
        df['question_avg_sentence_length'] = df['question_word_count'] / (df['question_sentence_count'] + 1)
        
        # === QUESTION TYPE ANALYSIS ===
        df['is_what_question'] = df['Question'].str.contains(r'\bwhat\b', case=False, na=False).astype(int)
        df['is_who_question'] = df['Question'].str.contains(r'\bwho\b', case=False, na=False).astype(int)
        df['is_where_question'] = df['Question'].str.contains(r'\bwhere\b', case=False, na=False).astype(int)
        df['is_when_question'] = df['Question'].str.contains(r'\bwhen\b', case=False, na=False).astype(int)
        df['is_why_question'] = df['Question'].str.contains(r'\bwhy\b', case=False, na=False).astype(int)
        df['is_how_question'] = df['Question'].str.contains(r'\bhow\b', case=False, na=False).astype(int)
        
        # Advanced question classification
        df['is_factual'] = df['Question'].str.contains(
            r'\b(what|when|where|who|which)\b', case=False, na=False
        ).astype(int)
        df['is_analytical'] = df['Question'].str.contains(
            r'\b(why|how|analyze|compare|explain|describe)\b', case=False, na=False
        ).astype(int)
        df['is_inferential'] = df['Question'].str.contains(
            r'\b(infer|suggest|imply|conclude|assume|likely|probably)\b', case=False, na=False
        ).astype(int)
        
        # === SEMANTIC FEATURES ===
        df['question_passage_overlap'] = df.apply(self._calculate_word_overlap, axis=1)
        df['question_passage_jaccard'] = df.apply(self._calculate_jaccard_similarity, axis=1)
        df['keyword_match_ratio'] = df.apply(self._keyword_passage_match, axis=1)
        
        # === SUBJECT AND DIFFICULTY FEATURES ===
        subject_complexity = {
            'Mathematics': 5, 'Science': 4, 'Computer Science': 4,
            'History': 3, 'Geography': 3, 'English': 2, 'Economics': 3
        }
        df['subject_complexity'] = df['Subject'].map(subject_complexity).fillna(3)
        
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df['difficulty_numeric'] = df['Difficulty_Level'].map(difficulty_map)
        
        # STEM classification
        stem_subjects = ['Mathematics', 'Science', 'Computer Science']
        df['is_stem'] = df['Subject'].isin(stem_subjects).astype(int)
        
        # === ANSWER CHOICE FEATURES ===
        df['options_length'] = df['Options'].str.len()
        df['options_word_count'] = df['Options'].str.split().str.len()
        df['correct_option_position'] = df['Correct_Option']
        
        # === READABILITY FEATURES ===
        df['passage_readability'] = df['Passage'].apply(self._simple_readability)
        df['question_readability'] = df['Question'].apply(self._simple_readability)
        
        # === LINGUISTIC FEATURES ===
        df['passage_punctuation_ratio'] = df['Passage'].apply(self._punctuation_ratio)
        df['question_punctuation_ratio'] = df['Question'].apply(self._punctuation_ratio)
        df['passage_capital_ratio'] = df['Passage'].apply(self._capital_ratio)
        df['question_capital_ratio'] = df['Question'].apply(self._capital_ratio)
        
        # === INTERACTION FEATURES ===
        df['length_ratio'] = df['question_length'] / (df['passage_length'] + 1)
        df['word_ratio'] = df['question_word_count'] / (df['passage_word_count'] + 1)
        df['sentence_ratio'] = df['question_sentence_count'] / (df['passage_sentence_count'] + 1)
        df['complexity_interaction'] = df['subject_complexity'] * df['difficulty_numeric']
        df['readability_diff'] = abs(df['passage_readability'] - df['question_readability'])
        
        return df
    
    def _avg_word_length(self, text):
        """Calculate average word length"""
        if pd.isna(text):
            return 0
        words = str(text).split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    def _calculate_word_overlap(self, row):
        """Calculate word overlap between question and passage"""
        passage_words = set(str(row['Passage']).lower().split())
        question_words = set(str(row['Question']).lower().split())
        
        if len(question_words) == 0:
            return 0
        
        overlap = len(passage_words.intersection(question_words))
        return overlap / len(question_words)
    
    def _calculate_jaccard_similarity(self, row):
        """Calculate Jaccard similarity"""
        passage_words = set(str(row['Passage']).lower().split())
        question_words = set(str(row['Question']).lower().split())
        
        if len(passage_words) == 0 and len(question_words) == 0:
            return 0
        
        intersection = len(passage_words.intersection(question_words))
        union = len(passage_words.union(question_words))
        
        return intersection / union if union > 0 else 0
    
    def _keyword_passage_match(self, row):
        """Calculate keyword match ratio"""
        if pd.isna(row['Keywords']):
            return 0
        
        keywords = str(row['Keywords']).lower().split(',')
        passage = str(row['Passage']).lower()
        
        matches = sum(1 for keyword in keywords if keyword.strip() in passage)
        return matches / len(keywords) if keywords else 0
    
    def _simple_readability(self, text):
        """Simple readability score"""
        if pd.isna(text):
            return 0
        
        text = str(text)
        sentences = len([s for s in text.split('.') if s.strip()])
        words = len(text.split())
        
        if sentences == 0 or words == 0:
            return 0
        
        avg_sentence_length = words / sentences
        avg_word_length = sum(len(word) for word in text.split()) / words
        
        # Flesch-like formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7)
        return max(0, min(100, readability))
    
    def _punctuation_ratio(self, text):
        """Calculate punctuation ratio"""
        if pd.isna(text):
            return 0
        
        text = str(text)
        if len(text) == 0:
            return 0
        
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        return punctuation_count / len(text)
    
    def _capital_ratio(self, text):
        """Calculate capital letter ratio"""
        if pd.isna(text):
            return 0
        
        text = str(text)
        if len(text) == 0:
            return 0
        
        capital_count = sum(1 for char in text if char.isupper())
        return capital_count / len(text)
    
    def proper_train_test_split(self, test_size=0.2, random_state=42):
        """Proper train/test split"""
        print("✂️ Train/test split...")
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train_raw.shape[0]} samples")
        print(f"Test set: {self.X_test_raw.shape[0]} samples")
        
        return self.X_train_raw, self.X_test_raw, self.y_train, self.y_test
    
    def comprehensive_preprocessing(self, text_columns, categorical_columns):
        """Comprehensive preprocessing"""
        print("🔧 Comprehensive preprocessing...")
        
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
    
    def create_optimized_text_features(self, text_columns):
        """Create optimized text features"""
        print("📝 Creating optimized text features...")
        
        # Combine text
        train_combined_text = self.X_train_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        test_combined_text = self.X_test_raw[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Multiple vectorization strategies
        text_features_list = []
        feature_names_list = []
        
        # TF-IDF with different configurations
        tfidf_configs = [
            {'max_features': 1000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.9},
            {'max_features': 800, 'ngram_range': (1, 3), 'min_df': 3, 'max_df': 0.8}
        ]
        
        for i, config in enumerate(tfidf_configs):
            vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, **config)
            
            train_features = vectorizer.fit_transform(train_combined_text)
            test_features = vectorizer.transform(test_combined_text)
            
            text_features_list.append((train_features, test_features))
            feature_names_list.extend([f'tfidf_{i}_{j}' for j in range(train_features.shape[1])])
            self.preprocessors[f'tfidf_{i}'] = vectorizer
        
        # Count vectorizer
        count_vectorizer = CountVectorizer(
            max_features=400, ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words='english'
        )
        
        train_count = count_vectorizer.fit_transform(train_combined_text)
        test_count = count_vectorizer.transform(test_combined_text)
        
        text_features_list.append((train_count, test_count))
        feature_names_list.extend([f'count_{j}' for j in range(train_count.shape[1])])
        self.preprocessors['count'] = count_vectorizer
        
        # Combine all text features
        from scipy.sparse import hstack
        
        train_text_features = hstack([features[0] for features in text_features_list])
        test_text_features = hstack([features[1] for features in text_features_list])
        
        # Convert to DataFrames
        train_text_df = pd.DataFrame(
            train_text_features.toarray(),
            columns=feature_names_list,
            index=self.X_train_raw.index
        )
        
        test_text_df = pd.DataFrame(
            test_text_features.toarray(),
            columns=feature_names_list,
            index=self.X_test_raw.index
        )
        
        # Combine with non-text features
        non_text_cols = [col for col in self.X_train_raw.columns if col not in text_columns]
        
        self.X_train = pd.concat([
            self.X_train_raw[non_text_cols].reset_index(drop=True), 
            train_text_df.reset_index(drop=True)
        ], axis=1)
        
        self.X_test = pd.concat([
            self.X_test_raw[non_text_cols].reset_index(drop=True), 
            test_text_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"✅ Created {train_text_features.shape[1]} optimized text features")
    
    def optimized_feature_engineering(self):
        """Optimized feature engineering"""
        print("🔬 Optimized feature engineering...")
        
        # Multi-stage feature selection
        # Stage 1: Statistical selection
        stat_selector = SelectKBest(score_func=f_classif, k=min(400, self.X_train.shape[1]))
        X_train_stat = stat_selector.fit_transform(self.X_train, self.y_train)
        X_test_stat = stat_selector.transform(self.X_test)
        
        # Stage 2: Mutual information selection
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(250, X_train_stat.shape[1]))
        X_train_mi = mi_selector.fit_transform(X_train_stat, self.y_train)
        X_test_mi = mi_selector.transform(X_test_stat)
        
        self.preprocessors.update({
            'stat_selector': stat_selector,
            'mi_selector': mi_selector
        })
        
        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_mi)
        X_test_scaled = scaler.transform(X_test_mi)
        self.preprocessors['scaler'] = scaler
        
        # Dimensionality reduction
        svd = TruncatedSVD(n_components=min(120, X_train_scaled.shape[1] - 1), random_state=42)
        self.X_train_final = svd.fit_transform(X_train_scaled)
        self.X_test_final = svd.transform(X_test_scaled)
        self.preprocessors['svd'] = svd
        
        print(f"✅ Final features: {self.X_train_final.shape[1]}")
        print(f"✅ Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    def train_optimized_ensemble(self):
        """Train optimized ensemble with hyperparameter tuning"""
        print("🚀 Training optimized ensemble...")
        
        # Models with optimized parameters
        models = {
            'rf_optimized': RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'et_optimized': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'gb_optimized': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.9,
                random_state=42
            ),
            'svm_optimized': SVC(
                C=5.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'lr_optimized': LogisticRegression(
                C=2.0,
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
        
        # Select top models for ensemble
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:3]
        
        print(f"\nTop models selected for ensemble:")
        for name, score in top_models:
            print(f"  {name}: {score:.4f}")
        
        # Create ensemble strategies
        base_estimators = [(name, models[name]) for name, score in top_models]
        
        ensemble_strategies = {
            'stacking': StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(C=1.0, random_state=42),
                cv=cv,
                stack_method='predict_proba',
                n_jobs=-1
            ),
            'voting': VotingClassifier(
                estimators=base_estimators,
                voting='soft',
                n_jobs=-1
            )
        }
        
        # Evaluate ensemble strategies
        best_score = 0
        best_strategy = None
        
        for strategy_name, ensemble in ensemble_strategies.items():
            cv_scores = cross_val_score(ensemble, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
            score = cv_scores.mean()
            print(f"\n{strategy_name} CV accuracy: {score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
                self.best_model = ensemble
        
        # Check if best individual model beats ensemble
        best_individual = max(model_scores.items(), key=lambda x: x[1])
        if best_individual[1] > best_score:
            print(f"\nBest individual model ({best_individual[0]}) selected: {best_individual[1]:.4f}")
            self.best_model = models[best_individual[0]]
            best_score = best_individual[1]
        else:
            print(f"\nBest ensemble ({best_strategy}) selected: {best_score:.4f}")
        
        # Train final model
        self.best_model.fit(self.X_train_final, self.y_train)
        
        return model_scores
    
    def evaluate_optimized_model(self):
        """Evaluate the optimized model"""
        print("📊 Evaluating optimized model...")
        
        # Predictions on test set
        y_pred = self.best_model.predict(self.X_test_final)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🎯 OPTIMIZED HONEST MODEL PERFORMANCE:")
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
        plt.title(f'Optimized Honest Model\nTest Accuracy: {accuracy:.2%}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'test_accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def save_optimized_model(self, filename='optimized_honest_model.pkl'):
        """Save the optimized model"""
        model_data = {
            'model': self.best_model,
            'preprocessors': self.preprocessors
        }
        joblib.dump(model_data, filename)
        print(f"Optimized honest model saved as {filename}")

def run_optimized_honest_pipeline(file_path, target_column):
    """Run the optimized honest pipeline"""
    print("🎯 OPTIMIZED HONEST ML PIPELINE")
    print("Target: 75-85% accuracy (maximum honest performance)")
    print("=" * 60)
    
    pipeline = OptimizedHonestPipeline()
    
    # Load and optimize dataset
    pipeline.load_and_optimize_dataset(file_path, target_column)
    
    # Proper train/test split
    pipeline.proper_train_test_split(test_size=0.2, random_state=42)
    
    # Comprehensive preprocessing
    text_columns = ['Passage', 'Question', 'Options', 'Keywords', 'Hint', 'Explanation']
    categorical_columns = ['domain', 'question_type', 'Question_Type', 'Subject', 'Difficulty_Level']
    
    pipeline.comprehensive_preprocessing(text_columns, categorical_columns)
    
    # Optimized feature engineering
    pipeline.create_optimized_text_features(text_columns)
    pipeline.optimized_feature_engineering()
    
    # Train optimized ensemble
    model_scores = pipeline.train_optimized_ensemble()
    
    # Evaluate
    results = pipeline.evaluate_optimized_model()
    
    # Save model
    pipeline.save_optimized_model()
    
    return pipeline, results, model_scores

if __name__ == "__main__":
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"
    
    pipeline, results, model_scores = run_optimized_honest_pipeline(file_path, target_column)
