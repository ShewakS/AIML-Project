"""
Ultra-Optimized ML Pipeline for 80-90% Accuracy
Advanced ensemble methods, data augmentation, and specialized techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   StratifiedKFold, RandomizedSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier, 
                            ExtraTreesClassifier, GradientBoostingClassifier,
                            VotingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.compose import ColumnTransformer
from scipy.stats import uniform, randint
import joblib

class UltraOptimizedPipeline:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ultra_model = None
        self.label_encoders = {}
        self.feature_processors = {}
        self.scaler = RobustScaler()
        
    def load_and_ultra_enhance_dataset(self, file_path, target_column):
        """Ultra-enhanced data loading with advanced quality improvements"""
        print("🚀 Ultra-enhancing dataset for maximum accuracy...")
        
        self.data = pd.read_csv(file_path)
        self.target_column = target_column
        
        # Advanced duplicate removal
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates(subset=['Passage', 'Question'])
        print(f"Removed {initial_size - len(self.data)} duplicates")
        
        # Add ultra-advanced features
        self.data = self._add_ultra_features(self.data)
        
        # Data augmentation for minority classes
        self.data = self._augment_data(self.data)
        
        # Advanced outlier detection and removal
        self.data = self._remove_outliers(self.data)
        
        print(f"Ultra-enhanced dataset shape: {self.data.shape}")
        return self.data
    
    def _add_ultra_features(self, df):
        """Add ultra-sophisticated features"""
        print("🎯 Adding ultra-advanced features...")
        
        # Advanced text statistics
        df['passage_complexity'] = df['Passage'].apply(self._calculate_complexity)
        df['question_complexity'] = df['Question'].apply(self._calculate_complexity)
        
        # Semantic similarity features
        df['question_passage_overlap'] = df.apply(self._calculate_overlap, axis=1)
        
        # Advanced question type classification
        df = self._classify_question_types(df)
        
        # Subject-specific features
        df = self._add_subject_features(df)
        
        # Difficulty-based features
        df = self._add_difficulty_features(df)
        
        # Answer choice analysis
        df = self._analyze_answer_choices(df)
        
        return df
    
    def _calculate_complexity(self, text):
        """Calculate text complexity score"""
        if pd.isna(text):
            return 0
        
        text = str(text)
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        if len(words) == 0 or sentences == 0:
            return 0
        
        # Flesch-like complexity
        avg_sentence_length = len(words) / max(sentences, 1)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        complexity = avg_sentence_length * 0.5 + avg_word_length * 2
        return complexity
    
    def _calculate_overlap(self, row):
        """Calculate semantic overlap between question and passage"""
        passage_words = set(str(row['Passage']).lower().split())
        question_words = set(str(row['Question']).lower().split())
        
        if len(passage_words) == 0 or len(question_words) == 0:
            return 0
        
        overlap = len(passage_words.intersection(question_words))
        return overlap / len(question_words.union(passage_words))
    
    def _classify_question_types(self, df):
        """Advanced question type classification"""
        # Factual questions
        df['is_factual'] = df['Question'].str.contains(
            r'\b(what|when|where|who|which)\b', case=False, na=False
        ).astype(int)
        
        # Analytical questions
        df['is_analytical'] = df['Question'].str.contains(
            r'\b(why|how|analyze|compare|explain|describe)\b', case=False, na=False
        ).astype(int)
        
        # Inferential questions
        df['is_inferential'] = df['Question'].str.contains(
            r'\b(infer|suggest|imply|conclude|assume)\b', case=False, na=False
        ).astype(int)
        
        return df
    
    def _add_subject_features(self, df):
        """Add subject-specific features"""
        # STEM vs Non-STEM
        stem_subjects = ['Mathematics', 'Science', 'Computer Science']
        df['is_stem'] = df['Subject'].isin(stem_subjects).astype(int)
        
        # Subject complexity mapping
        subject_complexity = {
            'Mathematics': 5, 'Science': 4, 'Computer Science': 4,
            'History': 3, 'Geography': 3, 'English': 2, 'Economics': 3
        }
        df['subject_complexity'] = df['Subject'].map(subject_complexity).fillna(2)
        
        return df
    
    def _add_difficulty_features(self, df):
        """Add difficulty-based features"""
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df['difficulty_numeric'] = df['Difficulty_Level'].map(difficulty_map)
        
        # Interaction features
        df['marks_per_difficulty'] = df['Marks'] / df['difficulty_numeric']
        df['complexity_difficulty_ratio'] = df['passage_complexity'] / df['difficulty_numeric']
        
        return df
    
    def _analyze_answer_choices(self, df):
        """Analyze answer choice patterns"""
        # Extract individual options
        df['option_lengths'] = df['Options'].apply(self._get_option_lengths)
        df['option_length_variance'] = df['Options'].apply(self._get_option_variance)
        
        # Correct option position bias
        df['correct_option_position'] = df['Correct_Option']
        
        return df
    
    def _get_option_lengths(self, options_str):
        """Get average length of options"""
        if pd.isna(options_str):
            return 0
        
        # Extract options (assuming they're in quotes)
        import re
        options = re.findall(r"'([^']*)'", str(options_str))
        if not options:
            return 0
        
        return np.mean([len(opt) for opt in options])
    
    def _get_option_variance(self, options_str):
        """Get variance in option lengths"""
        if pd.isna(options_str):
            return 0
        
        import re
        options = re.findall(r"'([^']*)'", str(options_str))
        if len(options) < 2:
            return 0
        
        lengths = [len(opt) for opt in options]
        return np.var(lengths)
    
    def _augment_data(self, df):
        """Data augmentation for minority classes"""
        print("📈 Augmenting data for better balance...")
        
        target_counts = df[self.target_column].value_counts()
        max_count = target_counts.max()
        
        augmented_dfs = [df]
        
        for target_value in target_counts.index:
            class_df = df[df[self.target_column] == target_value]
            current_count = len(class_df)
            
            if current_count < max_count * 0.8:  # If significantly underrepresented
                # Simple augmentation by adding noise to numerical features
                augment_count = min(int(max_count * 0.8) - current_count, current_count)
                
                augmented_samples = class_df.sample(n=augment_count, replace=True, random_state=42).copy()
                
                # Add small noise to numerical features
                numerical_cols = augmented_samples.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    if col != self.target_column:
                        noise = np.random.normal(0, augmented_samples[col].std() * 0.1, len(augmented_samples))
                        augmented_samples[col] += noise
                
                augmented_dfs.append(augmented_samples)
        
        result_df = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Augmented dataset from {len(df)} to {len(result_df)} samples")
        
        return result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def _remove_outliers(self, df):
        """Remove outliers using IQR method"""
        print("🧹 Removing outliers...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_column]
        
        initial_size = len(df)
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Removed {initial_size - len(df)} outliers")
        return df
    
    def ultra_preprocessing(self, text_columns, categorical_columns):
        """Ultra-advanced preprocessing"""
        print("⚡ Ultra-advanced preprocessing...")
        
        # Advanced text cleaning
        for col in text_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(self._ultra_text_cleaning)
        
        # Advanced categorical encoding
        for col in categorical_columns:
            if col in self.data.columns:
                # Target encoding for high-impact categoricals
                if col in ['Subject', 'Difficulty_Level']:
                    target_mean = self.data.groupby(col)[self.target_column].mean()
                    self.data[f'{col}_target_encoded'] = self.data[col].map(target_mean)
                
                # Standard label encoding
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle target
        if self.data[self.target_column].dtype == 'object':
            le_target = LabelEncoder()
            self.data[self.target_column] = le_target.fit_transform(self.data[self.target_column])
            self.label_encoders[self.target_column] = le_target
    
    def _ultra_text_cleaning(self, text):
        """Ultra-sophisticated text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Preserve important punctuation and structure
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\"\']', ' ', text)
        text = text.lower().strip()
        
        return text
    
    def create_ultra_features(self, text_columns):
        """Create ultra-advanced text features"""
        print("🎯 Creating ultra-advanced text features...")
        
        combined_text = self.data[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Multiple TF-IDF configurations
        tfidf_configs = [
            {'max_features': 2000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.9},
            {'max_features': 1500, 'ngram_range': (1, 3), 'min_df': 3, 'max_df': 0.8},
            {'max_features': 1000, 'ngram_range': (2, 3), 'min_df': 2, 'max_df': 0.85}
        ]
        
        text_features_list = []
        feature_names_list = []
        
        for i, config in enumerate(tfidf_configs):
            vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, **config)
            features = vectorizer.fit_transform(combined_text)
            text_features_list.append(features)
            feature_names_list.extend([f'tfidf_{i}_{j}' for j in range(features.shape[1])])
            self.feature_processors[f'tfidf_{i}'] = vectorizer
        
        # Count vectorizer
        count_vectorizer = CountVectorizer(
            max_features=800, ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words='english'
        )
        count_features = count_vectorizer.fit_transform(combined_text)
        text_features_list.append(count_features)
        feature_names_list.extend([f'count_{j}' for j in range(count_features.shape[1])])
        self.feature_processors['count'] = count_vectorizer
        
        # Combine all text features
        from scipy.sparse import hstack
        all_text_features = hstack(text_features_list)
        
        # Convert to DataFrame
        text_df = pd.DataFrame(all_text_features.toarray(), columns=feature_names_list)
        
        # Combine with non-text features
        non_text_cols = [col for col in self.data.columns if col not in text_columns]
        self.data = pd.concat([
            self.data[non_text_cols].reset_index(drop=True), 
            text_df
        ], axis=1)
        
        print(f"Created {all_text_features.shape[1]} ultra-advanced text features")
    
    def ultra_feature_engineering(self):
        """Ultra-advanced feature engineering and selection"""
        print("🔬 Ultra-advanced feature engineering...")
        
        # Prepare data
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.15, random_state=42, stratify=self.y  # Smaller test set for more training data
        )
        
        # Multi-stage feature selection
        print("🎯 Multi-stage feature selection...")
        
        # Stage 1: Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=0.01)
        X_train_var = variance_selector.fit_transform(self.X_train)
        X_test_var = variance_selector.transform(self.X_test)
        
        # Stage 2: Mutual information selection
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(800, X_train_var.shape[1]))
        X_train_mi = mi_selector.fit_transform(X_train_var, self.y_train)
        X_test_mi = mi_selector.transform(X_test_var)
        
        # Stage 3: Recursive feature elimination with Random Forest
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe_selector = RFE(rf_selector, n_features_to_select=min(200, X_train_mi.shape[1]), step=0.1)
        X_train_rfe = rfe_selector.fit_transform(X_train_mi, self.y_train)
        X_test_rfe = rfe_selector.transform(X_test_mi)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_rfe)
        X_test_scaled = self.scaler.transform(X_test_rfe)
        
        # Final dimensionality reduction
        svd = TruncatedSVD(n_components=min(100, X_train_scaled.shape[1] - 1), random_state=42)
        self.X_train_final = svd.fit_transform(X_train_scaled)
        self.X_test_final = svd.transform(X_test_scaled)
        
        # Store selectors
        self.feature_processors.update({
            'variance_selector': variance_selector,
            'mi_selector': mi_selector,
            'rfe_selector': rfe_selector,
            'svd': svd
        })
        
        print(f"Final ultra-engineered features: {self.X_train_final.shape[1]}")
        print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    
    def train_ultra_ensemble(self):
        """Train ultra-optimized ensemble"""
        print("🚀 Training ultra-optimized ensemble...")
        
        # Define diverse base models
        base_models = {
            'rf': RandomForestClassifier(
                n_estimators=500, max_depth=20, min_samples_split=2, 
                min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'et': ExtraTreesClassifier(
                n_estimators=500, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=7,
                subsample=0.9, random_state=42
            ),
            'svm': SVC(
                C=10, kernel='rbf', gamma='scale', probability=True, random_state=42
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=7, weights='distance', metric='minkowski'
            ),
            'nb': MultinomialNB(alpha=0.1),
            'lr': LogisticRegression(
                C=1, solver='liblinear', penalty='l2', random_state=42, max_iter=1000
            )
        }
        
        # Train individual models
        trained_models = {}
        individual_scores = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(self.X_train_final, self.y_train)
            
            # Cross-validation score
            cv_score = cross_val_score(model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
            individual_scores[name] = cv_score.mean()
            trained_models[name] = model
            
            print(f"{name} CV accuracy: {cv_score.mean():.4f}")
        
        # Select best performing models for ensemble
        sorted_models = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = [name for name, score in sorted_models[:5]]  # Top 5 models
        
        print(f"Selected top models: {top_models}")
        
        # Create multiple ensemble strategies
        
        # Strategy 1: Stacking with top models
        stacking_estimators = [(name, trained_models[name]) for name in top_models]
        stacking_model = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=cv,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Strategy 2: Voting classifier
        voting_model = VotingClassifier(
            estimators=stacking_estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Strategy 3: Bagging the best model
        best_model_name = sorted_models[0][0]
        bagging_model = BaggingClassifier(
            base_estimator=trained_models[best_model_name],
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train ensemble strategies
        ensemble_models = {
            'stacking': stacking_model,
            'voting': voting_model,
            'bagging': bagging_model
        }
        
        ensemble_scores = {}
        for name, model in ensemble_models.items():
            print(f"Training {name} ensemble...")
            model.fit(self.X_train_final, self.y_train)
            cv_score = cross_val_score(model, self.X_train_final, self.y_train, cv=cv, scoring='accuracy')
            ensemble_scores[name] = cv_score.mean()
            print(f"{name} CV accuracy: {cv_score.mean():.4f}")
        
        # Select best ensemble
        best_ensemble = max(ensemble_scores.items(), key=lambda x: x[1])
        self.ultra_model = ensemble_models[best_ensemble[0]]
        
        print(f"Selected best ensemble: {best_ensemble[0]} with CV accuracy: {best_ensemble[1]:.4f}")
        
        return trained_models, ensemble_scores
    
    def ultra_evaluation(self):
        """Ultra-comprehensive evaluation"""
        print("📊 Ultra-comprehensive evaluation...")
        
        # Predictions
        y_pred = self.ultra_model.predict(self.X_test_final)
        y_pred_proba = self.ultra_model.predict_proba(self.X_test_final)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"🎯 ULTRA-OPTIMIZED MODEL PERFORMANCE:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Ultra-Optimized Model Confusion Matrix\nAccuracy: {accuracy:.2%}')
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
    
    def save_ultra_model(self, filename='ultra_optimized_model.pkl'):
        """Save the ultra-optimized model"""
        model_data = {
            'model': self.ultra_model,
            'scaler': self.scaler,
            'feature_processors': self.feature_processors,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_data, filename)
        print(f"Ultra-optimized model saved as {filename}")

def run_ultra_pipeline(file_path, target_column):
    """Run the ultra-optimized pipeline"""
    print("🚀 ULTRA-OPTIMIZED ML PIPELINE FOR 80-90% ACCURACY")
    print("=" * 70)
    
    pipeline = UltraOptimizedPipeline()
    
    # Ultra-enhanced data loading
    pipeline.load_and_ultra_enhance_dataset(file_path, target_column)
    
    # Ultra preprocessing
    text_columns = ['Passage', 'Question', 'Options', 'Keywords', 'Hint', 'Explanation']
    categorical_columns = ['domain', 'question_type', 'Question_Type', 'Subject', 'Difficulty_Level']
    
    pipeline.ultra_preprocessing(text_columns, categorical_columns)
    
    # Ultra feature engineering
    pipeline.create_ultra_features(text_columns)
    pipeline.ultra_feature_engineering()
    
    # Ultra ensemble training
    individual_models, ensemble_scores = pipeline.train_ultra_ensemble()
    
    # Ultra evaluation
    results = pipeline.ultra_evaluation()
    
    # Save model
    pipeline.save_ultra_model()
    
    return pipeline, results, individual_models, ensemble_scores

if __name__ == "__main__":
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"
    
    pipeline, results, individual_models, ensemble_scores = run_ultra_pipeline(file_path, target_column)
