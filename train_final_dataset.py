"""
Training Script for final_modified_dataset.csv
Reading Comprehension Dataset with Stacking Ensemble
"""

from stacking_ml_pipeline import MLPipeline, run_complete_pipeline
import pandas as pd
import numpy as np

def train_reading_comprehension_model():
    """
    Train the stacking ensemble model on the reading comprehension dataset
    """
    print("🚀 Starting Training Process for Reading Comprehension Dataset")
    print("=" * 70)
    
    # Dataset configuration
    file_path = "c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv"
    target_column = "Correct_Option"  # This appears to be the target (correct answer)
    
    # Text columns that contain meaningful text for TF-IDF
    text_columns = [
        "Passage",      # Main text passages
        "Question",     # Questions text
        "Options",      # Multiple choice options
        "Keywords",     # Keywords
        "Hint",         # Hints
        "Explanation"   # Explanations
    ]
    
    # Categorical columns for encoding
    categorical_columns = [
        "domain",           # Domain type
        "question_type",    # Type of question
        "Question_Type",    # MCQ/Descriptive
        "Subject",          # Subject area
        "Difficulty_Level"  # Easy/Medium/Hard
    ]
    
    print(f"📁 Dataset: {file_path}")
    print(f"🎯 Target Column: {target_column}")
    print(f"📝 Text Columns: {text_columns}")
    print(f"🏷️ Categorical Columns: {categorical_columns}")
    print()
    
    try:
        # Run the complete pipeline
        pipeline, results = run_complete_pipeline(
            file_path=file_path,
            target_column=target_column,
            text_columns=text_columns,
            categorical_columns=categorical_columns
        )
        
        if results:
            print("\n" + "🎉 TRAINING COMPLETED SUCCESSFULLY!" + "\n")
            print("📊 FINAL RESULTS:")
            print("-" * 40)
            print(f"✅ Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"✅ Precision: {results['precision']:.4f}")
            print(f"✅ Recall: {results['recall']:.4f}")
            print(f"✅ F1-Score: {results['f1_score']:.4f}")
            
            print("\n📈 MODEL COMPARISON:")
            print("-" * 40)
            for model_name, accuracy in results['models_comparison']:
                print(f"• {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Save the trained model
            model_filename = "reading_comprehension_stacking_model.pkl"
            pipeline.save_model(model_filename)
            print(f"\n💾 Model saved as: {model_filename}")
            
            return pipeline, results
        else:
            print("❌ Training failed - no results returned")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the dataset file exists")
        print("2. Verify column names match the dataset")
        print("3. Ensure sufficient memory for processing")
        return None, None

def analyze_dataset_first():
    """
    Quick analysis of the dataset before training
    """
    print("🔍 DATASET ANALYSIS")
    print("=" * 50)
    
    try:
        # Load dataset for quick analysis
        df = pd.read_csv("c:/Users/shewa/Desktop/aiml/final_modified_dataset.csv")
        
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"📝 Total Questions: {df.shape[0]}")
        print(f"🏷️ Total Features: {df.shape[1]}")
        
        print(f"\n🎯 Target Variable (Correct_Option) Distribution:")
        target_dist = df['Correct_Option'].value_counts().sort_index()
        for option, count in target_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   Option {option}: {count} ({percentage:.1f}%)")
        
        print(f"\n📚 Subject Distribution:")
        subject_dist = df['Subject'].value_counts()
        for subject, count in subject_dist.head().items():
            percentage = (count / len(df)) * 100
            print(f"   {subject}: {count} ({percentage:.1f}%)")
        
        print(f"\n⚡ Difficulty Level Distribution:")
        diff_dist = df['Difficulty_Level'].value_counts()
        for level, count in diff_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   {level}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔤 Question Type Distribution:")
        qtype_dist = df['Question_Type'].value_counts()
        for qtype, count in qtype_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   {qtype}: {count} ({percentage:.1f}%)")
        
        print(f"\n📖 Domain Distribution:")
        domain_dist = df['domain'].value_counts()
        for domain, count in domain_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   {domain}: {count} ({percentage:.1f}%)")
        
        # Check for missing values
        print(f"\n❓ Missing Values:")
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                print(f"   {col}: {count} ({percentage:.1f}%)")
        else:
            print("   ✅ No missing values found!")
        
        print(f"\n🔄 Duplicate Rows: {df.duplicated().sum()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None

if __name__ == "__main__":
    print("🤖 READING COMPREHENSION ML PIPELINE")
    print("🎯 Stacking Ensemble: Random Forest + Logistic Regression")
    print("=" * 70)
    
    # First analyze the dataset
    dataset = analyze_dataset_first()
    
    if dataset is not None:
        print(f"\n{'='*70}")
        print("📋 Starting training automatically...")
        
        # Start training
        pipeline, results = train_reading_comprehension_model()
        
        if pipeline and results:
            print(f"\n{'='*70}")
            print("🎊 SUCCESS! Your reading comprehension model is ready!")
            print("📁 Files created:")
            print("   • reading_comprehension_stacking_model.pkl (trained model)")
            print("   • All preprocessing components saved")
            print(f"\n🚀 You can now use this model to predict correct answers for new questions!")
        else:
            print("\n❌ Training process failed. Please check the error messages above.")
    else:
        print("\n❌ Could not analyze dataset. Please check the file path and format.")
