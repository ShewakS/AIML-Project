"""
Example Usage of the Stacking ML Pipeline
This script demonstrates how to use the MLPipeline class with your dataset
"""

from stacking_ml_pipeline import MLPipeline, run_complete_pipeline
import pandas as pd
import numpy as np

def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes
    You can replace this with your actual dataset loading
    """
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    
    # Numerical features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(2, 1.5, n_samples)
    feature3 = np.random.uniform(-1, 1, n_samples)
    
    # Categorical features
    categories = ['A', 'B', 'C', 'D']
    cat_feature1 = np.random.choice(categories, n_samples)
    cat_feature2 = np.random.choice(['Type1', 'Type2', 'Type3'], n_samples)
    
    # Text features (simulated)
    text_samples = [
        "This is a positive example with good features",
        "Negative sample with poor quality",
        "Neutral text with average characteristics",
        "Excellent quality with outstanding features",
        "Bad example with terrible attributes"
    ]
    text_feature = np.random.choice(text_samples, n_samples)
    
    # Create target variable (binary classification)
    target = (feature1 + feature2 * 0.5 + feature3 * 0.3 + 
              np.random.normal(0, 0.5, n_samples) > 1).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'numerical_feature1': feature1,
        'numerical_feature2': feature2,
        'numerical_feature3': feature3,
        'categorical_feature1': cat_feature1,
        'categorical_feature2': cat_feature2,
        'text_feature': text_feature,
        'target': target
    })
    
    return data

def example_with_sample_data():
    """
    Example using generated sample data
    """
    print("Creating sample dataset...")
    data = create_sample_dataset()
    
    # Save sample data
    data.to_csv('sample_dataset.csv', index=False)
    print("Sample dataset saved as 'sample_dataset.csv'")
    
    # Run the complete pipeline
    pipeline, results = run_complete_pipeline(
        file_path='sample_dataset.csv',
        target_column='target',
        text_columns=['text_feature'],
        categorical_columns=['categorical_feature1', 'categorical_feature2']
    )
    
    return pipeline, results

def example_with_your_dataset():
    """
    Example template for using your own dataset
    Modify the parameters below to match your dataset
    """
    
    # MODIFY THESE PARAMETERS FOR YOUR DATASET
    file_path = "your_dataset.csv"  # Path to your dataset
    target_column = "target"        # Name of your target column
    text_columns = ["text_col1", "text_col2"]  # List of text columns (optional)
    categorical_columns = ["cat_col1", "cat_col2"]  # List of categorical columns (optional)
    
    # Run the pipeline
    try:
        pipeline, results = run_complete_pipeline(
            file_path=file_path,
            target_column=target_column,
            text_columns=text_columns,
            categorical_columns=categorical_columns
        )
        
        print("Pipeline completed successfully!")
        print(f"Final accuracy: {results['accuracy']:.4f}")
        
        return pipeline, results
        
    except FileNotFoundError:
        print(f"Dataset file '{file_path}' not found.")
        print("Please update the file_path variable with the correct path to your dataset.")
        return None, None
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return None, None

def step_by_step_example():
    """
    Example showing how to use the pipeline step by step
    """
    print("Step-by-step pipeline example...")
    
    # Create sample data
    data = create_sample_dataset()
    data.to_csv('step_by_step_dataset.csv', index=False)
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Step 2: Load dataset
    pipeline.load_dataset('step_by_step_dataset.csv', 'target')
    
    # Step 3: Inspect data
    pipeline.inspect_data()
    
    # Step 4: Preprocess data
    pipeline.preprocess_data(text_columns=['text_feature'])
    
    # Step 5: Encode categorical features
    pipeline.encode_categorical_features(['categorical_feature1', 'categorical_feature2'])
    
    # Step 6: Apply TF-IDF
    pipeline.apply_tfidf_vectorization(['text_feature'])
    
    # Step 7: Split data
    pipeline.split_data(test_size=0.2)
    
    # Step 8: Apply PCA
    pipeline.apply_pca(n_components=0.95)
    
    # Step 9: Train model
    pipeline.train_stacking_model()
    
    # Step 10: Evaluate
    results = pipeline.evaluate_model()
    
    # Save model
    pipeline.save_model('my_stacking_model.pkl')
    
    return pipeline, results

if __name__ == "__main__":
    print("Stacking ML Pipeline Examples")
    print("=" * 50)
    
    # Choose which example to run
    print("Choose an example to run:")
    print("1. Run with sample generated data")
    print("2. Run with your own dataset (modify parameters first)")
    print("3. Step-by-step example")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\nRunning example with sample data...")
        pipeline, results = example_with_sample_data()
        
    elif choice == "2":
        print("\nRunning example with your dataset...")
        print("Make sure to modify the parameters in example_with_your_dataset() function first!")
        pipeline, results = example_with_your_dataset()
        
    elif choice == "3":
        print("\nRunning step-by-step example...")
        pipeline, results = step_by_step_example()
        
    else:
        print("Invalid choice. Running sample data example by default...")
        pipeline, results = example_with_sample_data()
    
    if results:
        print(f"\nFinal Results:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed to complete.")
