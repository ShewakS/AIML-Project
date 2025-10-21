# Stacking ML Pipeline: Random Forest + Logistic Regression

A comprehensive machine learning pipeline that implements stacking ensemble using Random Forest and Logistic Regression as base estimators, following industry best practices for data preprocessing, feature engineering, and model evaluation.

## 🚀 Features

- **Complete ML Pipeline**: Follows all 10 essential steps from data loading to model evaluation
- **Stacking Ensemble**: Combines Random Forest and Logistic Regression for improved accuracy
- **Text Processing**: Advanced TF-IDF vectorization with n-grams and stop word removal
- **Dimensionality Reduction**: PCA implementation with explained variance visualization
- **Comprehensive Evaluation**: Multiple metrics, confusion matrix, ROC curves, and model comparison
- **Automated Preprocessing**: Handles missing values, duplicates, and categorical encoding
- **Visualization**: Rich plots for data exploration and model performance analysis
- **Model Persistence**: Save and load trained models with all preprocessors

## 📋 Pipeline Steps

1. **Import Libraries** - All necessary ML and visualization libraries
2. **Load Dataset** - Support for CSV and Excel files
3. **Inspect Data** - Comprehensive data exploration and visualization
4. **Clean Text & Handle Missing Values** - Text preprocessing and data cleaning
5. **Encode Categorical Features** - Label encoding for categorical variables
6. **Apply TF-IDF Vectorization** - Advanced text feature extraction
7. **Split into Train/Test** - Stratified splitting for balanced datasets
8. **Reduce Dimensions with PCA** - Dimensionality reduction with variance analysis
9. **Train ML Model** - Stacking ensemble with cross-validation
10. **Evaluate and Visualize Results** - Comprehensive performance analysis

## 🛠️ Installation

1. Clone or download the repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Usage

### Quick Start

```python
from stacking_ml_pipeline import run_complete_pipeline

# Run the complete pipeline
pipeline, results = run_complete_pipeline(
    file_path="your_dataset.csv",
    target_column="target",
    text_columns=["text_column1", "text_column2"],  # Optional
    categorical_columns=["cat_column1", "cat_column2"]  # Optional
)

print(f"Final Accuracy: {results['accuracy']:.4f}")
```

### Step-by-Step Usage

```python
from stacking_ml_pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline()

# Load and process data
pipeline.load_dataset("dataset.csv", "target_column")
pipeline.inspect_data()
pipeline.preprocess_data(text_columns=["text_col"])
pipeline.encode_categorical_features()
pipeline.apply_tfidf_vectorization()
pipeline.split_data()
pipeline.apply_pca()

# Train and evaluate
pipeline.train_stacking_model()
results = pipeline.evaluate_model()

# Save model
pipeline.save_model("my_model.pkl")
```

### Example with Sample Data

```python
# Run the example script
python example_usage.py
```

## 📈 Model Architecture

The stacking ensemble uses:

- **Base Estimators**:
  - Random Forest Classifier (100 trees, optimized hyperparameters)
  - Logistic Regression (L2 regularization, liblinear solver)

- **Meta Estimator**:
  - Logistic Regression (final prediction layer)

- **Stacking Configuration**:
  - 5-fold cross-validation
  - Probability-based stacking method
  - Parallel processing enabled

## 📊 Evaluation Metrics

The pipeline provides comprehensive evaluation including:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visual Analysis**: Confusion Matrix, ROC Curves, Feature Importance
- **Model Comparison**: Individual vs Ensemble performance
- **Learning Curves**: Training progression analysis
- **Cross-Validation**: Robust performance estimation

## 🔧 Customization

### Hyperparameter Tuning

Modify the model parameters in the `train_stacking_model()` method:

```python
rf_classifier = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=15,      # Adjust depth
    min_samples_split=10,
    random_state=42
)
```

### Feature Engineering

Customize TF-IDF parameters:

```python
self.tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,    # More features
    ngram_range=(1, 3),    # Include trigrams
    min_df=3,              # Minimum document frequency
    max_df=0.9             # Maximum document frequency
)
```

### PCA Configuration

Adjust dimensionality reduction:

```python
pipeline.apply_pca(n_components=0.99)  # Retain 99% variance
```

## 📁 File Structure

```
aiml/
├── stacking_ml_pipeline.py    # Main pipeline implementation
├── example_usage.py           # Usage examples and demonstrations
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation
└── models/                    # Saved models (created after training)
    └── stacking_model.pkl
```

## 🎯 Performance Features

- **Memory Efficient**: Optimized for large datasets
- **Parallel Processing**: Multi-core utilization for faster training
- **Scalable**: Handles datasets from small to large scale
- **Robust**: Comprehensive error handling and validation
- **Reproducible**: Fixed random seeds for consistent results

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.0.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `max_features` in TF-IDF or `n_components` in PCA
2. **Slow Training**: Decrease `n_estimators` in Random Forest or use fewer CV folds
3. **Poor Performance**: Check data quality, feature engineering, or try hyperparameter tuning

### Support

For issues and questions:
1. Check the example usage scripts
2. Review the troubleshooting section
3. Create an issue with detailed error information

## 🎉 Acknowledgments

Built with scikit-learn and follows ML best practices for production-ready pipelines.

---

**Happy Machine Learning! 🚀**
