"""
Flask Web Application for Question Generation and Evaluation
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import json

app = Flask(__name__)

# Global variables for model and data
model_data = None
dataset = None

def parse_options(options_raw):
    """Parse options from various formats"""
    try:
        if isinstance(options_raw, str):
            # Try to parse as list first
            import ast
            try:
                parsed = ast.literal_eval(options_raw)
                if isinstance(parsed, list):
                    # Check if options are concatenated (common issue in CSV)
                    if len(parsed) == 1 and isinstance(parsed[0], str):
                        # Try to split concatenated options
                        single_option = parsed[0]
                        
                        # Common patterns to split on
                        split_patterns = [
                            'not enough information',
                            'The narrator',
                            'the twelve-year-old boy',
                            'The narrator\'s colleague',
                            'The models trim their toenails',
                            'The models smoke cigars',
                            'The models have sex',
                            'pleasant',
                            'ok',
                            'miserable'
                        ]
                        
                        options = []
                        remaining = single_option
                        
                        for pattern in split_patterns:
                            if pattern in remaining:
                                # Find the position of the pattern
                                pos = remaining.find(pattern)
                                if pos > 0:
                                    # Add the text before the pattern
                                    before = remaining[:pos].strip()
                                    if before:
                                        options.append(before)
                                
                                # Add the pattern itself
                                options.append(pattern)
                                
                                # Update remaining text
                                remaining = remaining[pos + len(pattern):].strip()
                        
                        # Add any remaining text
                        if remaining.strip():
                            options.append(remaining.strip())
                        
                        if options:
                            return [opt.strip() for opt in options if opt.strip()]
                    
                    # If it's a proper list, return as is
                    return [str(opt).strip() for opt in parsed if str(opt).strip()]
            except:
                pass
            
            # Fallback: try to extract quoted strings
            import re
            matches = re.findall(r"'([^']+)'", options_raw)
            if matches:
                return [opt.strip() for opt in matches if opt.strip()]
            
            # Last resort
            return [str(options_raw)]
        
        elif isinstance(options_raw, list):
            return [str(opt).strip() for opt in options_raw if str(opt).strip()]
        
        else:
            return [str(options_raw)]
            
    except Exception as e:
        print(f"Error parsing options: {e}")
        return ["Error parsing options"]

def load_model_and_data():
    """Load the trained model and dataset"""
    global model_data, dataset
    
    try:
        # Load the trained model
        if os.path.exists('optimized_honest_model.pkl'):
            model_data = joblib.load('optimized_honest_model.pkl')
            print("Model loaded successfully")
        else:
            print("Model file not found")
            
        # Load the dataset
        if os.path.exists('final_modified_dataset.csv'):
            dataset = pd.read_csv('final_modified_dataset.csv')
            print(f"Dataset loaded: {dataset.shape}")
        else:
            print("Dataset file not found")
            
    except Exception as e:
        print(f"Error loading model/data: {e}")

def get_dataset_info():
    """Get information about the dataset for the frontend"""
    if dataset is None:
        return None
    
    # Get unique values for dropdowns
    subjects = dataset['Subject'].unique().tolist()
    difficulty_levels = dataset['Difficulty_Level'].unique().tolist()
    question_types = dataset['Question_Type'].unique().tolist()
    domains = dataset['domain'].unique().tolist()
    
    return {
        'subjects': subjects,
        'difficulty_levels': difficulty_levels,
        'question_types': question_types,
        'domains': domains,
        'total_samples': len(dataset)
    }

def generate_question_from_features(features):
    """Generate a question based on input features using ML model"""
    if dataset is None:
        return None, "Dataset not loaded"
    
    try:
        # Filter dataset based on input features
        filtered_data = dataset.copy()
        
        if features.get('subject'):
            filtered_data = filtered_data[filtered_data['Subject'] == features['subject']]
        
        if features.get('difficulty'):
            filtered_data = filtered_data[filtered_data['Difficulty_Level'] == features['difficulty']]
        
        if features.get('question_type'):
            filtered_data = filtered_data[filtered_data['Question_Type'] == features['question_type']]
        
        if features.get('domain'):
            filtered_data = filtered_data[filtered_data['domain'] == features['domain']]
        
        if len(filtered_data) == 0:
            return None, "No matching questions found with the given criteria"
        
        # Use ML model to predict difficulty and select appropriate question
        if model_data and 'model' in model_data:
            # Create features for the ML model
            question_features = create_question_features(filtered_data, features)
            
            # Use model to predict and select best question
            if question_features is not None:
                predictions = model_data['model'].predict_proba(question_features)
                # Select question with highest confidence
                best_idx = np.argmax(predictions.max(axis=1))
                selected_question = filtered_data.iloc[best_idx]
            else:
                # Fallback to random selection
                selected_question = filtered_data.sample(n=1).iloc[0]
        else:
            # Fallback to random selection if no model
            selected_question = filtered_data.sample(n=1).iloc[0]
        
        # Parse options correctly
        options_raw = selected_question['Options']
        options = parse_options(options_raw)
        
        # Format the question data
        question_data = {
            'question_id': int(selected_question['question_id']),
            'passage': selected_question['Passage'],
            'question': selected_question['Question'],
            'options': options,
            'correct_option': int(selected_question['Correct_Option']),
            'subject': selected_question['Subject'],
            'difficulty': selected_question['Difficulty_Level'],
            'question_type': selected_question['Question_Type'],
            'domain': selected_question['domain'],
            'keywords': selected_question['Keywords'],
            'hint': selected_question['Hint'],
            'explanation': selected_question['Explanation'],
            'marks': int(selected_question['Marks'])
        }
        
        return question_data, None
        
    except Exception as e:
        return None, f"Error generating question: {str(e)}"

def create_question_features(filtered_data, user_features):
    """Create features for ML model prediction"""
    try:
        # This is a simplified version - in practice, you'd need to recreate
        # the same feature engineering pipeline from optimized_honest_pipeline.py
        
        # For now, return None to use random selection
        # In a full implementation, you would:
        # 1. Apply the same preprocessing steps
        # 2. Create the same features as in the training pipeline
        # 3. Transform the data using the saved preprocessors
        
        return None
        
    except Exception as e:
        print(f"Error creating question features: {e}")
        return None

def evaluate_answer(question_data, selected_option):
    """Evaluate the selected answer"""
    try:
        correct_option = question_data['correct_option']
        is_correct = selected_option == correct_option
        
        result = {
            'is_correct': is_correct,
            'selected_option': selected_option,
            'correct_option': correct_option,
            'explanation': question_data['explanation'],
            'hint': question_data['hint']
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Error evaluating answer: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    dataset_info = get_dataset_info()
    return render_template('index.html', dataset_info=dataset_info)

@app.route('/api/generate', methods=['POST'])
def generate_question():
    """API endpoint to generate a question"""
    try:
        features = request.json
        
        question_data, error = generate_question_from_features(features)
        
        if error:
            return jsonify({'success': False, 'error': error})
        
        return jsonify({'success': True, 'question': question_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/evaluate', methods=['POST'])
def evaluate_question():
    """API endpoint to evaluate an answer"""
    try:
        data = request.json
        question_data = data['question']
        selected_option = int(data['selected_option'])
        
        result, error = evaluate_answer(question_data, selected_option)
        
        if error:
            return jsonify({'success': False, 'error': error})
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/dataset-info')
def dataset_info():
    """API endpoint to get dataset information"""
    info = get_dataset_info()
    if info is None:
        return jsonify({'success': False, 'error': 'Dataset not loaded'})
    
    return jsonify({'success': True, 'info': info})

if __name__ == '__main__':
    # Load model and data on startup
    load_model_and_data()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
