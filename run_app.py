"""
Startup script for the Flask web application
"""

import os
import sys
from app import app, load_model_and_data

def main():
    """Main function to start the Flask application"""
    print("=" * 60)
    print("AI Question Generator & Evaluator")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        'final_modified_dataset.csv',
        'optimized_honest_model.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all required files are in the current directory.")
        return
    
    print("✅ All required files found")
    
    # Load model and data
    print("🔄 Loading model and dataset...")
    load_model_and_data()
    
    print("🚀 Starting Flask application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
