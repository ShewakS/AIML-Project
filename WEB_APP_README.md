# AI Question Generator & Evaluator Web Application

A Flask-based web application that generates intelligent questions from your dataset and provides evaluation capabilities.

## Features

- **Smart Question Generation**: Generate questions based on subject, difficulty, type, and domain criteria
- **Interactive Interface**: Modern, responsive web interface with Bootstrap styling
- **Real-time Evaluation**: Get instant feedback on your answers with explanations
- **Dataset Statistics**: View comprehensive statistics about your dataset
- **ML Integration**: Uses your trained ML model for intelligent question selection

## Files Structure

```
├── app.py                          # Main Flask application
├── run_app.py                      # Startup script
├── templates/
│   └── index.html                  # Main web interface
├── requirements_flask.txt          # Python dependencies
├── final_modified_dataset.csv      # Your dataset
├── optimized_honest_model.pkl      # Your trained ML model
└── WEB_APP_README.md              # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_flask.txt
```

### 2. Run the Application
```bash
python run_app.py
```

### 3. Open in Browser
Navigate to: `http://localhost:5000`

## How to Use

1. **Select Criteria**: Choose subject, difficulty level, question type, and domain
2. **Generate Question**: Click "Generate Question" to get a question based on your criteria
3. **Answer Question**: Read the passage and question, then select your answer
4. **Evaluate**: Click "Evaluate Answer" to get instant feedback with explanations
5. **Generate New**: Click "Generate New Question" to start over

## Features Explained

### Question Generation
- Filters your dataset based on selected criteria
- Uses ML model for intelligent question selection (when available)
- Falls back to random selection if ML model is not available

### Evaluation System
- Compares your selected answer with the correct option
- Provides detailed explanations and hints
- Shows performance feedback

### Dataset Integration
- Automatically loads your dataset statistics
- Populates dropdown menus with available options
- Handles missing or invalid data gracefully

## Technical Details

### Backend (Flask)
- **app.py**: Main application with API endpoints
- **API Endpoints**:
  - `/`: Main page
  - `/api/generate`: Generate questions based on criteria
  - `/api/evaluate`: Evaluate selected answers
  - `/api/dataset-info`: Get dataset statistics

### Frontend (HTML/CSS/JavaScript)
- **Bootstrap 5**: Modern, responsive design
- **Font Awesome**: Icons and visual elements
- **Custom CSS**: Gradient backgrounds and animations
- **JavaScript**: Dynamic content loading and user interactions

### ML Integration
- Loads your trained model from `optimized_honest_model.pkl`
- Uses model for intelligent question selection
- Graceful fallback to random selection if model unavailable

## Troubleshooting

### Common Issues

1. **"Dataset not loaded" error**:
   - Ensure `final_modified_dataset.csv` is in the same directory
   - Check file permissions

2. **"Model not found" error**:
   - Ensure `optimized_honest_model.pkl` is in the same directory
   - The app will work without the model (uses random selection)

3. **Port already in use**:
   - Change the port in `run_app.py` (line: `app.run(debug=True, host='0.0.0.0', port=5000)`)
   - Or kill the process using port 5000

4. **Unicode/Emoji errors**:
   - The web interface handles Unicode properly
   - If you see encoding issues, ensure your terminal supports UTF-8

### Performance Tips

- The app loads the entire dataset into memory for fast filtering
- For very large datasets, consider implementing pagination
- Model loading happens once at startup for better performance

## Customization

### Adding New Features
1. Modify `app.py` to add new API endpoints
2. Update `templates/index.html` for new UI elements
3. Add new JavaScript functions for client-side functionality

### Styling Changes
- Modify the CSS in `templates/index.html`
- Add new Bootstrap classes or custom styles
- Update color schemes and animations

### Dataset Modifications
- The app automatically adapts to your dataset structure
- Add new columns to get more filtering options
- Modify the feature extraction in `app.py` if needed

## Security Notes

- The app runs in debug mode for development
- For production, set `debug=False` in `run_app.py`
- Consider adding authentication for production use
- Validate all user inputs on the server side

## Next Steps

1. **Deploy to Production**: Use services like Heroku, AWS, or DigitalOcean
2. **Add Authentication**: Implement user login and session management
3. **Database Integration**: Store user progress and results
4. **Advanced ML**: Implement more sophisticated question generation
5. **Analytics**: Add user behavior tracking and performance metrics

---

**Happy Question Generating! 🚀**
