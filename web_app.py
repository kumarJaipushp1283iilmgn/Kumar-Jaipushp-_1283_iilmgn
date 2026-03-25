from flask import Flask, render_template, request, jsonify
import pickle
import os
from fake_news_detector import FakeNewsDetector
import nltk

app = Flask(__name__)

# Global detector instance
detector = None

def initialize_detector():
    """Initialize and train the detector"""
    global detector
    detector = FakeNewsDetector()
    
    # Train with sample data
    df = detector.load_and_prepare_data()
    detector.train_models(df)
    
    print("Detector initialized and trained!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model = data.get('model', 'Naive Bayes')
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text'})
        
        # Get detailed prediction with probabilities
        cleaned_text = detector.preprocess_text(text)
        text_vec = detector.vectorizer.transform([cleaned_text])
        
        prediction = detector.trained_models[model].predict(text_vec)[0]
        probabilities = detector.trained_models[model].predict_proba(text_vec)[0]
        
        real_confidence = probabilities[0] * 100
        fake_confidence = probabilities[1] * 100
        
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'real_confidence': round(real_confidence, 1),
            'fake_confidence': round(fake_confidence, 1)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Initialize detector on startup
    initialize_detector()
    app.run(debug=True, port=5000)