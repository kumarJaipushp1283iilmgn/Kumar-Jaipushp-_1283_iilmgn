# Fake News Detection AI/ML Project

A machine learning project that detects fake news using natural language processing and multiple classification algorithms.

## Features

- **Multiple ML Models**: Naive Bayes, Random Forest, and SVM classifiers
- **Text Preprocessing**: Cleaning, stemming, and stopword removal
- **TF-IDF Vectorization**: Convert text to numerical features
- **Model Comparison**: Evaluate and compare different algorithms
- **Web Interface**: Interactive Flask web app for testing
- **Visualization**: Performance charts and confusion matrices

## Project Structure

```
├── fake_news_detector.py    # Main ML pipeline
├── web_app.py              # Flask web application
├── templates/
│   └── index.html          # Web interface
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if needed):
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Command Line Interface

Run the main detection system:
```bash
python fake_news_detector.py
```

This will:
- Train multiple ML models
- Display accuracy comparisons
- Show confusion matrices
- Test with sample articles

### Web Interface

1. Start the Flask app:
```bash
python web_app.py
```

2. Open your browser and go to: `http://localhost:5000`

3. Enter news text and select a model to get predictions

## Models Used

1. **Naive Bayes**: Fast and effective for text classification
2. **Random Forest**: Ensemble method with good accuracy
3. **SVM**: Support Vector Machine with linear kernel

## Text Preprocessing Steps

1. Convert to lowercase
2. Remove special characters and digits
3. Remove stopwords
4. Apply stemming
5. TF-IDF vectorization

## Sample Results

The system can distinguish between fake and real news with high accuracy. Example predictions:

- **Real News**: "Scientists develop new renewable energy technology"
- **Fake News**: "Government confirms aliens living among us"

## Extending the Project

To improve the model:

1. **Add More Data**: Use larger datasets like LIAR or FakeNewsNet
2. **Feature Engineering**: Add metadata features (author, source, etc.)
3. **Deep Learning**: Implement LSTM or BERT models
4. **Ensemble Methods**: Combine multiple model predictions

## Dataset

Currently uses sample data for demonstration. For production use, consider:
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [Fake News Detection Dataset](https://www.kaggle.com/c/fake-news/data)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- flask
- matplotlib
