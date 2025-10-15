# 🎯 Sentiment Analysis AI

A production-ready sentiment analysis model built with Python and scikit-learn. This project demonstrates best practices in ML engineering with clean architecture, comprehensive documentation, and modular design.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Details](#model-details)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎓 Overview

This project implements a sentiment analysis classifier that can categorize text into positive, negative, or neutral sentiments. It's designed with production deployment in mind, featuring:

- **Modular architecture** for easy maintenance and extension
- **Comprehensive preprocessing pipeline** with text cleaning and normalization
- **Multiple model support** (Naive Bayes, Logistic Regression, SVM)
- **Robust evaluation metrics** and visualization
- **CLI tools** for training and inference
- **Well-documented code** with docstrings and type hints

## ✨ Features

- 🧹 **Text Preprocessing**: Cleaning, tokenization, stopword removal, lemmatization
- 🤖 **Multiple Models**: Support for different classifiers
- 📊 **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
- 💾 **Model Persistence**: Save and load trained models
- 🔧 **Configurable**: YAML-based configuration
- 📝 **Logging**: Detailed logging for debugging
- 🧪 **Unit Tests**: Comprehensive test coverage
- 📓 **Jupyter Notebooks**: Interactive examples and tutorials
- ⚡ **Performance Optimized**: Caching, parallel processing, memory efficiency
- 🚀 **Production Ready**: Benchmarking tools and optimized configurations

## 📁 Project Structure

```
sentiment-analysis-ai/
│
├── data/                      # Data directory
│   ├── raw/                   # Raw dataset files
│   │   └── reviews.csv        # Sample review dataset
│   └── processed/             # Processed data files
│       └── .gitkeep
│
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessor.py       # Text preprocessing (optimized)
│   ├── model.py              # Model definitions (optimized)
│   ├── trainer.py            # Training logic
│   ├── evaluator.py          # Model evaluation
│   ├── inference.py          # Prediction utilities
│   └── utils.py              # Helper functions
│
├── models/                    # Saved models
│   └── .gitkeep
│
├── notebooks/                 # Jupyter notebooks
│   └── demo.ipynb            # Demo and examples
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_model.py
│   └── test_trainer.py
│
├── docs/                      # Documentation
│   ├── API.md                # API documentation
│   ├── CONTRIBUTING.md       # Contribution guidelines
│   └── OPTIMIZATION.md       # Performance optimization guide
│
├── train.py                   # Training script
├── predict.py                 # Inference script
├── benchmark.py              # Performance benchmarking
├── config.yaml               # Configuration file
├── config_optimized.yaml     # Optimized configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── .gitignore               # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-ai.git
   cd sentiment-analysis-ai
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (for preprocessing)
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
   ```

## ⚡ Quick Start

### Training a Model

```bash
python train.py
```

This will:
- Load the sample dataset
- Preprocess the text data
- Train the sentiment classifier
- Save the model to `models/`
- Display evaluation metrics

**With optimized configuration:**
```bash
python train.py --config config_optimized.yaml
```

### Making Predictions

```bash
python predict.py --text "This product is amazing! I love it!"
```

Or for batch predictions:

```bash
python predict.py --file data/new_reviews.csv
```

## 📖 Usage

### Using the Python API

```python
from src.data_loader import load_data
from src.preprocessor import TextPreprocessor
from src.model import SentimentModel
from src.trainer import ModelTrainer

# Load data
X_train, X_test, y_train, y_test = load_data('data/raw/reviews.csv')

# Initialize components
preprocessor = TextPreprocessor()
model = SentimentModel(model_type='naive_bayes')
trainer = ModelTrainer(model, preprocessor)

# Train model
trainer.train(X_train, y_train)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save model
trainer.save_model('models/my_model.pkl')
```

### Making Predictions

```python
from src.inference import SentimentPredictor

# Load trained model
predictor = SentimentPredictor('models/sentiment_model.pkl')

# Predict sentiment
text = "This is an excellent product!"
result = predictor.predict(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🧠 Model Details

### Architecture

The default model uses a **Multinomial Naive Bayes** classifier with TF-IDF vectorization:

- **Feature Extraction**: TF-IDF with n-grams (1-2)
- **Max Features**: 5000
- **Classifier**: Multinomial Naive Bayes
- **Classes**: Positive, Negative, Neutral

### Preprocessing Pipeline

1. **Text Cleaning**
   - Lowercase conversion
   - Punctuation removal
   - Number handling
   - Special character removal

2. **Tokenization**
   - Word-level tokenization
   - Stopword removal

3. **Normalization**
   - Lemmatization
   - Stemming (optional)

4. **Vectorization**
   - TF-IDF transformation
   - N-gram features

### Performance

On the sample dataset:
- **Accuracy**: ~85-90%
- **Precision**: ~86%
- **Recall**: ~85%
- **F1-Score**: ~85%

*Note: Performance varies based on the dataset and model configuration.*

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocessor.py
```

### Performance Benchmarking

```bash
# Compare default vs optimized configurations
python benchmark.py --compare

# Benchmark specific configuration
python benchmark.py --config config_optimized.yaml
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Check code style with flake8
flake8 src/ tests/
```

### Configuration

Edit `config.yaml` to customize:
- Data paths
- Model hyperparameters
- Preprocessing options
- Training settings
- Performance optimizations

For production use, consider `config_optimized.yaml` which includes:
- Parallel preprocessing (3-7x faster)
- Intelligent caching
- Optimized model parameters
- Multi-core utilization
- Memory-efficient settings

See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) for detailed optimization guide.

## 🧪 Testing

The project includes comprehensive unit tests for all major components:

- `test_preprocessor.py`: Text preprocessing tests
- `test_model.py`: Model initialization and prediction tests
- `test_trainer.py`: Training and evaluation tests

Run tests regularly during development to ensure code quality.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/sentiment-analysis-ai](https://github.com/yourusername/sentiment-analysis-ai)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [NLTK](https://www.nltk.org/) for NLP utilities
- [pandas](https://pandas.pydata.org/) for data manipulation

---

**Made with ❤️ for the AI community**
