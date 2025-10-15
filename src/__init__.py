"""
Sentiment Analysis AI Package

A production-ready sentiment analysis model with modular design.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.data_loader import load_data, DataLoader
from src.preprocessor import TextPreprocessor
from src.model import SentimentModel
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.inference import SentimentPredictor

__all__ = [
    "load_data",
    "DataLoader",
    "TextPreprocessor",
    "SentimentModel",
    "ModelTrainer",
    "ModelEvaluator",
    "SentimentPredictor",
]
