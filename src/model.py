"""
Sentiment analysis model definitions.

This module contains the model classes for sentiment classification.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentModel:
    """
    Sentiment analysis model wrapper.
    
    Supports multiple classifier types: Naive Bayes, Logistic Regression, SVM.
    Includes TF-IDF vectorization for text feature extraction.
    
    Attributes:
        model_type (str): Type of classifier to use.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer for text.
        classifier: The underlying classifier model.
    """
    
    SUPPORTED_MODELS = {
        'naive_bayes': MultinomialNB,
        'logistic_regression': LogisticRegression,
        'svm': SVC
    }
    
    def __init__(
        self,
        model_type: str = 'naive_bayes',
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        **model_kwargs
    ):
        """
        Initialize the sentiment model.
        
        Args:
            model_type (str): Type of model ('naive_bayes', 'logistic_regression', 'svm').
            max_features (int): Maximum number of features for TF-IDF. Default is 5000.
            ngram_range (tuple): N-gram range for TF-IDF. Default is (1, 2).
            min_df (int): Minimum document frequency. Default is 2.
            max_df (float): Maximum document frequency. Default is 0.95.
            **model_kwargs: Additional arguments for the classifier.
        
        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_type = model_type
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        # Initialize classifier
        if model_type == 'naive_bayes':
            self.classifier = MultinomialNB(**model_kwargs)
        elif model_type == 'logistic_regression':
            default_kwargs = {'max_iter': 1000, 'random_state': 42}
            default_kwargs.update(model_kwargs)
            self.classifier = LogisticRegression(**default_kwargs)
        elif model_type == 'svm':
            default_kwargs = {'kernel': 'linear', 'random_state': 42, 'probability': True}
            default_kwargs.update(model_kwargs)
            self.classifier = SVC(**default_kwargs)
        
        logger.info(f"SentimentModel initialized with {model_type}")
    
    def fit(self, X_train: list, y_train: np.ndarray) -> 'SentimentModel':
        """
        Fit the model on training data.
        
        Args:
            X_train (list): Training texts.
            y_train (np.ndarray): Training labels.
        
        Returns:
            SentimentModel: The fitted model (self).
        """
        logger.info(f"Fitting vectorizer on {len(X_train)} samples")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        logger.info(f"Training {self.model_type} classifier")
        self.classifier.fit(X_train_vec, y_train)
        
        return self
    
    def predict(self, X: list) -> np.ndarray:
        """
        Predict sentiments for input texts.
        
        Args:
            X (list): Input texts.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)
    
    def predict_proba(self, X: list) -> np.ndarray:
        """
        Predict sentiment probabilities for input texts.
        
        Args:
            X (list): Input texts.
        
        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)
    
    def get_feature_names(self) -> list:
        """
        Get the feature names from the vectorizer.
        
        Returns:
            list: List of feature names (words/n-grams).
        """
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n: int = 20, class_idx: int = 1) -> list:
        """
        Get top features (words) for a specific class.
        
        Args:
            n (int): Number of top features to return. Default is 20.
            class_idx (int): Index of the class. Default is 1.
        
        Returns:
            list: List of (feature, score) tuples.
        """
        if hasattr(self.classifier, 'feature_log_prob_'):
            # For Naive Bayes
            feature_names = self.get_feature_names()
            scores = self.classifier.feature_log_prob_[class_idx]
            top_indices = np.argsort(scores)[-n:][::-1]
            return [(feature_names[i], scores[i]) for i in top_indices]
        elif hasattr(self.classifier, 'coef_'):
            # For Logistic Regression and SVM
            feature_names = self.get_feature_names()
            if len(self.classifier.coef_.shape) == 1:
                scores = self.classifier.coef_
            else:
                scores = self.classifier.coef_[class_idx]
            top_indices = np.argsort(scores)[-n:][::-1]
            return [(feature_names[i], scores[i]) for i in top_indices]
        else:
            logger.warning("Model doesn't support feature importance extraction")
            return []
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters.
        """
        return {
            'model_type': self.model_type,
            'vectorizer_params': self.vectorizer.get_params(),
            'classifier_params': self.classifier.get_params()
        }


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import fetch_20newsgroups
    
    # Load sample data
    categories = ['alt.atheism', 'soc.religion.christian']
    data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    
    # Create and train model
    model = SentimentModel(model_type='naive_bayes')
    model.fit(data.data[:100], data.target[:100])
    
    # Make predictions
    test_text = ["This is about religion", "This is about atheism"]
    predictions = model.predict(test_text)
    print(f"Predictions: {predictions}")
