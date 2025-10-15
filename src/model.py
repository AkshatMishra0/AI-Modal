"""
Sentiment analysis model definitions.

This module contains the model classes for sentiment classification
with performance optimizations.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import logging
import joblib
from functools import lru_cache

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
        use_idf: bool = True,
        sublinear_tf: bool = True,
        **model_kwargs
    ):
        """
        Initialize the sentiment model with optimizations.
        
        Args:
            model_type (str): Type of model ('naive_bayes', 'logistic_regression', 'svm').
            max_features (int): Maximum number of features for TF-IDF. Default is 5000.
            ngram_range (tuple): N-gram range for TF-IDF. Default is (1, 2).
            min_df (int): Minimum document frequency. Default is 2.
            max_df (float): Maximum document frequency. Default is 0.95.
            use_idf (bool): Enable IDF weighting. Default is True.
            sublinear_tf (bool): Apply sublinear tf scaling. Default is True.
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
        
        # Initialize optimized TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b',
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            dtype=np.float32  # Use float32 for memory efficiency
        )
        
        # Initialize classifier with optimized parameters
        if model_type == 'naive_bayes':
            default_kwargs = {'alpha': 0.1}  # Optimized smoothing
            default_kwargs.update(model_kwargs)
            self.classifier = MultinomialNB(**default_kwargs)
        elif model_type == 'logistic_regression':
            default_kwargs = {
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'saga',  # Faster for large datasets
                'C': 1.0,
                'n_jobs': -1  # Use all CPU cores
            }
            default_kwargs.update(model_kwargs)
            self.classifier = LogisticRegression(**default_kwargs)
        elif model_type == 'svm':
            default_kwargs = {
                'kernel': 'linear',
                'random_state': 42,
                'probability': True,
                'C': 1.0,
                'cache_size': 500  # Increased cache for faster training
            }
            default_kwargs.update(model_kwargs)
            self.classifier = SVC(**default_kwargs)
        
        logger.info(f"SentimentModel initialized with {model_type} (optimized)")
    
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
    
    def cross_validate(
        self,
        X: list,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X (list): Training texts.
            y (np.ndarray): Training labels.
            cv (int): Number of cross-validation folds. Default is 5.
            scoring (str): Scoring metric. Default is 'accuracy'.
        
        Returns:
            Dict[str, Any]: Cross-validation results.
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Transform texts
        X_vec = self.vectorizer.fit_transform(X)
        
        # Perform cross-validation
        scores = cross_val_score(
            self.classifier,
            X_vec,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1  # Use all cores
        )
        
        results = {
            'scores': scores.tolist(),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores))
        }
        
        logger.info(f"CV {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get approximate memory usage of model components.
        
        Returns:
            Dict[str, float]: Memory usage in MB for each component.
        """
        import sys
        
        memory = {
            'vectorizer': sys.getsizeof(self.vectorizer) / 1024 / 1024,
            'classifier': sys.getsizeof(self.classifier) / 1024 / 1024,
        }
        
        # Add vocabulary size if fitted
        if hasattr(self.vectorizer, 'vocabulary_'):
            memory['vocabulary_size'] = len(self.vectorizer.vocabulary_)
        
        return memory


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
