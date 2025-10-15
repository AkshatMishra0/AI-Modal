"""
Unit tests for the SentimentModel class.

Run with: pytest tests/test_model.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SentimentModel


class TestSentimentModel:
    """Test cases for SentimentModel."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X_train = [
            "This is great and amazing",
            "Terrible and awful product",
            "It is okay and acceptable",
            "Excellent and fantastic",
            "Bad and disappointing",
            "Average and normal"
        ]
        y_train = np.array([2, 0, 1, 2, 0, 1])  # 0=negative, 1=neutral, 2=positive
        return X_train, y_train
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a model."""
        X_train, y_train = sample_data
        model = SentimentModel(model_type='naive_bayes')
        model.fit(X_train, y_train)
        return model
    
    def test_model_initialization_naive_bayes(self):
        """Test Naive Bayes model initialization."""
        model = SentimentModel(model_type='naive_bayes')
        assert model is not None
        assert model.model_type == 'naive_bayes'
        assert model.vectorizer is not None
        assert model.classifier is not None
    
    def test_model_initialization_logistic_regression(self):
        """Test Logistic Regression model initialization."""
        model = SentimentModel(model_type='logistic_regression')
        assert model.model_type == 'logistic_regression'
    
    def test_model_initialization_svm(self):
        """Test SVM model initialization."""
        model = SentimentModel(model_type='svm')
        assert model.model_type == 'svm'
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            SentimentModel(model_type='invalid_model')
    
    def test_vectorizer_parameters(self):
        """Test vectorizer initialization with custom parameters."""
        model = SentimentModel(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.8
        )
        assert model.vectorizer.max_features == 1000
        assert model.vectorizer.ngram_range == (1, 3)
        assert model.vectorizer.min_df == 3
        assert model.vectorizer.max_df == 0.8
    
    def test_model_fit(self, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        model = SentimentModel(model_type='naive_bayes')
        
        # Should not raise any errors
        result = model.fit(X_train, y_train)
        
        # Should return self
        assert result is model
        
        # Classifier should be fitted
        assert hasattr(model.classifier, 'classes_')
    
    def test_model_predict(self, trained_model):
        """Test model prediction."""
        test_texts = ["This is amazing", "This is terrible"]
        predictions = trained_model.predict(test_texts)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(test_texts)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
    
    def test_model_predict_proba(self, trained_model):
        """Test probability prediction."""
        test_texts = ["This is amazing"]
        probabilities = trained_model.predict_proba(test_texts)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(test_texts)
        assert probabilities.shape[1] > 0  # Should have probabilities for each class
        
        # Probabilities should sum to 1
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # All probabilities should be between 0 and 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_get_feature_names(self, trained_model):
        """Test feature name extraction."""
        features = trained_model.get_feature_names()
        assert isinstance(features, (list, np.ndarray))
        assert len(features) > 0
    
    def test_get_top_features(self, trained_model):
        """Test top feature extraction."""
        top_features = trained_model.get_top_features(n=5, class_idx=0)
        assert isinstance(top_features, list)
        assert len(top_features) <= 5
        
        # Each item should be a tuple of (feature, score)
        for item in top_features:
            assert isinstance(item, tuple)
            assert len(item) == 2
    
    def test_get_params(self, trained_model):
        """Test parameter retrieval."""
        params = trained_model.get_params()
        assert isinstance(params, dict)
        assert 'model_type' in params
        assert 'vectorizer_params' in params
        assert 'classifier_params' in params
    
    def test_prediction_consistency(self, trained_model):
        """Test that predictions are consistent."""
        test_text = ["This is a test"]
        pred1 = trained_model.predict(test_text)
        pred2 = trained_model.predict(test_text)
        
        assert np.array_equal(pred1, pred2)
    
    def test_empty_text_handling(self, trained_model):
        """Test handling of empty text."""
        empty_texts = [""]
        # Should not raise an error
        predictions = trained_model.predict(empty_texts)
        assert len(predictions) == 1
    
    def test_multiple_predictions(self, trained_model):
        """Test batch prediction."""
        test_texts = [
            "Great product",
            "Bad quality",
            "Okay product",
            "Excellent service",
            "Terrible experience"
        ]
        predictions = trained_model.predict(test_texts)
        
        assert len(predictions) == len(test_texts)
        assert isinstance(predictions, np.ndarray)
    
    def test_different_model_types(self, sample_data):
        """Test that all model types can be trained and predict."""
        X_train, y_train = sample_data
        test_texts = ["This is great"]
        
        for model_type in ['naive_bayes', 'logistic_regression']:
            model = SentimentModel(model_type=model_type)
            model.fit(X_train, y_train)
            predictions = model.predict(test_texts)
            
            assert len(predictions) == 1
            assert isinstance(predictions[0], (int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
