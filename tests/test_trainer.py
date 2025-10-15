"""
Unit tests for the ModelTrainer class.

Run with: pytest tests/test_trainer.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer import ModelTrainer
from src.model import SentimentModel
from src.preprocessor import TextPreprocessor


class TestModelTrainer:
    """Test cases for ModelTrainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X_train = [
            "This is great and amazing",
            "Terrible and awful product",
            "It is okay and acceptable",
            "Excellent and fantastic",
            "Bad and disappointing",
            "Average and normal",
            "Wonderful and superb",
            "Poor and unsatisfactory"
        ]
        y_train = np.array([2, 0, 1, 2, 0, 1, 2, 0])
        
        X_test = [
            "Great product",
            "Bad quality",
            "Okay service"
        ]
        y_test = np.array([2, 0, 1])
        
        return X_train, y_train, X_test, y_test
    
    @pytest.fixture
    def trainer(self):
        """Create a trainer instance."""
        model = SentimentModel(model_type='naive_bayes', max_features=100)
        preprocessor = TextPreprocessor()
        return ModelTrainer(model=model, preprocessor=preprocessor)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer()
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.preprocessor is not None
        assert trainer.evaluator is not None
    
    def test_trainer_with_custom_components(self):
        """Test trainer with custom model and preprocessor."""
        model = SentimentModel(model_type='logistic_regression')
        preprocessor = TextPreprocessor(lowercase=False)
        trainer = ModelTrainer(model=model, preprocessor=preprocessor)
        
        assert trainer.model.model_type == 'logistic_regression'
        assert trainer.preprocessor.lowercase is False
    
    def test_train_method(self, trainer, sample_data):
        """Test training method."""
        X_train, y_train, _, _ = sample_data
        
        result = trainer.train(X_train, y_train, preprocess=True)
        
        # Should return self
        assert result is trainer
        
        # Model should be trained
        assert hasattr(trainer.model.classifier, 'classes_')
    
    def test_train_without_preprocessing(self, trainer, sample_data):
        """Test training without preprocessing."""
        X_train, y_train, _, _ = sample_data
        
        # Pre-preprocess the data
        X_train_processed = trainer.preprocessor.preprocess_batch(X_train)
        
        # Train without additional preprocessing
        trainer.train(X_train_processed, y_train, preprocess=False)
        
        assert hasattr(trainer.model.classifier, 'classes_')
    
    def test_evaluate_method(self, trainer, sample_data):
        """Test evaluation method."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Train first
        trainer.train(X_train, y_train)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
        
        # Metrics should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_save_and_load_model(self, trainer, sample_data):
        """Test model saving and loading."""
        X_train, y_train, _, _ = sample_data
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_package = ModelTrainer.load_model(model_path)
            
            assert 'model' in loaded_package
            assert 'preprocessor' in loaded_package
            assert 'evaluator' in loaded_package
            
            # Test loaded model works
            test_text = ["This is great"]
            prediction = loaded_package['model'].predict(
                loaded_package['preprocessor'].preprocess_batch(test_text)
            )
            assert len(prediction) == 1
            
        finally:
            # Cleanup
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_save_model_with_metrics(self, trainer, sample_data):
        """Test saving model with metrics."""
        X_train, y_train, X_test, y_test = sample_data
        
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'model.pkl')
            
            trainer.save_model(model_path, metrics=metrics)
            
            # Check that metrics file was created
            metrics_path = os.path.join(tmp_dir, 'metrics.json')
            assert os.path.exists(metrics_path)
    
    def test_get_feature_importance(self, trainer, sample_data):
        """Test feature importance extraction."""
        X_train, y_train, _, _ = sample_data
        
        trainer.train(X_train, y_train)
        
        feature_importance = trainer.get_feature_importance(n=5)
        
        assert isinstance(feature_importance, dict)
        
        # Should have importance for each class
        for class_idx, features in feature_importance.items():
            assert isinstance(features, list)
            assert len(features) <= 5
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ModelTrainer.load_model('nonexistent_model.pkl')
    
    def test_training_with_small_dataset(self, trainer):
        """Test training with very small dataset."""
        X_train = ["good", "bad"]
        y_train = np.array([1, 0])
        
        # Should not crash
        trainer.train(X_train, y_train)
        
        # Should be able to make predictions
        predictions = trainer.model.predict(["test"])
        assert len(predictions) == 1
    
    def test_evaluation_metrics_range(self, trainer, sample_data):
        """Test that all metrics are in valid range."""
        X_train, y_train, X_test, y_test = sample_data
        
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        # Check metric ranges
        for metric_name in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            assert 0 <= metrics[metric_name] <= 1, f"{metric_name} out of range"
    
    def test_train_and_evaluate_pipeline(self, trainer, sample_data):
        """Test complete train-evaluate pipeline."""
        X_train, y_train, X_test, y_test = sample_data
        
        # Complete pipeline
        trainer.train(X_train, y_train)
        metrics = trainer.evaluate(X_test, y_test)
        
        # Verify we can make predictions
        predictions = trainer.model.predict(
            trainer.preprocessor.preprocess_batch(X_test)
        )
        
        assert len(predictions) == len(X_test)
        assert metrics['accuracy'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
