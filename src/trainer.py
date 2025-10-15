"""
Model training module.

This module handles the training process including preprocessing,
model fitting, and saving.
"""

import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import numpy as np
from src.model import SentimentModel
from src.preprocessor import TextPreprocessor
from src.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model trainer for sentiment analysis.
    
    Handles the complete training pipeline including preprocessing,
    training, evaluation, and model persistence.
    
    Attributes:
        model (SentimentModel): The sentiment model to train.
        preprocessor (TextPreprocessor): Text preprocessor.
        evaluator (ModelEvaluator): Model evaluator.
    """
    
    def __init__(
        self,
        model: Optional[SentimentModel] = None,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            model (Optional[SentimentModel]): Sentiment model. If None, creates default.
            preprocessor (Optional[TextPreprocessor]): Preprocessor. If None, creates default.
        """
        self.model = model if model is not None else SentimentModel()
        self.preprocessor = preprocessor if preprocessor is not None else TextPreprocessor()
        self.evaluator = ModelEvaluator()
        
        logger.info("ModelTrainer initialized")
    
    def train(
        self,
        X_train: list,
        y_train: np.ndarray,
        preprocess: bool = True
    ) -> 'ModelTrainer':
        """
        Train the sentiment model.
        
        Args:
            X_train (list): Training texts.
            y_train (np.ndarray): Training labels.
            preprocess (bool): Whether to preprocess texts. Default is True.
        
        Returns:
            ModelTrainer: The trainer instance (self).
        """
        logger.info(f"Starting training on {len(X_train)} samples")
        
        # Preprocess if needed
        if preprocess:
            logger.info("Preprocessing training data")
            X_train = self.preprocessor.preprocess_batch(X_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        logger.info("Training completed")
        return self
    
    def evaluate(
        self,
        X_test: list,
        y_test: np.ndarray,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test (list): Test texts.
            y_test (np.ndarray): Test labels.
            preprocess (bool): Whether to preprocess texts. Default is True.
        
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating on {len(X_test)} test samples")
        
        # Preprocess if needed
        if preprocess:
            X_test = self.preprocessor.preprocess_batch(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.evaluator.evaluate(y_test, y_pred, y_proba)
        
        # Log metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision_macro']:.4f}")
        logger.info(f"Recall: {metrics['recall_macro']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def save_model(
        self,
        model_path: str,
        vectorizer_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the trained model and components.
        
        Args:
            model_path (str): Path to save the complete model.
            vectorizer_path (Optional[str]): Path to save vectorizer separately.
            preprocessor_path (Optional[str]): Path to save preprocessor separately.
            metrics (Optional[Dict[str, Any]]): Metrics to save.
        """
        logger.info(f"Saving model to {model_path}")
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save complete model package
        model_package = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'evaluator': self.evaluator
        }
        joblib.dump(model_package, model_path)
        
        # Save vectorizer separately if path provided
        if vectorizer_path:
            joblib.dump(self.model.vectorizer, vectorizer_path)
            logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        # Save preprocessor separately if path provided
        if preprocessor_path:
            joblib.dump(self.preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save metrics if provided
        if metrics:
            metrics_path = str(Path(model_path).parent / 'metrics.json')
            # Convert numpy types to native Python types
            serializable_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
                if k != 'confusion_matrix'  # Skip confusion matrix for JSON
            }
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")
        
        logger.info("Model saved successfully")
    
    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """
        Load a saved model package.
        
        Args:
            model_path (str): Path to the saved model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model, preprocessor, and evaluator.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        logger.info(f"Loading model from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_package = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        return model_package
    
    def get_feature_importance(self, n: int = 20) -> Dict[int, list]:
        """
        Get top features for each class.
        
        Args:
            n (int): Number of top features per class. Default is 20.
        
        Returns:
            Dict[int, list]: Dictionary mapping class index to top features.
        """
        feature_importance = {}
        
        # Get number of classes
        if hasattr(self.model.classifier, 'classes_'):
            n_classes = len(self.model.classifier.classes_)
            
            for i in range(n_classes):
                top_features = self.model.get_top_features(n=n, class_idx=i)
                feature_importance[i] = top_features
                
                logger.info(f"Top features for class {i}:")
                for feature, score in top_features[:5]:
                    logger.info(f"  {feature}: {score:.4f}")
        
        return feature_importance


if __name__ == "__main__":
    # Example usage
    from src.data_loader import load_data
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data('data/raw/reviews.csv')
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        
        # Save model
        trainer.save_model('models/sentiment_model.pkl', metrics=metrics)
        
    except Exception as e:
        logger.error(f"Error: {e}")
