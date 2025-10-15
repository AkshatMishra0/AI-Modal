"""
Inference module for making predictions with trained models.

This module provides easy-to-use interfaces for making predictions
with trained sentiment analysis models.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Sentiment predictor for making predictions with trained models.
    
    Provides a simple interface for loading models and making predictions.
    
    Attributes:
        model_package (Dict): Loaded model package containing model, preprocessor, etc.
        model: The trained sentiment model.
        preprocessor: The text preprocessor.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the sentiment predictor.
        
        Args:
            model_path (str): Path to the saved model package.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        self.model_package = joblib.load(model_path)
        
        self.model = self.model_package['model']
        self.preprocessor = self.model_package['preprocessor']
        
        # Get class names if available
        if hasattr(self.model.classifier, 'classes_'):
            self.classes = self.model.classifier.classes_
        else:
            self.classes = None
        
        logger.info("Model loaded successfully")
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_proba: bool = False,
        preprocess: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict sentiment for input text(s).
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts.
            return_proba (bool): Whether to return probabilities. Default is False.
            preprocess (bool): Whether to preprocess text. Default is True.
        
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Prediction results.
                For single text: {'text': str, 'sentiment': int, 'confidence': float, ...}
                For multiple texts: List of prediction dictionaries.
        """
        # Handle single text
        if isinstance(text, str):
            return self._predict_single(text, return_proba, preprocess)
        
        # Handle multiple texts
        elif isinstance(text, list):
            return self._predict_batch(text, return_proba, preprocess)
        
        else:
            raise TypeError("Input must be a string or list of strings")
    
    def _predict_single(
        self,
        text: str,
        return_proba: bool = False,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text.
            return_proba (bool): Whether to return probabilities.
            preprocess (bool): Whether to preprocess text.
        
        Returns:
            Dict[str, Any]: Prediction result.
        """
        # Preprocess if needed
        if preprocess:
            processed_text = self.preprocessor.preprocess(text)
        else:
            processed_text = text
        
        # Make prediction
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]
        
        # Build result
        result = {
            'text': text,
            'processed_text': processed_text if preprocess else None,
            'sentiment': int(prediction),
            'confidence': float(np.max(probabilities))
        }
        
        # Add class name if available
        if self.classes is not None:
            result['sentiment_label'] = str(self.classes[prediction])
        
        # Add probabilities if requested
        if return_proba:
            if self.classes is not None:
                result['probabilities'] = {
                    str(self.classes[i]): float(prob)
                    for i, prob in enumerate(probabilities)
                }
            else:
                result['probabilities'] = {
                    f'class_{i}': float(prob)
                    for i, prob in enumerate(probabilities)
                }
        
        return result
    
    def _predict_batch(
        self,
        texts: List[str],
        return_proba: bool = False,
        preprocess: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict sentiments for multiple texts.
        
        Args:
            texts (List[str]): List of input texts.
            return_proba (bool): Whether to return probabilities.
            preprocess (bool): Whether to preprocess texts.
        
        Returns:
            List[Dict[str, Any]]: List of prediction results.
        """
        logger.info(f"Predicting for {len(texts)} texts")
        
        # Preprocess if needed
        if preprocess:
            processed_texts = self.preprocessor.preprocess_batch(texts)
        else:
            processed_texts = texts
        
        # Make predictions
        predictions = self.model.predict(processed_texts)
        probabilities = self.model.predict_proba(processed_texts)
        
        # Build results
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'processed_text': processed_texts[i] if preprocess else None,
                'sentiment': int(predictions[i]),
                'confidence': float(np.max(probabilities[i]))
            }
            
            # Add class name if available
            if self.classes is not None:
                result['sentiment_label'] = str(self.classes[predictions[i]])
            
            # Add probabilities if requested
            if return_proba:
                if self.classes is not None:
                    result['probabilities'] = {
                        str(self.classes[j]): float(prob)
                        for j, prob in enumerate(probabilities[i])
                    }
                else:
                    result['probabilities'] = {
                        f'class_{j}': float(prob)
                        for j, prob in enumerate(probabilities[i])
                    }
            
            results.append(result)
        
        return results
    
    def predict_sentiment_label(self, text: str) -> str:
        """
        Predict sentiment label for text (convenience method).
        
        Args:
            text (str): Input text.
        
        Returns:
            str: Sentiment label.
        """
        result = self.predict(text)
        return result.get('sentiment_label', str(result['sentiment']))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information.
        """
        info = {
            'model_type': self.model.model_type,
            'classes': self.classes.tolist() if self.classes is not None else None,
            'n_features': len(self.model.get_feature_names()),
        }
        
        return info


def predict_from_file(
    model_path: str,
    input_file: str,
    output_file: str,
    text_column: str = 'text'
) -> None:
    """
    Make predictions for texts in a CSV file and save results.
    
    Args:
        model_path (str): Path to the saved model.
        input_file (str): Path to input CSV file.
        output_file (str): Path to output CSV file.
        text_column (str): Name of the text column. Default is 'text'.
    """
    import pandas as pd
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if text column exists
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in {input_file}")
    
    # Initialize predictor
    predictor = SentimentPredictor(model_path)
    
    # Make predictions
    results = predictor.predict(df[text_column].tolist(), return_proba=True)
    
    # Add predictions to dataframe
    df['predicted_sentiment'] = [r['sentiment'] for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    
    if 'sentiment_label' in results[0]:
        df['sentiment_label'] = [r['sentiment_label'] for r in results]
    
    # Save results
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize predictor
        predictor = SentimentPredictor('models/sentiment_model.pkl')
        
        # Single prediction
        text = "This product is amazing! I love it!"
        result = predictor.predict(text, return_proba=True)
        
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        if 'probabilities' in result:
            print(f"Probabilities: {result['probabilities']}")
        
        # Batch prediction
        texts = [
            "Great product!",
            "Terrible experience.",
            "It's okay, nothing special."
        ]
        results = predictor.predict(texts)
        
        print("\nBatch predictions:")
        for r in results:
            print(f"  {r['text'][:30]:30s} -> {r['sentiment']} ({r['confidence']:.2%})")
        
    except FileNotFoundError:
        logger.error("Model file not found. Please train a model first.")
