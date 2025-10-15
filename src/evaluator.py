"""
Model evaluation module.

This module provides comprehensive evaluation metrics and visualizations
for sentiment analysis models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluator for sentiment analysis.
    
    Provides comprehensive metrics including accuracy, precision, recall,
    F1-score, confusion matrix, and ROC-AUC.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        logger.info("ModelEvaluator initialized")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'macro'
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions with comprehensive metrics.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            y_proba (Optional[np.ndarray]): Predicted probabilities.
            average (str): Averaging method for multi-class metrics. Default is 'macro'.
        
        Returns:
            Dict[str, Any]: Dictionary containing all evaluation metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        
        # ROC-AUC (if probabilities provided)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='macro'
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics (Dict[str, Any]): Dictionary of metrics.
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        # Basic metrics
        print(f"\nAccuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):   {metrics['f1_macro']:.4f}")
        
        if metrics.get('roc_auc') is not None:
            print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        # Classification report
        print("\nClassification Report:")
        report = metrics['classification_report']
        for label, scores in report.items():
            if isinstance(scores, dict):
                print(f"\nClass {label}:")
                print(f"  Precision: {scores['precision']:.4f}")
                print(f"  Recall:    {scores['recall']:.4f}")
                print(f"  F1-Score:  {scores['f1-score']:.4f}")
                print(f"  Support:   {scores['support']}")
        
        print("\n" + "="*60 + "\n")
    
    def get_confusion_matrix_stats(self, cm: np.ndarray) -> Dict[str, Any]:
        """
        Extract statistics from confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix.
        
        Returns:
            Dict[str, Any]: Dictionary of confusion matrix statistics.
        """
        stats = {
            'total_samples': np.sum(cm),
            'correct_predictions': np.trace(cm),
            'incorrect_predictions': np.sum(cm) - np.trace(cm)
        }
        
        # Per-class statistics
        for i in range(len(cm)):
            stats[f'class_{i}_total'] = np.sum(cm[i, :])
            stats[f'class_{i}_correct'] = cm[i, i]
            stats[f'class_{i}_accuracy'] = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        
        return stats
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'accuracy'
    ) -> None:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            results (Dict[str, Dict[str, Any]]): Dictionary of model names to metrics.
            metric (str): Metric to compare. Default is 'accuracy'.
        """
        print("\n" + "="*60)
        print(f"MODEL COMPARISON - {metric.upper()}")
        print("="*60 + "\n")
        
        # Sort models by metric
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            value = metrics.get(metric, 'N/A')
            if isinstance(value, float):
                print(f"{rank}. {model_name:30s} {value:.4f}")
            else:
                print(f"{rank}. {model_name:30s} {value}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # Simulate predictions
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    y_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.85, 0.1, 0.05],
        [0.2, 0.3, 0.5],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.1, 0.85, 0.05],
        [0.2, 0.6, 0.2]
    ])
    
    # Evaluate
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)
    evaluator.print_metrics(metrics)
