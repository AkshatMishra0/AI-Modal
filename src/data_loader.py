"""
Data loading utilities for sentiment analysis.

This module provides functions to load and split data for training and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader class for handling dataset loading and splitting.
    
    Attributes:
        random_seed (int): Random seed for reproducibility.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            random_seed (int): Random seed for reproducibility. Default is 42.
        """
        self.random_seed = random_seed
        logger.info(f"DataLoader initialized with random_seed={random_seed}")
    
    def load_csv(
        self, 
        filepath: str, 
        text_column: str = 'text', 
        label_column: str = 'sentiment'
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            text_column (str): Name of the text column. Default is 'text'.
            label_column (str): Name of the label column. Default is 'sentiment'.
        
        Returns:
            Tuple[pd.Series, pd.Series]: Text data and labels.
        
        Raises:
            FileNotFoundError: If the file doesn't exist.
            KeyError: If specified columns are not found.
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Validate columns
            if text_column not in df.columns:
                raise KeyError(f"Column '{text_column}' not found in CSV")
            if label_column not in df.columns:
                raise KeyError(f"Column '{label_column}' not found in CSV")
            
            # Remove missing values
            df = df.dropna(subset=[text_column, label_column])
            
            logger.info(f"Loaded {len(df)} samples")
            return df[text_column], df[label_column]
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def split_data(
        self,
        X: pd.Series,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify: bool = True
    ) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (pd.Series): Feature data.
            y (pd.Series): Labels.
            test_size (float): Proportion of test set. Default is 0.2.
            val_size (Optional[float]): Proportion of validation set. If None, no validation set.
            stratify (bool): Whether to stratify split by labels. Default is True.
        
        Returns:
            Tuple: Split datasets (X_train, X_test, y_train, y_test) or
                   (X_train, X_val, X_test, y_train, y_val, y_test) if val_size is specified.
        """
        stratify_split = y if stratify else None
        
        if val_size is None:
            # Simple train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=stratify_split
            )
            logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
            return X_train, X_test, y_train, y_test
        else:
            # Train-val-test split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=(test_size + val_size),
                random_state=self.random_seed,
                stratify=stratify_split
            )
            
            val_ratio = val_size / (test_size + val_size)
            stratify_temp = y_temp if stratify else None
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_ratio),
                random_state=self.random_seed,
                stratify=stratify_temp
            )
            
            logger.info(
                f"Data split: train={len(X_train)}, "
                f"val={len(X_val)}, test={len(X_test)}"
            )
            return X_train, X_val, X_test, y_train, y_val, y_test


def load_data(
    filepath: str,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_seed: int = 42
) -> Tuple:
    """
    Convenience function to load and split data in one call.
    
    Args:
        filepath (str): Path to the CSV file.
        test_size (float): Proportion of test set. Default is 0.2.
        val_size (Optional[float]): Proportion of validation set.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        Tuple: Split datasets.
    
    Example:
        >>> X_train, X_test, y_train, y_test = load_data('data/reviews.csv')
    """
    loader = DataLoader(random_seed=random_seed)
    X, y = loader.load_csv(filepath)
    return loader.split_data(X, y, test_size=test_size, val_size=val_size)


if __name__ == "__main__":
    # Example usage
    try:
        X_train, X_test, y_train, y_test = load_data('data/raw/reviews.csv')
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
    except Exception as e:
        print(f"Error: {e}")
