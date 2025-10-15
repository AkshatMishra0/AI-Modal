"""
Utility functions for the sentiment analysis project.

This module contains helper functions for logging, configuration,
and other common tasks.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file. Default is 'config.yaml'.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data (Dict[str, Any]): Data to save.
        filepath (str): Path to save file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath (str): Path to JSON file.
    
    Returns:
        Dict[str, Any]: Loaded data.
    
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level (str): Logging level. Default is 'INFO'.
        log_file (Optional[str]): Path to log file. If None, logs to console only.
        format_string (Optional[str]): Custom format string.
    
    Returns:
        logging.Logger: Configured logger.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Optional[list] = None,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix using seaborn heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix.
        classes (Optional[list]): Class labels. If None, uses numeric labels.
        title (str): Plot title. Default is 'Confusion Matrix'.
        cmap (str): Color map. Default is 'Blues'.
        save_path (Optional[str]): Path to save figure. If None, displays plot.
    """
    plt.figure(figsize=(10, 8))
    
    if classes is None:
        classes = [str(i) for i in range(len(cm))]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[list] = None,
    title: str = 'Model Metrics Comparison',
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of metrics across models or configurations.
    
    Args:
        metrics_dict (Dict[str, Dict[str, float]]): Dictionary of model names to metrics.
        metric_names (Optional[list]): List of metric names to plot. If None, plots all.
        title (str): Plot title.
        save_path (Optional[str]): Path to save figure.
    """
    if metric_names is None:
        # Get all metric names from first model
        first_model = list(metrics_dict.values())[0]
        metric_names = [k for k in first_model.keys() if isinstance(first_model[k], (int, float))]
    
    models = list(metrics_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_directory_structure(base_path: str = '.') -> None:
    """
    Create the project directory structure.
    
    Args:
        base_path (str): Base path for the project. Default is current directory.
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'tests',
        'docs',
        'src',
        'logs'
    ]
    
    base_path = Path(base_path)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep files
        gitkeep = dir_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    print(f"Directory structure created at {base_path}")


if __name__ == "__main__":
    # Example usage
    
    # Load config
    try:
        config = load_config('config.yaml')
        print("Config loaded successfully")
        print(f"Model type: {config.get('model', {}).get('type')}")
    except FileNotFoundError:
        print("Config file not found")
    
    # Setup logging
    logger = setup_logging(level='INFO')
    logger.info("This is a test log message")
    
    # Plot example confusion matrix
    cm_example = np.array([[50, 5, 2], [3, 45, 4], [1, 6, 48]])
    plot_confusion_matrix(
        cm_example,
        classes=['Positive', 'Negative', 'Neutral'],
        title='Example Confusion Matrix'
    )
