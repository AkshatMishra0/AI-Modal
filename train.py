"""
Training script for sentiment analysis model.

This script trains a sentiment analysis model on the provided dataset,
evaluates it, and saves the trained model.

Usage:
    python train.py [--config CONFIG_PATH]
    python train.py --config custom_config.yaml
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data
from src.preprocessor import TextPreprocessor
from src.model import SentimentModel
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.utils import load_config, setup_logging, plot_confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train sentiment analysis model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data CSV file (overrides config)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        choices=['naive_bayes', 'logistic_regression', 'svm'],
        help='Model type (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save trained model (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"✓ Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {args.config}")
        print("  Using default configuration...")
        config = {
            'data': {
                'raw_data_path': 'data/raw/reviews.csv',
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'random_seed': 42
            },
            'model': {
                'type': 'naive_bayes',
                'max_features': 5000,
                'ngram_range': [1, 2],
                'min_df': 2,
                'max_df': 0.95
            },
            'training': {
                'save_model_path': 'models/sentiment_model.pkl'
            },
            'preprocessing': {
                'lowercase': True,
                'remove_stopwords': True,
                'remove_punctuation': True,
                'remove_numbers': False,
                'lemmatize': True
            }
        }
    
    # Override config with command line arguments
    if args.data:
        config['data']['raw_data_path'] = args.data
    if args.model_type:
        config['model']['type'] = args.model_type
    if args.output:
        config['training']['save_model_path'] = args.output
    
    # Setup logging
    logger = setup_logging(level='INFO')
    
    print("\n" + "="*70)
    print(" "*20 + "SENTIMENT ANALYSIS TRAINING")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading data...")
    try:
        data_path = config['data']['raw_data_path']
        test_size = config['data'].get('test_split', 0.1)
        val_size = config['data'].get('val_split', 0.1)
        random_seed = config['data'].get('random_seed', 42)
        
        if val_size > 0:
            X_train, X_val, X_test, y_train, y_val, y_test = load_data(
                data_path,
                test_size=test_size,
                val_size=val_size,
                random_seed=random_seed
            )
            print(f"✓ Data loaded successfully!")
            print(f"  - Training samples:   {len(X_train)}")
            print(f"  - Validation samples: {len(X_val)}")
            print(f"  - Test samples:       {len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = load_data(
                data_path,
                test_size=test_size,
                random_seed=random_seed
            )
            print(f"✓ Data loaded successfully!")
            print(f"  - Training samples: {len(X_train)}")
            print(f"  - Test samples:     {len(X_test)}")
            
    except FileNotFoundError:
        print(f"✗ Data file not found: {data_path}")
        print("  Please ensure the data file exists or update the config.")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Initialize preprocessor
    print("\n🔧 Initializing preprocessor...")
    preprocess_config = config.get('preprocessing', {})
    preprocessor = TextPreprocessor(**preprocess_config)
    print("✓ Preprocessor initialized")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model_config = config.get('model', {})
    model_type = model_config.pop('type', 'naive_bayes')
    
    # Convert ngram_range list to tuple
    if 'ngram_range' in model_config:
        model_config['ngram_range'] = tuple(model_config['ngram_range'])
    
    model = SentimentModel(model_type=model_type, **model_config)
    print(f"✓ Model initialized: {model_type}")
    
    # Initialize trainer
    trainer = ModelTrainer(model=model, preprocessor=preprocessor)
    
    # Train model
    print("\n🚀 Training model...")
    trainer.train(X_train, y_train)
    print("✓ Training completed!")
    
    # Evaluate on validation set if available
    if val_size > 0:
        print("\n📊 Evaluating on validation set...")
        val_metrics = trainer.evaluate(X_val, y_val)
        print(f"\nValidation Results:")
        print(f"  - Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  - Precision: {val_metrics['precision_macro']:.4f}")
        print(f"  - Recall:    {val_metrics['recall_macro']:.4f}")
        print(f"  - F1-Score:  {val_metrics['f1_macro']:.4f}")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    
    evaluator = ModelEvaluator()
    evaluator.print_metrics(test_metrics)
    
    # Plot confusion matrix
    try:
        cm_path = Path(config['training']['save_model_path']).parent / 'confusion_matrix.png'
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            classes=['Negative', 'Neutral', 'Positive'],
            save_path=str(cm_path)
        )
    except Exception as e:
        logger.warning(f"Could not save confusion matrix plot: {e}")
    
    # Save model
    print("\n💾 Saving model...")
    model_path = config['training']['save_model_path']
    trainer.save_model(model_path, metrics=test_metrics)
    print(f"✓ Model saved to: {model_path}")
    
    # Display feature importance
    print("\n🔍 Top predictive features:")
    try:
        feature_importance = trainer.get_feature_importance(n=10)
        for class_idx, features in feature_importance.items():
            class_name = ['Negative', 'Neutral', 'Positive'][class_idx] if class_idx < 3 else f'Class {class_idx}'
            print(f"\n  {class_name}:")
            for i, (feature, score) in enumerate(features[:5], 1):
                print(f"    {i}. {feature:20s} ({score:.4f})")
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    print("\n" + "="*70)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    print("📝 Next steps:")
    print(f"  1. Review metrics in: {Path(model_path).parent / 'metrics.json'}")
    print(f"  2. Make predictions: python predict.py --text 'Your text here'")
    print(f"  3. Check notebook: notebooks/demo.ipynb")
    print()


if __name__ == "__main__":
    main()
