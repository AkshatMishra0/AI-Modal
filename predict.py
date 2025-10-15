"""
Prediction script for sentiment analysis.

This script loads a trained model and makes predictions on new text data.

Usage:
    python predict.py --text "Your text here"
    python predict.py --file input.csv --output predictions.csv
    python predict.py --text "Great product!" --model models/sentiment_model.pkl
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import SentimentPredictor, predict_from_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make sentiment predictions with trained model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/sentiment_model.pkl',
        help='Path to trained model file (default: models/sentiment_model.pkl)'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to analyze (for single prediction)'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='CSV file with texts to analyze (for batch prediction)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for batch predictions (required if --file is used)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of text column in input CSV (default: text)'
    )
    parser.add_argument(
        '--show-proba',
        action='store_true',
        help='Show probability scores for each class'
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print(" "*20 + "SENTIMENT ANALYSIS PREDICTION")
    print("="*70 + "\n")
    
    # Load model
    print(f"📂 Loading model from {args.model}...")
    try:
        predictor = SentimentPredictor(args.model)
        print("✓ Model loaded successfully!")
        
        # Display model info
        info = predictor.get_model_info()
        print(f"\nModel Information:")
        print(f"  - Type: {info['model_type']}")
        print(f"  - Classes: {info['classes']}")
        print(f"  - Features: {info['n_features']}")
        
    except FileNotFoundError:
        print(f"✗ Model file not found: {args.model}")
        print("\n💡 Tip: Train a model first using:")
        print("   python train.py")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Single text prediction
    if args.text:
        print(f"\n🔮 Analyzing text...\n")
        result = predictor.predict(args.text, return_proba=args.show_proba)
        
        print("-" * 70)
        print(f"Text: {result['text']}")
        print("-" * 70)
        
        sentiment_label = result.get('sentiment_label', f"Class {result['sentiment']}")
        print(f"\n✨ Predicted Sentiment: {sentiment_label}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        
        if args.show_proba and 'probabilities' in result:
            print(f"\nProbability Distribution:")
            for label, prob in result['probabilities'].items():
                bar_length = int(prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"  {label:10s} [{bar}] {prob:.2%}")
        
        print()
    
    # Batch file prediction
    elif args.file:
        if not args.output:
            print("✗ Error: --output is required when using --file")
            print("\n💡 Usage:")
            print("   python predict.py --file input.csv --output predictions.csv")
            return
        
        print(f"\n🔮 Processing file: {args.file}...")
        
        try:
            predict_from_file(
                args.model,
                args.file,
                args.output,
                text_column=args.text_column
            )
            print(f"✓ Predictions saved to: {args.output}")
            
        except FileNotFoundError:
            print(f"✗ Input file not found: {args.file}")
            return
        except KeyError as e:
            print(f"✗ Column error: {e}")
            print(f"   Make sure your CSV has a '{args.text_column}' column")
            return
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            return
    
    # No input provided
    else:
        print("✗ Error: Please provide either --text or --file\n")
        print("💡 Examples:")
        print("   Single prediction:")
        print('     python predict.py --text "This is amazing!"')
        print()
        print("   Batch prediction:")
        print("     python predict.py --file reviews.csv --output predictions.csv")
        print()
        print("   With probabilities:")
        print('     python predict.py --text "Great product!" --show-proba')
        print()
        return
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
