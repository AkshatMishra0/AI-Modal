"""
Performance benchmarking script for sentiment analysis model.

This script compares different configurations and measures performance metrics
including training time, inference time, memory usage, and accuracy.

Usage:
    python benchmark.py
    python benchmark.py --config config_optimized.yaml
    python benchmark.py --compare
"""

import argparse
import time
import psutil
import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data
from src.preprocessor import TextPreprocessor
from src.model import SentimentModel
from src.trainer import ModelTrainer
from src.utils import load_config


class PerformanceBenchmark:
    """Performance benchmarking for sentiment analysis models."""
    
    def __init__(self):
        """Initialize the benchmark."""
        self.results = []
        self.process = psutil.Process(os.getpid())
    
    def measure_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_preprocessing(
        self,
        texts: List[str],
        config: Dict[str, Any],
        name: str = "default"
    ) -> Dict[str, Any]:
        """
        Benchmark preprocessing performance.
        
        Args:
            texts (List[str]): Texts to preprocess.
            config (Dict[str, Any]): Preprocessing configuration.
            name (str): Benchmark name.
        
        Returns:
            Dict[str, Any]: Benchmark results.
        """
        print(f"\n📊 Benchmarking preprocessing: {name}")
        
        # Initialize preprocessor
        preprocess_config = config.get('preprocessing', {})
        preprocessor = TextPreprocessor(**preprocess_config)
        
        # Measure memory before
        mem_before = self.measure_memory()
        
        # Measure preprocessing time
        start_time = time.time()
        processed = preprocessor.preprocess_batch(texts)
        end_time = time.time()
        
        # Measure memory after
        mem_after = self.measure_memory()
        
        # Calculate metrics
        preprocessing_time = end_time - start_time
        memory_used = mem_after - mem_before
        throughput = len(texts) / preprocessing_time
        
        results = {
            'name': name,
            'phase': 'preprocessing',
            'num_texts': len(texts),
            'time_seconds': preprocessing_time,
            'throughput_texts_per_sec': throughput,
            'memory_mb': memory_used,
            'cache_size': preprocessor.get_cache_size() if hasattr(preprocessor, 'get_cache_size') else 0
        }
        
        print(f"  ⏱️  Time: {preprocessing_time:.2f}s")
        print(f"  🚀 Throughput: {throughput:.0f} texts/sec")
        print(f"  💾 Memory: {memory_used:.2f} MB")
        
        return results
    
    def benchmark_training(
        self,
        X_train: list,
        y_train: np.ndarray,
        config: Dict[str, Any],
        name: str = "default"
    ) -> Dict[str, Any]:
        """
        Benchmark model training performance.
        
        Args:
            X_train (list): Training texts.
            y_train (np.ndarray): Training labels.
            config (Dict[str, Any]): Model configuration.
            name (str): Benchmark name.
        
        Returns:
            Dict[str, Any]: Benchmark results.
        """
        print(f"\n📊 Benchmarking training: {name}")
        
        # Initialize components
        preprocess_config = config.get('preprocessing', {})
        model_config = config.get('model', {})
        model_type = model_config.pop('type', 'naive_bayes')
        
        # Handle ngram_range
        if 'ngram_range' in model_config:
            model_config['ngram_range'] = tuple(model_config['ngram_range'])
        
        preprocessor = TextPreprocessor(**preprocess_config)
        model = SentimentModel(model_type=model_type, **model_config)
        trainer = ModelTrainer(model=model, preprocessor=preprocessor)
        
        # Measure memory before
        mem_before = self.measure_memory()
        
        # Measure training time
        start_time = time.time()
        trainer.train(X_train, y_train)
        end_time = time.time()
        
        # Measure memory after
        mem_after = self.measure_memory()
        
        # Get model memory
        model_memory = model.get_memory_usage() if hasattr(model, 'get_memory_usage') else {}
        
        # Calculate metrics
        training_time = end_time - start_time
        memory_used = mem_after - mem_before
        
        results = {
            'name': name,
            'phase': 'training',
            'model_type': model_type,
            'num_samples': len(X_train),
            'time_seconds': training_time,
            'memory_mb': memory_used,
            'model_memory': model_memory
        }
        
        print(f"  ⏱️  Time: {training_time:.2f}s")
        print(f"  💾 Memory: {memory_used:.2f} MB")
        
        return results, trainer
    
    def benchmark_inference(
        self,
        trainer: ModelTrainer,
        X_test: list,
        name: str = "default"
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            trainer (ModelTrainer): Trained model.
            X_test (list): Test texts.
            name (str): Benchmark name.
        
        Returns:
            Dict[str, Any]: Benchmark results.
        """
        print(f"\n📊 Benchmarking inference: {name}")
        
        # Preprocess test data
        X_test_processed = trainer.preprocessor.preprocess_batch(X_test)
        
        # Measure prediction time
        start_time = time.time()
        predictions = trainer.model.predict(X_test_processed)
        end_time = time.time()
        
        # Calculate metrics
        inference_time = end_time - start_time
        throughput = len(X_test) / inference_time
        latency = inference_time / len(X_test) * 1000  # ms per prediction
        
        results = {
            'name': name,
            'phase': 'inference',
            'num_samples': len(X_test),
            'time_seconds': inference_time,
            'throughput_predictions_per_sec': throughput,
            'avg_latency_ms': latency
        }
        
        print(f"  ⏱️  Time: {inference_time:.2f}s")
        print(f"  🚀 Throughput: {throughput:.0f} predictions/sec")
        print(f"  ⚡ Latency: {latency:.2f} ms/prediction")
        
        return results
    
    def run_full_benchmark(
        self,
        data_path: str,
        config: Dict[str, Any],
        name: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Run complete benchmark including all phases.
        
        Args:
            data_path (str): Path to data file.
            config (Dict[str, Any]): Configuration.
            name (str): Benchmark name.
        
        Returns:
            List[Dict[str, Any]]: All benchmark results.
        """
        print(f"\n{'='*70}")
        print(f"🎯 Running Full Benchmark: {name}")
        print(f"{'='*70}")
        
        results_list = []
        
        # Load data
        print("\n📂 Loading data...")
        X_train, X_test, y_train, y_test = load_data(data_path, test_size=0.2)
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Benchmark preprocessing
        preprocess_results = self.benchmark_preprocessing(
            list(X_train), config, name
        )
        results_list.append(preprocess_results)
        
        # Benchmark training
        training_results, trainer = self.benchmark_training(
            X_train, y_train, config, name
        )
        results_list.append(training_results)
        
        # Benchmark inference
        inference_results = self.benchmark_inference(
            trainer, list(X_test), name
        )
        results_list.append(inference_results)
        
        # Evaluate accuracy
        print(f"\n📊 Evaluating accuracy...")
        metrics = trainer.evaluate(X_test, y_test)
        
        accuracy_results = {
            'name': name,
            'phase': 'evaluation',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1_score': metrics['f1_macro']
        }
        results_list.append(accuracy_results)
        
        print(f"  ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ✅ F1-Score: {metrics['f1_macro']:.4f}")
        
        return results_list


def compare_configurations():
    """Compare default and optimized configurations."""
    print("\n" + "="*70)
    print("🔍 CONFIGURATION COMPARISON")
    print("="*70)
    
    benchmark = PerformanceBenchmark()
    data_path = "data/raw/reviews.csv"
    
    # Default configuration
    default_config = {
        'preprocessing': {
            'lowercase': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
            'lemmatize': True,
            'use_cache': False,
            'use_parallel': False
        },
        'model': {
            'type': 'naive_bayes',
            'max_features': 5000,
            'ngram_range': [1, 2]
        }
    }
    
    # Optimized configuration
    optimized_config = {
        'preprocessing': {
            'lowercase': True,
            'remove_stopwords': True,
            'remove_punctuation': True,
            'lemmatize': True,
            'use_cache': True,
            'use_parallel': True,
            'max_workers': None
        },
        'model': {
            'type': 'logistic_regression',
            'max_features': 10000,
            'ngram_range': [1, 2],
            'use_idf': True,
            'sublinear_tf': True
        }
    }
    
    # Run benchmarks
    default_results = benchmark.run_full_benchmark(data_path, default_config, "Default")
    optimized_results = benchmark.run_full_benchmark(data_path, optimized_config, "Optimized")
    
    # Compare results
    print("\n" + "="*70)
    print("📈 COMPARISON SUMMARY")
    print("="*70)
    
    all_results = default_results + optimized_results
    df = pd.DataFrame(all_results)
    
    # Save results
    output_file = "benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print comparison
    print("\nPerformance Improvements:")
    phases = df['phase'].unique()
    for phase in phases:
        default_row = df[(df['name'] == 'Default') & (df['phase'] == phase)]
        optimized_row = df[(df['name'] == 'Optimized') & (df['phase'] == phase)]
        
        if not default_row.empty and not optimized_row.empty:
            print(f"\n{phase.upper()}:")
            if 'time_seconds' in default_row.columns:
                default_time = default_row['time_seconds'].values[0]
                optimized_time = optimized_row['time_seconds'].values[0]
                speedup = (default_time / optimized_time - 1) * 100
                print(f"  ⚡ Speedup: {speedup:+.1f}%")
            
            if 'accuracy' in default_row.columns:
                default_acc = default_row['accuracy'].values[0]
                optimized_acc = optimized_row['accuracy'].values[0]
                improvement = (optimized_acc - default_acc) * 100
                print(f"  📊 Accuracy Change: {improvement:+.2f}%")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark sentiment analysis performance')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--compare', action='store_true', help='Compare default vs optimized')
    parser.add_argument('--data', type=str, default='data/raw/reviews.csv', help='Path to data file')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations()
    else:
        # Single benchmark
        if args.config:
            config = load_config(args.config)
            name = Path(args.config).stem
        else:
            # Use default
            config = load_config('config.yaml')
            name = "default"
        
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark(args.data, config, name)
        
        # Save results
        df = pd.DataFrame(results)
        output_file = f"benchmark_{name}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
