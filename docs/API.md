# API Documentation

## Overview

This document provides detailed API documentation for the Sentiment Analysis AI project.

## Table of Contents

- [Data Loading](#data-loading)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Utilities](#utilities)

---

## Data Loading

### `DataLoader`

Class for loading and splitting datasets.

#### Constructor

```python
DataLoader(random_seed: int = 42)
```

**Parameters:**
- `random_seed` (int): Random seed for reproducibility. Default is 42.

#### Methods

##### `load_csv()`

```python
load_csv(
    filepath: str,
    text_column: str = 'text',
    label_column: str = 'sentiment'
) -> Tuple[pd.Series, pd.Series]
```

Load data from a CSV file.

**Parameters:**
- `filepath` (str): Path to the CSV file
- `text_column` (str): Name of the text column. Default is 'text'
- `label_column` (str): Name of the label column. Default is 'sentiment'

**Returns:**
- Tuple of (texts, labels)

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `KeyError`: If specified columns are not found

##### `split_data()`

```python
split_data(
    X: pd.Series,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    stratify: bool = True
) -> Tuple
```

Split data into train, validation, and test sets.

**Parameters:**
- `X` (pd.Series): Feature data
- `y` (pd.Series): Labels
- `test_size` (float): Proportion of test set. Default is 0.2
- `val_size` (Optional[float]): Proportion of validation set
- `stratify` (bool): Whether to stratify split. Default is True

**Returns:**
- Tuple of split datasets

### Convenience Functions

##### `load_data()`

```python
load_data(
    filepath: str,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_seed: int = 42
) -> Tuple
```

Load and split data in one call.

---

## Preprocessing

### `TextPreprocessor`

Text preprocessing pipeline for sentiment analysis.

#### Constructor

```python
TextPreprocessor(
    lowercase: bool = True,
    remove_stopwords: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    lemmatize: bool = True
)
```

**Parameters:**
- `lowercase` (bool): Convert text to lowercase
- `remove_stopwords` (bool): Remove stopwords
- `remove_punctuation` (bool): Remove punctuation
- `remove_numbers` (bool): Remove numbers
- `lemmatize` (bool): Apply lemmatization

#### Methods

##### `preprocess()`

```python
preprocess(text: str) -> str
```

Apply full preprocessing pipeline to text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- Preprocessed text string

##### `preprocess_batch()`

```python
preprocess_batch(texts: Union[List[str], pd.Series]) -> List[str]
```

Preprocess a batch of texts.

**Parameters:**
- `texts` (Union[List[str], pd.Series]): List or Series of texts

**Returns:**
- List of preprocessed texts

---

## Model

### `SentimentModel`

Sentiment analysis model wrapper supporting multiple classifiers.

#### Constructor

```python
SentimentModel(
    model_type: str = 'naive_bayes',
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    **model_kwargs
)
```

**Parameters:**
- `model_type` (str): Type of model ('naive_bayes', 'logistic_regression', 'svm')
- `max_features` (int): Maximum number of features for TF-IDF
- `ngram_range` (tuple): N-gram range for TF-IDF
- `min_df` (int): Minimum document frequency
- `max_df` (float): Maximum document frequency
- `**model_kwargs`: Additional arguments for the classifier

#### Methods

##### `fit()`

```python
fit(X_train: list, y_train: np.ndarray) -> 'SentimentModel'
```

Fit the model on training data.

##### `predict()`

```python
predict(X: list) -> np.ndarray
```

Predict sentiments for input texts.

##### `predict_proba()`

```python
predict_proba(X: list) -> np.ndarray
```

Predict sentiment probabilities.

##### `get_top_features()`

```python
get_top_features(n: int = 20, class_idx: int = 1) -> list
```

Get top features for a specific class.

---

## Training

### `ModelTrainer`

Model trainer for sentiment analysis.

#### Constructor

```python
ModelTrainer(
    model: Optional[SentimentModel] = None,
    preprocessor: Optional[TextPreprocessor] = None
)
```

#### Methods

##### `train()`

```python
train(
    X_train: list,
    y_train: np.ndarray,
    preprocess: bool = True
) -> 'ModelTrainer'
```

Train the sentiment model.

##### `evaluate()`

```python
evaluate(
    X_test: list,
    y_test: np.ndarray,
    preprocess: bool = True
) -> Dict[str, Any]
```

Evaluate the trained model.

##### `save_model()`

```python
save_model(
    model_path: str,
    vectorizer_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None
) -> None
```

Save the trained model and components.

##### `load_model()` (static)

```python
@staticmethod
load_model(model_path: str) -> Dict[str, Any]
```

Load a saved model package.

---

## Evaluation

### `ModelEvaluator`

Model evaluator for sentiment analysis.

#### Methods

##### `evaluate()`

```python
evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict[str, Any]
```

Evaluate model predictions with comprehensive metrics.

**Returns:**
Dictionary containing:
- `accuracy`: Overall accuracy
- `precision_macro`: Macro-averaged precision
- `recall_macro`: Macro-averaged recall
- `f1_macro`: Macro-averaged F1-score
- `confusion_matrix`: Confusion matrix
- `classification_report`: Detailed per-class metrics
- `roc_auc`: ROC-AUC score (if probabilities provided)

##### `print_metrics()`

```python
print_metrics(metrics: Dict[str, Any]) -> None
```

Print evaluation metrics in a formatted way.

---

## Inference

### `SentimentPredictor`

Sentiment predictor for making predictions with trained models.

#### Constructor

```python
SentimentPredictor(model_path: str)
```

**Parameters:**
- `model_path` (str): Path to the saved model package

#### Methods

##### `predict()`

```python
predict(
    text: Union[str, List[str]],
    return_proba: bool = False,
    preprocess: bool = True
) -> Union[Dict[str, Any], List[Dict[str, Any]]]
```

Predict sentiment for input text(s).

**Parameters:**
- `text` (Union[str, List[str]]): Input text or list of texts
- `return_proba` (bool): Whether to return probabilities
- `preprocess` (bool): Whether to preprocess text

**Returns:**
For single text:
```python
{
    'text': str,
    'sentiment': int,
    'confidence': float,
    'sentiment_label': str,  # if available
    'probabilities': dict    # if return_proba=True
}
```

##### `get_model_info()`

```python
get_model_info() -> Dict[str, Any]
```

Get information about the loaded model.

### Utility Functions

##### `predict_from_file()`

```python
predict_from_file(
    model_path: str,
    input_file: str,
    output_file: str,
    text_column: str = 'text'
) -> None
```

Make predictions for texts in a CSV file and save results.

---

## Utilities

### Configuration

##### `load_config()`

```python
load_config(config_path: str = 'config.yaml') -> Dict[str, Any]
```

Load configuration from YAML file.

### Logging

##### `setup_logging()`

```python
setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger
```

Set up logging configuration.

### Visualization

##### `plot_confusion_matrix()`

```python
plot_confusion_matrix(
    cm: np.ndarray,
    classes: Optional[list] = None,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> None
```

Plot confusion matrix using seaborn heatmap.

---

## Example Usage

### Complete Pipeline

```python
from src.data_loader import load_data
from src.preprocessor import TextPreprocessor
from src.model import SentimentModel
from src.trainer import ModelTrainer
from src.inference import SentimentPredictor

# Load data
X_train, X_test, y_train, y_test = load_data('data/raw/reviews.csv')

# Initialize components
preprocessor = TextPreprocessor()
model = SentimentModel(model_type='naive_bayes')
trainer = ModelTrainer(model=model, preprocessor=preprocessor)

# Train
trainer.train(X_train, y_train)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Save
trainer.save_model('models/my_model.pkl', metrics=metrics)

# Load and predict
predictor = SentimentPredictor('models/my_model.pkl')
result = predictor.predict("This is amazing!", return_proba=True)
print(result)
```

---

## Error Handling

All functions include appropriate error handling:

- `FileNotFoundError`: Raised when files don't exist
- `KeyError`: Raised when required columns are missing
- `ValueError`: Raised for invalid parameter values
- `TypeError`: Raised for incorrect input types

Always wrap function calls in try-except blocks when appropriate:

```python
try:
    predictor = SentimentPredictor('models/model.pkl')
    result = predictor.predict("Test text")
except FileNotFoundError:
    print("Model file not found")
except Exception as e:
    print(f"Error: {e}")
```
