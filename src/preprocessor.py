"""
Text preprocessing module for sentiment analysis.

This module handles all text preprocessing tasks including cleaning,
tokenization, and normalization.
"""

import re
import string
from typing import List, Union
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing pipeline for sentiment analysis.
    
    Handles text cleaning, tokenization, stopword removal, and lemmatization.
    
    Attributes:
        lowercase (bool): Whether to convert text to lowercase.
        remove_stopwords (bool): Whether to remove stopwords.
        remove_punctuation (bool): Whether to remove punctuation.
        remove_numbers (bool): Whether to remove numbers.
        lemmatize (bool): Whether to apply lemmatization.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        lemmatize: bool = True
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase (bool): Convert text to lowercase. Default is True.
            remove_stopwords (bool): Remove stopwords. Default is True.
            remove_punctuation (bool): Remove punctuation. Default is True.
            remove_numbers (bool): Remove numbers. Default is False.
            lemmatize (bool): Apply lemmatization. Default is True.
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        
        # Initialize NLTK components
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting.
        
        Args:
            text (str): Input text to clean.
        
        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_punctuation_func(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text (str): Input text.
        
        Returns:
            str: Text without punctuation.
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers_func(self, text: str) -> str:
        """
        Remove numbers from text.
        
        Args:
            text (str): Input text.
        
        Returns:
            str: Text without numbers.
        """
        return re.sub(r'\d+', '', text)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Input text.
        
        Returns:
            List[str]: List of tokens.
        """
        return word_tokenize(text)
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens (List[str]): List of tokens.
        
        Returns:
            List[str]: Tokens without stopwords.
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens (List[str]): List of tokens.
        
        Returns:
            List[str]: Lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text (str): Input text.
        
        Returns:
            str: Preprocessed text.
        """
        # Clean text
        text = self.clean_text(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self.remove_punctuation_func(text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self.remove_numbers_func(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: Union[List[str], 'pd.Series']) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts (Union[List[str], pd.Series]): List or Series of texts.
        
        Returns:
            List[str]: List of preprocessed texts.
        """
        logger.info(f"Preprocessing {len(texts)} texts")
        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = "This is an AMAZING product! I love it so much!!! https://example.com"
    processed = preprocessor.preprocess(sample_text)
    
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
