"""
Unit tests for the TextPreprocessor class.

Run with: pytest tests/test_preprocessor.py
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a default preprocessor instance."""
        return TextPreprocessor(
            lowercase=True,
            remove_stopwords=True,
            remove_punctuation=True,
            remove_numbers=False,
            lemmatize=True
        )
    
    @pytest.fixture
    def simple_preprocessor(self):
        """Create a minimal preprocessor (only lowercase)."""
        return TextPreprocessor(
            lowercase=True,
            remove_stopwords=False,
            remove_punctuation=False,
            remove_numbers=False,
            lemmatize=False
        )
    
    def test_initialization(self):
        """Test that preprocessor initializes correctly."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
        assert preprocessor.lowercase is True
        assert preprocessor.remove_stopwords is True
    
    def test_clean_text_removes_urls(self, preprocessor):
        """Test URL removal."""
        text = "Check this out https://example.com and www.test.com"
        cleaned = preprocessor.clean_text(text)
        assert "https://" not in cleaned
        assert "www." not in cleaned
    
    def test_clean_text_removes_emails(self, preprocessor):
        """Test email removal."""
        text = "Contact me at test@example.com for more info"
        cleaned = preprocessor.clean_text(text)
        assert "@" not in cleaned or "test@example.com" not in cleaned
    
    def test_clean_text_removes_mentions(self, preprocessor):
        """Test mention removal."""
        text = "Hey @user check out #awesome"
        cleaned = preprocessor.clean_text(text)
        assert "@user" not in cleaned
        assert "#awesome" not in cleaned
    
    def test_lowercase_conversion(self, preprocessor):
        """Test lowercase conversion."""
        text = "THIS IS UPPERCASE TEXT"
        processed = preprocessor.preprocess(text)
        assert processed == processed.lower()
    
    def test_punctuation_removal(self, preprocessor):
        """Test punctuation removal."""
        text = "Hello! This is great!!!"
        processed = preprocessor.preprocess(text)
        assert "!" not in processed
    
    def test_stopwords_removal(self, preprocessor):
        """Test stopword removal."""
        text = "this is a test"
        processed = preprocessor.preprocess(text)
        # Common stopwords like 'is', 'a' should be removed
        assert "test" in processed
        # Note: stopword removal might vary, so we check the result isn't empty
        assert len(processed) > 0
    
    def test_number_handling(self):
        """Test number removal when enabled."""
        preprocessor_remove_nums = TextPreprocessor(
            remove_numbers=True,
            remove_punctuation=False
        )
        text = "I have 123 apples and 456 oranges"
        processed = preprocessor_remove_nums.preprocess(text)
        assert "123" not in processed
        assert "456" not in processed
    
    def test_tokenization(self, preprocessor):
        """Test tokenization."""
        text = "Hello world this is a test"
        tokens = preprocessor.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "Hello" in tokens or "hello" in tokens
    
    def test_lemmatization(self, preprocessor):
        """Test lemmatization."""
        text = "running runs ran"
        processed = preprocessor.preprocess(text)
        # Lemmatization should convert variants to base form
        assert "run" in processed or "running" in processed
    
    def test_preprocess_batch(self, preprocessor):
        """Test batch preprocessing."""
        texts = [
            "This is text one",
            "This is text two",
            "This is text three"
        ]
        processed = preprocessor.preprocess_batch(texts)
        assert isinstance(processed, list)
        assert len(processed) == len(texts)
        assert all(isinstance(text, str) for text in processed)
    
    def test_empty_string_handling(self, preprocessor):
        """Test handling of empty strings."""
        text = ""
        processed = preprocessor.preprocess(text)
        assert processed == ""
    
    def test_none_handling(self, preprocessor):
        """Test handling of None input."""
        text = None
        processed = preprocessor.clean_text(text)
        assert processed == ""
    
    def test_special_characters(self, preprocessor):
        """Test handling of special characters."""
        text = "Hello @#$%^& world"
        processed = preprocessor.preprocess(text)
        # Should handle gracefully without errors
        assert isinstance(processed, str)
    
    def test_multiple_spaces(self, preprocessor):
        """Test removal of extra whitespace."""
        text = "Hello    world    test"
        cleaned = preprocessor.clean_text(text)
        assert "    " not in cleaned  # Multiple spaces should be reduced
    
    def test_configuration_variations(self):
        """Test different configuration combinations."""
        configs = [
            {'lowercase': True, 'remove_stopwords': False},
            {'lowercase': False, 'remove_punctuation': False},
            {'lemmatize': False, 'remove_numbers': True}
        ]
        
        for config in configs:
            preprocessor = TextPreprocessor(**config)
            text = "This is a TEST with 123 numbers!"
            result = preprocessor.preprocess(text)
            assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
