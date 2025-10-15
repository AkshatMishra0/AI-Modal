# Contributing to Sentiment Analysis AI

First off, thank you for considering contributing to Sentiment Analysis AI! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-ai.git
   cd sentiment-analysis-ai
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-owner/sentiment-analysis-ai.git
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Code samples** or error messages

Example:

```markdown
**Bug**: Model fails to load on Windows

**Steps to Reproduce:**
1. Train a model on Windows
2. Save with `trainer.save_model('model.pkl')`
3. Try to load with `SentimentPredictor('model.pkl')`

**Expected:** Model loads successfully
**Actual:** FileNotFoundError raised

**Environment:**
- OS: Windows 10
- Python: 3.9.5
- Package version: 1.0.0
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Clear title and description**
- **Use case** - why is this enhancement useful?
- **Proposed solution** - how would you implement it?
- **Alternatives considered**
- **Examples** from other projects (if applicable)

### Contributing Code

1. **Pick an issue** or create one
2. **Comment on the issue** to let others know you're working on it
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**
5. **Test your changes**
6. **Commit and push**
7. **Create a pull request**

## Development Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Verify Setup

```bash
# Run tests
pytest

# Check code style
flake8 src/ tests/

# Format code
black src/ tests/
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 100 characters (instead of 79)
- **Imports**: Group in order: standard library, third-party, local
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for function parameters and returns

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black src/ tests/
```

### Linting

We use [Flake8](https://flake8.pycqa.org/) for linting:

```bash
flake8 src/ tests/ --max-line-length=100
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1 (str): Description of param1.
        param2 (int): Description of param2.
    
    Returns:
        bool: Description of return value.
    
    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is not an integer.
    
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(model): add support for SVM classifier

Added SVM classifier option to SentimentModel class.
Includes parameter tuning and probability calibration.

Closes #42
```

```
fix(preprocessor): handle None input gracefully

Modified clean_text() to return empty string when
input is None instead of raising TypeError.

Fixes #38
```

### Best Practices

- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests in the footer

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run all tests** and ensure they pass
4. **Update CHANGELOG** (if applicable)
5. **Format code** with Black
6. **Check linting** with Flake8

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] All tests passing
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. At least one maintainer must review
2. All tests must pass
3. Code must follow style guidelines
4. Documentation must be updated
5. No merge conflicts

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessor.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings in test functions
- Use fixtures for setup/teardown

Example:

```python
"""Tests for the preprocessor module."""

import pytest
from src.preprocessor import TextPreprocessor

class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return TextPreprocessor()
    
    def test_lowercase_conversion(self, preprocessor):
        """Test that text is converted to lowercase."""
        text = "HELLO WORLD"
        result = preprocessor.preprocess(text)
        assert result == result.lower()
```

### Test Coverage

Aim for at least 80% code coverage:

```bash
pytest --cov=src --cov-report=html tests/
```

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Keep docstrings up-to-date with code changes

### README Updates

Update README.md if you:
- Add new features
- Change installation process
- Modify usage examples
- Update dependencies

### API Documentation

Update `docs/API.md` when:
- Adding new public APIs
- Changing function signatures
- Modifying return values
- Adding new modules

## Questions?

Feel free to:
- Open an issue for discussion
- Reach out to maintainers
- Join our community chat (if available)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing! 🎉
