# Contributing to Dataseter

Thank you for your interest in contributing to Dataseter! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/dataseter/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)

### Suggesting Features

1. Check existing [feature requests](https://github.com/yourusername/dataseter/issues?q=is%3Aissue+label%3Aenhancement)
2. Open a new issue with the `enhancement` label
3. Describe the feature and its use case

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting:
   ```bash
   pytest
   black src/
   flake8 src/
   mypy src/
   ```
6. Commit with descriptive message
7. Push to your fork
8. Open a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/dataseter.git
cd dataseter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow PEP 8
- Use Black for formatting
- Add type hints where possible
- Write docstrings for all public functions
- Keep functions focused and small

## Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest for testing
- Mock external dependencies

## Documentation

- Update docstrings
- Update README if needed
- Add examples for new features
- Update configuration documentation

## Community

- Be respectful and inclusive
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
- Help others in issues and discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.