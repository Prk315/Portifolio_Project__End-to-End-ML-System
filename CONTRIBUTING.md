# Contributing Guidelines

Thank you for your interest in contributing to this project!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Standards

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Add tests for new features

## Testing

Run tests before submitting:
```bash
pytest tests/ -v --cov=src
```

## Submitting Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for new functionality
4. Run tests and ensure they pass
5. Commit your changes:
   ```bash
   git commit -m "Add feature: description"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include test coverage for new features
- Ensure all tests pass
- Update documentation as needed

## Areas for Contribution

- Additional model algorithms
- Enhanced explainability features
- Improved monitoring capabilities
- Performance optimizations
- Documentation improvements
- Bug fixes

## Questions?

Feel free to open an issue for any questions or discussions.
