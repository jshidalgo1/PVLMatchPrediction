# Contributing to Volleyball AI Project

Thank you for your interest in contributing to the Volleyball AI Project! This document provides guidelines and instructions for contributing.

---

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)
- Relevant error messages or logs

### Suggesting Features

Feature requests are welcome! Please:
- Check existing issues to avoid duplicates
- Describe the feature and its use case
- Explain why it would benefit the project
- Provide examples if possible

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style guidelines

3. **Test your changes** thoroughly
   ```bash
   pytest tests/
   ```

4. **Update documentation** if needed

5. **Commit your changes** with clear messages
   ```bash
   git commit -m "Add feature: description of changes"
   ```

6. **Push to your fork** and submit a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

---

## 💻 Development Setup

### Prerequisites

- Python 3.13+
- Git
- Virtual environment tool (venv)

### Setting Up Your Environment

1. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/VolleyballAIProject.git
   cd VolleyballAIProject
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install dev dependencies
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

---

## 📝 Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) conventions
- Use **black** for code formatting (line length: 100)
- Use **isort** for import sorting
- Use **flake8** for linting
- Use **mypy** for type hints (encouraged but not required)

### Pre-commit Hooks

The project uses pre-commit hooks to enforce code quality:
- **black**: Automatic code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **trailing-whitespace**: Remove trailing spaces
- **end-of-file-fixer**: Ensure files end with newline

Run manually with:
```bash
pre-commit run --all-files
```

### Code Structure

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add docstrings for public functions and classes
- Comment complex logic
- Avoid deep nesting (max 3-4 levels)

### Example Function

```python
def calculate_elo_rating(
    current_rating: float,
    opponent_rating: float,
    actual_score: float,
    k_factor: float = 32.0
) -> float:
    """
    Calculate new ELO rating after a match.

    Args:
        current_rating: Player's current ELO rating
        opponent_rating: Opponent's current ELO rating
        actual_score: Actual match outcome (1.0 for win, 0.5 for draw, 0.0 for loss)
        k_factor: Rating adjustment factor (default: 32.0)

    Returns:
        Updated ELO rating

    Example:
        >>> calculate_elo_rating(1500, 1600, 1.0)
        1516.0
    """
    expected_score = 1 / (1 + 10 ** ((opponent_rating - current_rating) / 400))
    return current_rating + k_factor * (actual_score - expected_score)
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=scripts tests/

# Run specific test file
pytest tests/test_feature_engineering.py

# Run with verbose output
pytest -v tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names: `test_<functionality>_<scenario>`
- Include both positive and negative test cases
- Mock external dependencies (database, file I/O)

Example test:
```python
def test_elo_rating_calculation_player_wins():
    """Test ELO calculation when player wins against higher-rated opponent."""
    new_rating = calculate_elo_rating(
        current_rating=1500,
        opponent_rating=1600,
        actual_score=1.0
    )
    assert new_rating > 1500  # Rating should increase
    assert new_rating < 1550  # But not by more than K-factor
```

---

## 📊 Model Changes

### Training New Models

If your PR includes model changes:
1. Document the rationale for changes
2. Include before/after metrics comparison
3. Run `scripts/compare_metrics.py` and attach results
4. Ensure CI metrics checks pass
5. Update model documentation in `docs/`

### Model Performance Standards

New models must meet these thresholds:
- Accuracy: Must not drop >2% from baseline
- LogLoss: Must not increase >0.05 from baseline
- AUC: Must not drop >2% from baseline

---

## 📚 Documentation

### When to Update Docs

Update documentation when:
- Adding new features
- Changing API or CLI interfaces
- Modifying data pipeline
- Updating model training procedures
- Adding new dependencies

### Documentation Standards

- Update `README.md` for user-facing changes
- Update `docs/` for technical documentation
- Include code examples where appropriate
- Keep documentation in sync with code
- Use clear, concise language

---

## 🔀 Git Workflow

### Branch Naming

- `feature/` - New features
- `bugfix/` - Bug fixes
- `hotfix/` - Critical production fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

Examples:
- `feature/add-player-clustering`
- `bugfix/fix-elo-calculation`
- `docs/update-model-training-guide`

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(simulation): add champion analysis flag

Add --champion_analysis flag to tournament simulation that displays:
- Bracket favorite (seed #1)
- Champion and runner-up seeds
- Upset detection
- Average bracket confidence

Closes #42
```

---

## 📋 Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Pre-commit hooks pass
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts with main
- [ ] CI checks pass
- [ ] Model metrics thresholds met (if applicable)

---

## 🎯 Priority Areas

We're especially interested in contributions in these areas:

1. **Player ID Linkage**: Persistent player tracking across matches
2. **Set-Level Features**: Momentum and clutch performance metrics
3. **Calibration Monitoring**: Drift detection and reliability diagrams
4. **API Development**: REST API for live predictions
5. **Data Visualization**: Interactive tournament brackets and player stats
6. **Testing**: Increased test coverage
7. **Documentation**: Examples, tutorials, and use cases

---

## ❓ Questions?

- Open an issue for questions
- Join discussions in existing issues
- Check `docs/` for technical details
- Review existing PRs for examples

---

## 🙏 Thank You!

Your contributions help make volleyball analytics more accessible and powerful. We appreciate your time and effort!

---

**Happy coding! 🏐**
