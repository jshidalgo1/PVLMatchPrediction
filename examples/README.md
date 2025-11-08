# Examples

This directory contains example scripts and notebooks demonstrating how to use the Volleyball AI Project.

## 📋 Available Examples

### Basic Usage

- **[basic_simulation.py](basic_simulation.py)**: Simple tournament simulation example
- **[train_custom_model.py](train_custom_model.py)**: Training a model with custom parameters
- **[compare_models.py](compare_models.py)**: Comparing multiple model artifacts

### Data Processing

- **[parse_single_match.py](parse_single_match.py)**: Parse and analyze a single XML match file
- **[extract_features.py](extract_features.py)**: Extract features from database for custom analysis

### Analysis

- **[player_statistics.py](player_statistics.py)**: Analyze player performance metrics
- **[team_rankings.py](team_rankings.py)**: Calculate and visualize team rankings

## 🚀 Running Examples

All examples should be run from the project root directory:

```bash
# From project root
python examples/basic_simulation.py
```

## 📝 Prerequisites

Ensure you have:
1. Installed all dependencies: `pip install -r requirements.txt`
2. Processed match data: `python scripts/batch_processor.py`
3. Trained at least one model: `python scripts/train_xgboost_with_players.py`

## 💡 Tips

- Modify the examples to fit your specific use case
- Check `docs/` for detailed documentation
- See `scripts/` for production-ready implementations
- Use examples as templates for your own scripts

## 🙋 Need Help?

- Check the main [README.md](../README.md)
- Review [CONTRIBUTING.md](../CONTRIBUTING.md)
- Open an issue on GitHub
