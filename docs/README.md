# Documentation

Comprehensive documentation for the PVL Match Prediction project.

## 📚 Documentation Overview

This directory contains detailed documentation about the project's methodology, data quality, model performance, and system architecture.

### Available Documents

#### [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
Detailed overview of the entire project including:
- System architecture and design
- Data pipeline and processing workflow
- Feature engineering methodology
- Model training and evaluation approach
- Tournament simulation logic

#### [FINAL_MODEL_SUMMARY.md](FINAL_MODEL_SUMMARY.md)
Comprehensive analysis of model performance:
- Model comparison and benchmarking
- Performance metrics (Accuracy, LogLoss, Brier, AUC, ECE, MCE)
- Calibration diagnostics
- Feature importance analysis
- Recommendations for model selection

#### [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) ✨ NEW
Complete dashboard documentation:
- Features and UI components
- Player statistics with per-set averages
- Match predictions and playoff brackets
- Data flow and technical implementation
- Customization guide
- Troubleshooting

#### [DATA_PIPELINE.md](DATA_PIPELINE.md) ✨ NEW
End-to-end data processing pipeline:
- XML file fetching and parsing
- Player-jersey number mapping
- Sets played calculation methodology
- Database schema and operations
- Feature engineering process
- Export to dashboard workflow

#### [PLAYER_STATISTICS.md](PLAYER_STATISTICS.md) ✨ NEW
Player statistics tracking and calculation:
- Roster-based sets_played calculation
- Per-set average formulas
- Data validation procedures
- Common issues and solutions
- Example SQL queries

#### [tournament_format.md](tournament_format.md)
FIVB tournament rules and implementation:
- Tournament structure and phases
- Ranking rules and tie-breakers
- Head-to-head comparison logic
- Pool play and bracket format
- Championship determination

## 🎯 Quick Links

### For Users
- [Installation Guide](../README.md#-installation)
- [Quick Start](../QUICK_START.md)
- [Dashboard Setup](DASHBOARD_GUIDE.md)

### For Contributors
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Code Style Guide](../CONTRIBUTING.md#-code-style-guidelines)
- [Development Setup](../CONTRIBUTING.md#-development-setup)

### For Researchers
- [Feature Engineering Details](PROJECT_OVERVIEW.md)
- [Model Architecture](FINAL_MODEL_SUMMARY.md)
- [Data Pipeline Flow](DATA_PIPELINE.md)
- [Statistical Methodology](PLAYER_STATISTICS.md)

## 🔍 Documentation by Topic

### Getting Started
1. Start with [Quick Start Guide](../QUICK_START.md)
2. Review [Project Overview](PROJECT_OVERVIEW.md)
3. Understand the [Data Pipeline](DATA_PIPELINE.md)

### Working with Data
- [Fetching Match Data](DATA_PIPELINE.md#1-data-acquisition)
- [Processing XML Files](DATA_PIPELINE.md#2-data-parsing)
- [Database Schema](DATA_PIPELINE.md#3-database-storage)
- [Player Statistics](PLAYER_STATISTICS.md)

### Machine Learning
- [Feature Engineering](DATA_PIPELINE.md#4-feature-engineering)
- [Model Training](DATA_PIPELINE.md#5-model-training)
- [Model Performance](FINAL_MODEL_SUMMARY.md)

### Simulation & Visualization
- [Tournament Simulation](DATA_PIPELINE.md#6-tournament-simulation)
- [Dashboard Export](DATA_PIPELINE.md#7-dashboard-export)
- [Dashboard Features](DASHBOARD_GUIDE.md)

## 💡 Tips for Reading Documentation

1. **New Users**: Start with [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for a high-level understanding
2. **Data Scientists**: Check [FINAL_MODEL_SUMMARY.md](FINAL_MODEL_SUMMARY.md) for model details
3. **Developers**: Review [DATA_PIPELINE.md](DATA_PIPELINE.md) and [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)
4. **Troubleshooting**: Each guide has a dedicated troubleshooting section

## 🙋 Questions?

If you can't find what you're looking for:
- Check the main [README](../README.md)
- Review the [Quick Start Guide](../QUICK_START.md)
- Open an issue on GitHub
- Check existing issues and discussions

## 📝 Contributing to Documentation

Documentation contributions are highly valued! To contribute:

1. Identify gaps or outdated information
2. Fork the repository
3. Update the relevant `.md` file
4. Submit a pull request with clear description
5. Ensure markdown formatting is consistent

**Style Guidelines**:
- Use clear, concise language
- Include code examples where helpful
- Add links to related documentation
- Update "Last Updated" dates

---

**Last Updated**: November 23, 2025
