# Volleyball AI Project

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Player-aware volleyball match prediction and tournament simulation using machine learning**

Analyze Philippine Volleyball League (PVL) match data, train calibrated machine learning models with player-specific features, and simulate complete tournaments with FIVB-compliant ranking rules.

---

## 🎯 Features

- **🏐 Comprehensive Data Pipeline**: Parse 500+ PVL XML match files (2023-2025) into structured SQLite database
- **👥 Player-Aware Features**: Individual player statistics, lineup analysis, and team composition metrics
- **📊 Advanced Feature Engineering**: ELO ratings, historical aggregates, momentum indicators, and time-aware features
- **🤖 Calibrated ML Models**: XGBoost with Platt scaling for reliable probability predictions
- **🏆 Tournament Simulation**: Full bracket simulation with FIVB ranking rules and head-to-head tiebreakers
- **📈 Model Metrics Tracking**: Comprehensive evaluation with calibration diagnostics and CI integration
- **🎪 Champion Analysis**: Bracket favorite detection, upset identification, and confidence scoring

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VolleyballAIProject.git
   cd VolleyballAIProject
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Quick Start

### 1. Process Match Data

Place your PVL XML match files in `data/xml_files/` and run the batch processor:

```bash
python scripts/batch_processor.py
```

This will:
- Parse XML files into structured data
- Build SQLite database (`data/databases/volleyball_data.db`)
- Generate feature matrices with player statistics
- Create training datasets in `data/csv_files/`

### 2. Train Models

Train a calibrated XGBoost model with time-aware cross-validation:

```bash
python scripts/train_xgboost_with_players.py
```

Output: `models/calibrated_xgboost_with_players.pkl`

### 3. Simulate Tournament

Run a complete tournament simulation:

```bash
python scripts/simulate_tournament.py \
  --model models/calibrated_xgboost_with_players.pkl \
  --save_outputs \
  --champion_analysis
```

Optional flags:
- `--save_outputs`: Save results to `outputs/` directory
- `--keep_latest N`: Keep only the N most recent simulation outputs
- `--champion_analysis`: Show bracket favorite, upset detection, and confidence analysis

---

## 💻 Usage

### Compare Model Metrics

Compare two model artifacts on the same holdout set with calibration diagnostics:

```bash
python scripts/compare_metrics.py \
  models/best_model_with_players_timeaware.pkl \
  models/calibrated_xgboost_with_players.pkl
```

Outputs:
- `outputs/metrics_comparison_YYYYMMDD_HHMMSS.md` - Formatted report
- `outputs/metrics_comparison_YYYYMMDD_HHMMSS.json` - Structured data with calibration bins

### Advanced Simulation

```bash
# Use alternative model
python scripts/simulate_tournament.py \
  --model models/best_model_with_players_timeaware.pkl \
  --save_outputs

# Legacy entry point (calls simulate_tournament.py internally)
python run_simulation.py --model models/calibrated_xgboost_with_players.pkl
```

---

## 📁 Project Structure

```
VolleyballAIProject/
├── data/
│   ├── xml_files/              # Raw PVL XML match files
│   ├── databases/              # SQLite database
│   └── csv_files/              # Engineered feature matrices
├── models/                     # Trained model artifacts (.pkl)
├── outputs/                    # Simulation results and metrics
├── scripts/
│   ├── batch_processor.py      # XML → DB → Features pipeline
│   ├── parse_volleyball_data.py    # XML parser
│   ├── database_manager.py     # Database operations
│   ├── feature_engineering_with_players.py  # Feature extraction
│   ├── train_xgboost_with_players.py       # Model training
│   ├── simulate_tournament.py  # Tournament simulation
│   ├── compare_metrics.py      # Model comparison
│   └── archive/                # Legacy scripts
├── docs/                       # Documentation
├── .github/workflows/          # CI/CD pipelines
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🤖 Models

### Available Model Artifacts

| Model | Description | Metrics |
|-------|-------------|---------|
| `calibrated_xgboost_with_players.pkl` | **Recommended**: Time-aware XGBoost with Platt calibration | Acc: 73.5%, AUC: 0.80 |
| `best_model_with_players_timeaware.pkl` | Calibrated ensemble with ELO features | Acc: 64.7%, AUC: 0.73 |
| `best_model_with_players_timeaware_stacking.pkl` | Stacked meta-learner (time-aware) | - |
| `volleyball_predictor_with_players_uncalibrated.pkl` | Raw XGBoost (pre-calibration) | - |

### Feature Categories

1. **ELO Ratings**: Pre-match ratings and win probabilities (leak-free)
2. **Team Aggregates**: Historical attack, block, serve, points, win rates
3. **Player Stats**: Starter averages, libero performance, top scorers, roster depth
4. **Temporal Features**: Time-aware splitting for realistic evaluation

### Training Pipeline

- **Cross-validation**: Time-aware blocked k-fold on chronological training data
- **Calibration**: Sigmoid/Platt scaling on separate calibration slice
- **Evaluation**: Holdout test set with metrics: Accuracy, LogLoss, Brier, AUC, ECE, MCE

---

## 📚 Documentation

Comprehensive documentation available in `docs/`:

- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)**: Methodology and architecture
- **[FINAL_MODEL_SUMMARY.md](docs/FINAL_MODEL_SUMMARY.md)**: Model performance analysis
- **[DATA_QUALITY_REPORT.md](docs/DATA_QUALITY_REPORT.md)**: Data validation and coverage
- **[tournament_format.md](docs/tournament_format.md)**: FIVB ranking rules

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Quality

This project uses:
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

---

## 🔬 Research & Next Steps

- [ ] Player ID linkage for longitudinal tracking
- [ ] Set-level granular features (momentum, clutch points)
- [ ] Reliability diagrams and calibration drift monitoring
- [ ] Confidence intervals for predictions
- [ ] Live match prediction API

---

## �� CI/CD

The project includes GitHub Actions workflows:

- **Model Metrics Check**: Automatically validates model performance on push/PR
- **Regression Gating**: Fails if accuracy drops >2% or logloss increases >0.05
- **Artifact Upload**: Stores comparison reports for 30 days

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Philippine Volleyball League (PVL) for match data
- XGBoost and scikit-learn communities
- Contributors and testers

---

## �� Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Made with ❤️ for volleyball analytics**
