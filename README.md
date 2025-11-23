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

### 1. Fetch Match Data

Download PVL XML match files using the fetch script:

```bash
python scripts/fetch_all_matches.py --download --tournament PVL2025D --week-range 46-48
```

Or download all available matches:

```bash
python scripts/fetch_all_matches.py --download --limit 50
```

### 2. Process Match Data

Process XML files into structured data and generate features:

```bash
python scripts/batch_processor.py
```

This will:
- Parse XML files into structured data
- Build SQLite database (`data/databases/volleyball_data.db`)
- Map players to jersey numbers using roster data
- Calculate accurate sets_played from roster tags
- Generate feature matrices with player statistics
- Create training datasets in `data/csv_files/`

### 3. Train Models

Train a calibrated XGBoost model with time-aware cross-validation:

```bash
python scripts/train_xgboost_with_players.py
```

Output: `models/calibrated_xgboost_with_players.pkl`

### 4. Simulate Tournament

Run a complete tournament simulation with playoff bracket:

```bash
python scripts/simulate_tournament.py --save_outputs --champion_analysis
```

This generates:
- Quarterfinal, semifinal, and championship predictions
- Third place match prediction
- Simulation outputs saved to `outputs/`

### 5. Export to Dashboard

Export simulation results and player statistics to the dashboard:

```bash
python scripts/export_dashboard_data.py
```

Generates: `dashboard/public/data.json` with:
- Tournament standings and predictions
- Player statistics (per-set averages)
- Playoff bracket with all matchups
- Match history organized by phase

### 6. Run Dashboard

Start the Next.js dashboard to visualize results:

```bash
cd dashboard
npm install  # First time only
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view:
- Tournament standings and projections
- Player statistics with sortable metrics
- Playoff bracket visualization
- Match predictions by phase

---

## 📊 Dashboard Features

The project includes a modern Next.js dashboard for visualizing tournament data, predictions, and player statistics.

### Features

- **🏆 Tournament Standings**: Current and projected standings with FIVB ranking rules
- **👥 Player Statistics**: Sortable player stats with per-set averages
  - Total Points, Attack Points (displayed as totals)
  - Block, Serve, Dig, Reception, Set (displayed as per-set averages)
  - Filter by current tournament
  - Dynamic sorting by any metric
- **🎯 Match Predictions**: Simulated results organized by tournament phase
  - Preliminary rounds (Groups A, B, C, D)
  - Quarterfinals, Semifinals
  - Championship and Third Place matches
- **📈 Playoff Bracket**: Visual bracket showing all playoff matchups
  - Confidence scores for each prediction
  - Winner highlighting
  - Complete path from quarterfinals to champion

### Updating Dashboard Data

To refresh dashboard with latest data:

```bash
# 1. Run simulation if new matches added
python scripts/simulate_tournament.py --save_outputs

# 2. Export to dashboard
python scripts/export_dashboard_data.py

# 3. Dashboard auto-refreshes on data.json change
```

---

## 💻 Usage

### Fetch Match Data

Download specific matches using filters:

```bash
# List available files
python scripts/fetch_all_matches.py --list

# Download specific tournament and week range
python scripts/fetch_all_matches.py --download --tournament PVL2025D --week-range 46-48

# Download by year
python scripts/fetch_all_matches.py --download --year 2025 --limit 50

# Download all with delay between requests
python scripts/fetch_all_matches.py --download --delay 0.5
```

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

# Keep only latest 5 simulation outputs
python scripts/simulate_tournament.py --save_outputs --keep_latest 5

# Legacy entry point (calls simulate_tournament.py internally)
python run_simulation.py --model models/calibrated_xgboost_with_players.pkl
```

---

## 📁 Project Structure

```
VolleyballAIProject/
├── data/
│   ├── xml_files/              # Raw PVL XML match files
│   ├── databases/              # SQLite database (volleyball_data.db)
│   └── csv_files/              # Engineered feature matrices
├── models/                     # Trained model artifacts (.pkl)
├── outputs/                    # Simulation results (JSON + TXT)
├── dashboard/                  # Next.js dashboard application
│   ├── src/
│   │   ├── app/                # Next.js app router pages
│   │   ├── components/         # React components
│   │   └── types/              # TypeScript interfaces
│   ├── public/
│   │   └── data.json           # Exported dashboard data
│   └── package.json            # Dashboard dependencies
├── scripts/
│   ├── batch_processor.py      # XML → DB → Features pipeline
│   ├── parse_volleyball_data.py    # XML parser with player mapping
│   ├── database_manager.py     # Database operations
│   ├── feature_engineering_with_players.py  # Feature extraction
│   ├── train_xgboost_with_players.py       # Model training
│   ├── simulate_tournament.py  # Tournament simulation with playoffs
│   ├── export_dashboard_data.py  # Export data to dashboard
│   ├── fetch_all_matches.py    # Download PVL XML files
│   ├── compare_metrics.py      # Model comparison
│   └── config.py               # Centralized configuration
├── docs/                       # Comprehensive documentation
├── .github/workflows/          # CI/CD pipelines
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🤖 Models

### Available Model Artifacts

| Model | Description | Metrics (Nov 2025) |
|-------|-------------|---------|
| `calibrated_xgboost_with_players.pkl` | **Recommended**: Time-aware XGBoost with Platt calibration | Acc: 74.29%, AUC: 0.8155, Brier: 0.1803 |
| `volleyball_predictor_with_players_uncalibrated.pkl` | Raw XGBoost (pre-calibration) | Acc: 74.29%, AUC: 0.8155, Brier: 0.1900 |
| `best_model_with_players_timeaware.pkl` | Legacy calibrated model | Acc: 64.7%, AUC: 0.73 |
| `best_model_with_players_timeaware_stacking.pkl` | Stacked meta-learner (deprecated) | - |

### Feature Categories (34 features from 521 matches)

1. **ELO Ratings** (3 features): Pre-match ratings and win probabilities (leak-free)
2. **Team Aggregates** (10 features): Historical attack, block, serve, points, win rates
3. **Player Stats** (18 features): Starter averages, libero performance, top scorers, roster depth
4. **Match Context** (3 features): Head-to-head records, form indicators

### Training Pipeline (Updated Nov 2025)

- **Dataset**: 521 matches (2023-2025 PVL seasons)
- **Cross-validation**: 4-fold time-aware blocked CV (Avg: Acc=67.47%, AUC=0.7310)
- **Calibration**: Platt scaling on 42-sample calibration window
- **Test Set**: 105 holdout matches (chronologically latest)
- **Calibration Improvement**: ECE reduced from 0.1333 → 0.1125, MCE: 0.3099 → 0.2473
- **Evaluation Metrics**: Accuracy, LogLoss, Brier Score, AUC-ROC, ECE, MCE

---

## 📚 Documentation

Comprehensive documentation available in `docs/`:

-   **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)**: Complete system architecture and methodology
-   **[FINAL_MODEL_SUMMARY.md](docs/FINAL_MODEL_SUMMARY.md)**: Model performance and calibration analysis
-   **[DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md)**: Dashboard features and usage guide
-   **[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)**: Data processing pipeline details
-   **[PLAYER_STATISTICS.md](docs/PLAYER_STATISTICS.md)**: Player stats methodology and calculations
-   **[tournament_format.md](docs/tournament_format.md)**: FIVB ranking rules and tournament structure

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

- [x] Player ID linkage for longitudinal tracking ✅
- [x] Set-level granular features (sets_played from roster) ✅
- [x] Dashboard for visualization ✅
- [ ] Reliability diagrams and calibration drift monitoring
- [ ] Confidence intervals for predictions
- [ ] Live match prediction API

---

## 🔄 CI/CD

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
- Next.js and React communities
- Contributors and testers

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Made with ❤️ for volleyball analytics**

**Last Updated**: November 23, 2025
```
