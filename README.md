# Volleyball AI Project

This project analyzes volleyball match data and uses machine learning to predict tournament outcomes.

## Project Structure

```
VolleyballAIProject/
├── certificates/          # SSL/TLS certificates (.pem files)
├── data/                  # All data files
│   ├── csv_files/         # Feature data and targets
│   ├── databases/         # SQLite database files
│   └── xml_files/         # Match data in XML format
│       └── (400+ match files from PVL 2023-2025)
├── docs/                  # Project documentation
│   ├── DATA_QUALITY_REPORT.md
│   ├── FINAL_MODEL_SUMMARY.md
│   ├── PROJECT_OVERVIEW.md
│   └── tournament_format.md
├── models/                # Trained ML models
│   ├── best_model_with_players.pkl
│   └── catboost_info/     # CatBoost training logs
├── outputs/               # Generated visualizations and results
│   ├── pool_standings.png
│   ├── tournament_bracket.png
│   ├── simulation_output.txt
│   └── requirements.txt
├── scripts/               # Python scripts
│   ├── batch_processor.py
│   ├── complete_tournament_simulation.py
│   ├── database_manager.py
│   ├── download_matches.py
│   ├── feature_engineering.py
│   ├── feature_engineering_with_players.py
│   ├── fetch_all_matches.py
│   ├── fix_test_pvlr25_data.py
│   ├── multi_model_tournament_simulation.py
│   ├── parse_volleyball_data.py
│   ├── predict_tournament.py
│   ├── quickstart.py
│   ├── simulate_remaining_matches.py
│   ├── simulate_second_round.py
│   ├── simulate_tournament.py
│   ├── tournament_visualization.py
│   ├── train_naive_bayes.py
│   ├── train_with_player_features.py
│   ├── train_xgboost.py
│   ├── train_xgboost_with_players.py
│   └── validate_data.py
└── .venv/                 # Python virtual environment

```

## Quick Start

1. **Set up the environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r outputs/requirements.txt
   ```

2. **Run the main scripts:**
   ```bash
   cd scripts
   python quickstart.py
   ```

## Data

- **XML Files**: 400+ match files from Philippine Volleyball League (PVL) seasons 2023-2025
- **Database**: SQLite database with parsed match data
- **CSV Files**: Engineered features and target variables for ML models

## Models

The project uses various machine learning models including:
- XGBoost
- CatBoost
- Naive Bayes

Models are trained with player-specific features to improve prediction accuracy.

## Scripts

### Data Processing
- `parse_volleyball_data.py` - Parse XML match files
- `feature_engineering.py` - Create features for ML models
- `database_manager.py` - Manage SQLite database

### Model Training
- `train_xgboost_with_players.py` - Train XGBoost model
- `train_naive_bayes.py` - Train Naive Bayes model

### Simulation & Prediction
- `predict_tournament.py` - Predict tournament outcomes
- `simulate_tournament.py` - Run tournament simulations
- `tournament_visualization.py` - Generate visualizations

## Documentation

See the `docs/` folder for detailed documentation:
- Project overview and methodology
- Data quality reports
- Model performance summaries
- Tournament format details
