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
│   ├── best_model_with_players.pkl                 # legacy default (may exist)
│   ├── best_model_with_players_random.pkl          # best from random-split pipeline
│   ├── best_model_with_players_timeaware.pkl       # best from time-aware + calibrated pipeline
│   ├── best_model_with_players_random_stacking.pkl # stacked meta-learner (random)
│   ├── best_model_with_players_timeaware_stacking.pkl # stacked meta-learner (time-aware)
│   ├── calibrated_xgboost_with_players.pkl         # calibrated XGBoost (time-aware trainer)
│   ├── volleyball_predictor_with_players_uncalibrated.pkl
│   └── catboost_info/                              # CatBoost training logs
├── outputs/               # Generated visualizations and results
│   ├── pool_standings.png
│   ├── tournament_bracket.png
│   ├── simulation_output.txt
│   ├── simulation_output_random.txt                # baseline summary (--save-summary)
│   └── simulation_output_timeaware.txt             # time-aware summary (--save-summary)
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

3. Run a simulation with a specific model (optional):
   ```bash
   # From project root
   # Use the time-aware calibrated best model
   python run_simulation.py --model models/best_model_with_players_timeaware.pkl

   # Compare with the random-split best model
   python run_simulation.py --model models/best_model_with_players_random.pkl

   # Or use a calibrated XGBoost directly
   python run_simulation.py --model models/calibrated_xgboost_with_players.pkl

   # Try the stacked meta-learner variants
   python run_simulation.py --model models/best_model_with_players_timeaware_stacking.pkl
   python run_simulation.py --model models/best_model_with_players_random_stacking.pkl
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

Models are trained with player-specific features to improve prediction accuracy. The time-aware pipeline adds chronological splitting, probability calibration (Platt/sigmoid), and ELO-based features for better probability quality and tournament realism. A stacked meta-learner can combine multiple base models via OOF predictions and a LogisticRegression meta model.

### Selecting a model at runtime

Use the `--model` flag to choose a specific model file for simulations without changing code:

```bash
# Time-aware calibrated ensemble best
python run_simulation.py --model models/best_model_with_players_timeaware.pkl

# Random-split ensemble best
python run_simulation.py --model models/best_model_with_players_random.pkl

# Calibrated XGBoost baseline
python run_simulation.py --model models/calibrated_xgboost_with_players.pkl
```

Typical model files:
- `models/best_model_with_players_timeaware.pkl` — best time-aware calibrated model
- `models/best_model_with_players_random.pkl` — best model from random-split pipeline
- `models/best_model_with_players_timeaware_stacking.pkl` — stacked meta-learner (time-aware)
- `models/best_model_with_players_random_stacking.pkl` — stacked meta-learner (random)
- `models/calibrated_xgboost_with_players.pkl` — calibrated XGBoost (time-aware trainer)
- `models/volleyball_predictor_with_players_uncalibrated.pkl` — raw XGBoost (no calibration)

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
- `multi_model_tournament_simulation.py` - Baseline multi-model (random split). Flags: `--save-summary`, `--summary-file`
- `multi_model_tournament_simulation_timeaware.py` - Time-aware + calibrated + ELO. Flags: `--save-summary`, `--summary-file`
- `tournament_visualization.py` - Generate visualizations

## Documentation

See the `docs/` folder for detailed documentation:
- Project overview and methodology
- Data quality reports
- Model performance summaries
- Tournament format details

Tip: Use the `--save-summary` flag on multi-model scripts to create concise diffable summaries under `outputs/simulation_output_*.txt`.
