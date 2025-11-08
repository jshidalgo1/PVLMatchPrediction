# Volleyball AI Project
Player-aware volleyball match analytics and tournament simulation. Parses PVL XML match files (2023–2025), stores structured team, player, lineup, and statistics data in SQLite, engineers ELO + historical + player performance features, trains calibrated ML models, and simulates tournaments.

## Project Structure

```
VolleyballAIProject/
├── data/
│   ├── xml_files/                  # Raw PVL XML match files (500+)
│   ├── databases/                  # SQLite DB (volleyball_data.db)
│   └── csv_files/                  # Engineered feature matrices
├── models/                         # Trained model artifacts
│   ├── best_model_with_players_timeaware.pkl
│   ├── best_model_with_players_timeaware_stacking.pkl
│   ├── best_model_with_players_random.pkl
│   ├── best_model_with_players_random_stacking.pkl
│   ├── calibrated_xgboost_with_players.pkl
│   └── volleyball_predictor_with_players_uncalibrated.pkl
├── outputs/                        # Simulation summaries, visuals
├── scripts/
│   ├── batch_processor.py          # XML → JSON → DB → player/ELO features
│   ├── parse_volleyball_data.py    # XML parsing (players, stats, lineups)
│   ├── database_manager.py         # SQLite schema + load ops
│   ├── feature_engineering_with_players.py # Direct DB feature extraction
│   ├── train_xgboost_with_players.py       # Calibrated time-aware XGBoost
│   ├── simulate_tournament.py      # Full tournament simulation logic
│   ├── run_simulation.py           # Entry point that invokes simulate_tournament
│   ├── tournament_visualization.py # Bracket / standings plots
│   └── archive/                    # Archived legacy & multi-model scripts
├── docs/                           # Reports & methodology
└── certificates/                   # SSL materials (if any)
```

## Player-Aware Pipeline (Recommended)

1. Ensure XML files are placed under `data/xml_files/`.
2. Run the batch processor:
   ```bash
   python scripts/batch_processor.py
   ```
   This builds `volleyball_data.db` and creates:
   - `volleyball_features_with_players.csv`
   - `X_features_with_players.csv`
   - `y_target_with_players.csv`
3. Train calibrated XGBoost:
   ```bash
   python scripts/train_xgboost_with_players.py
   ```
4. Simulate a tournament with a chosen model:
   ```bash
   python run_simulation.py --model models/best_model_with_players_timeaware.pkl
   ```

## Key Features Engineered

- ELO pre-match ratings & win probability (no leakage)
- Team historical aggregates (attack, block, serve, points, win rate)
- Player aggregates (starter attack/block/serve averages, libero digs/reception, top scorer attack, roster depth, sets per player, count of 10+ scorers)
- Chronological (time-aware) splitting for realistic evaluation & calibration

## Models & Artifacts

| Artifact | Description |
|----------|-------------|
| `best_model_with_players_timeaware.pkl` | Best calibrated ensemble (time-aware CV + ELO) |
| `best_model_with_players_timeaware_stacking.pkl` | Stacked meta-learner (logistic meta over calibrated bases) |
| `calibrated_xgboost_with_players.pkl` | Single calibrated XGBoost baseline |
| `best_model_with_players_random.pkl` | Legacy random-split ensemble (benchmark only) |
| `best_model_with_players_random_stacking.pkl` | Random-split stacked variant |
| `volleyball_predictor_with_players_uncalibrated.pkl` | Raw XGBoost (pre-calibration) |

### Selecting a Model for Simulation

Run with `--model` referencing any calibrated artifact:
```bash
python run_simulation.py --model models/best_model_with_players_timeaware.pkl
python run_simulation.py --model models/calibrated_xgboost_with_players.pkl
```
Legacy benchmark (random split) and stacking variants are retained only in `models/` for historical comparison; their training scripts are archived.

## Data Sources

- Philippine Volleyball League (PVL) XML exports (seasons 2023–2025)
- Parsed into normalized tables: tournaments, teams, matches, team_match_stats, player_match_stats, match_lineups, set_scores

## Calibration & Evaluation

- Time-aware blocked k-fold CV on chronological training window
- Separate calibration slice (sigmoid / Platt scaling) for probability reliability
- Holdout test set for final evaluation
- Metrics: Accuracy, Log Loss, Brier Score, ROC AUC, Confusion Matrix, Classification Report

## Typical Commands

```bash
# Full pipeline + train + simulate
python scripts/batch_processor.py
python scripts/train_xgboost_with_players.py
python run_simulation.py --model models/best_model_with_players_timeaware.pkl
```

## Archived Scripts

Multi-model experimentation, partial round simulations, and earlier non-player-aware utilities now live in `scripts/archive/` to keep the active surface minimal. They can be restored if ensemble benchmarking is revisited.

## Contributing / Next Steps

- Add player ID linkage (map jersey → persistent player_id) for longitudinal per-player projections.
- Introduce set-level granular features (momentum, clutch points) for richer predictive signals.
- Reliability diagrams & calibration drift checks over time.
- Export model predictions with confidence intervals.

## License

Internal/research use. Ensure PVL data usage complies with source terms.

---
For detailed methodology, see `docs/PROJECT_OVERVIEW.md` and `docs/FINAL_MODEL_SUMMARY.md`.
<!-- Removed duplicated legacy structure section to reduce noise -->
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
