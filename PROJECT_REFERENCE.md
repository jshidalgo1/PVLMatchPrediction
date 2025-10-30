# Volleyball AI Project - File Reference Guide

This document provides a comprehensive overview of the project structure and the purpose of each file. Use this as a reference for understanding the codebase.

## Project Overview

This project analyzes Philippine Volleyball League (PVL) match data using machine learning to predict match outcomes and simulate tournament results. It processes XML match files, extracts features, trains models, and simulates tournament progression. The codebase now includes a time-aware, calibrated pipeline with ELO features and an optional stacked meta-learner, alongside the original random-split baseline pipeline.

---

## Directory Structure

### `/data/` - Data Storage
All data files including raw match data, processed datasets, and databases.

#### `/data/xml_files/` - Raw Match Data
- **Purpose**: Contains official PVL match data in XML format
- **File Pattern**: `PVL{YEAR}{CONFERENCE}-{WEEK}-{TEAM1}-{TEAM2}-XML.xml`
- **Content**: Complete match statistics including:
  - Player rosters and statistics
  - Set-by-set scores and details
  - Team performance metrics
  - Match metadata (date, venue, referees)
- **Example**: `PVL2025D-W06-AKA-CCS-XML.xml` - Match between AKARI (AKA) and CREAMLINE (CCS)

#### `/data/databases/` - SQLite Database
- **File**: `volleyball_data.db`
- **Purpose**: Structured storage of match data with relational tables
- **Tables**:
  - `tournaments` - Tournament information
  - `teams` - Team details
  - `matches` - Match results and metadata
  - `players` - Player information
  - `player_statistics` - Per-player, per-match stats

#### `/data/csv_files/` - Processed Datasets
- **`volleyball_features.csv`** - Complete feature dataset with metadata
- **`volleyball_features_with_players.csv`** - Features including player-level stats
- **`X_features.csv`** - ML feature matrix (input features)
- **`X_features_with_players.csv`** - ML features with player stats
- **`y_target.csv`** - ML target labels (match outcomes)
- **`y_target_with_players.csv`** - Target labels for player-based models
- **`feature_importance_with_players.csv`** - Feature importance rankings

#### `/data/` - JSON Files
- **`volleyball_matches.json`** - Parsed match data in JSON format
  - Contains all match information extracted from XML files
  - Used as intermediate format between XML parsing and database insertion

---

## `/scripts/` - Core Python Scripts

### Data Processing Pipeline

#### `parse_volleyball_data.py` - XML Parser
**Purpose**: Parses XML match files and extracts structured data
**Class**: `VolleyballDataParser`
**Key Methods**:
- `parse_xml_file()` - Parse single XML file
- `parse_multiple_files()` - Batch parse XML files
- `save_to_json()` - Export parsed data to JSON
**Output**: JSON representation of match data

#### `batch_processor.py` - Complete Processing Pipeline
**Purpose**: Orchestrates the entire data processing workflow
**Pipeline Steps**:
1. Find and parse all XML files
2. Save parsed data to JSON
3. Create SQLite database and load data
4. Generate machine learning features
5. Prepare datasets for training
**Usage**: `python scripts/batch_processor.py`
**When to Run**: 
- After adding new XML files
- When XML files are modified
- To rebuild the entire database

#### `database_manager.py` - Database Operations
**Purpose**: Manages SQLite database operations
**Class**: `VolleyballDatabase`
**Key Methods**:
- `create_schema()` - Create database tables
- `load_from_json()` - Import data from JSON
- `get_summary()` - Database statistics
- `query_*()` - Various query methods for data retrieval
**Features**:
- Handles tournament, team, match, and player data
- Prevents duplicate entries
- Provides aggregation queries

#### `feature_engineering.py` - Feature Generation
**Purpose**: Creates machine learning features from raw match data
**Class**: `VolleyballFeatureEngineer`
**Generated Features**:
- Team performance metrics (attack efficiency, block effectiveness)
- Historical win rates and streaks
- Head-to-head statistics
- Set differential patterns
- Momentum indicators
**Output**: Feature matrices ready for ML training

#### `feature_engineering_with_players.py` - Player-Based + ELO Features
**Purpose**: Generates features that include player-level statistics
**Enhanced Features**:
- Player contribution metrics
- Star player performance
- Team roster strength
- Player form and consistency
 - Chronological ELO ratings per team (1500 base, K=20) and derived features (`team_a_elo`, `team_b_elo`, `elo_diff`, `elo_prob_team_a`) without leakage
**Use Case**: More detailed predictions considering individual player impact

### Model Training Scripts

#### `train_xgboost.py` - XGBoost Model Training
**Purpose**: Train gradient boosting model for match prediction
**Model**: XGBoost Classifier
**Features**:
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Model persistence (saves trained model)
**Output**: Trained XGBoost model file

#### `train_xgboost_with_players.py` - Player-Aware XGBoost
**Purpose**: Train XGBoost model with player features
**Difference**: Incorporates player-level statistics for improved accuracy
**Output**: Enhanced model with player considerations

#### `train_naive_bayes.py` - Baseline Model
**Purpose**: Train simple Naive Bayes classifier as baseline
**Use Case**: Quick baseline for comparison with more complex models

### Simulation and Prediction Scripts

#### `simulate_remaining_matches.py` - First Round Simulation
**Purpose**: Predict remaining first-round matches and calculate standings
**Features**:
- Loads trained model
- Identifies played vs. remaining matches
- Predicts match outcomes
- Calculates pool standings
- Determines playoff qualifiers
**Output**: Tournament standings and playoff bracket seeds

#### `simulate_second_round.py` - Playoff Simulation
**Purpose**: Simulates playoff rounds (quarterfinals, semifinals, finals)
**Tournament Format**:
- Single elimination
- Best-of-3 or Best-of-5 series
- Seeding based on first round performance

#### `simulate_tournament.py` - Complete Tournament Simulation
**Purpose**: Full tournament simulation from start to finish
**Scope**: First round + playoffs in one execution

#### `multi_model_tournament_simulation.py` - Baseline Multi-Model (Random Split)
**Purpose**: Train multiple models on a stratified random split and simulate tournaments
**Models**: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, Voting (soft), Stacking (OOF + LogisticRegression)
**CLI**:
- `--save-summary` to write a concise summary to `outputs/simulation_output_random.txt`
- `--summary-file PATH` to override the summary location
**Artifacts**:
- `models/best_model_with_players_random.pkl` (best by test accuracy)
- `models/best_model_with_players_random_stacking.pkl` (stacked meta-learner)

#### `multi_model_tournament_simulation_timeaware.py` - Time-aware + Calibrated + ELO
**Purpose**: Chronological training/validation with probability calibration and ELO features
**Validation**:
- Chronological split: sub-train → calibration → test (last 20%)
- Blocked time-ordered CV on the training window
**Calibration**: Platt (sigmoid) via `CalibratedClassifierCV(cv='prefit')`
**Models**: Same as baseline, with calibrated probabilities; includes Voting and Stacking
**CLI**:
- `--save-summary` to write `outputs/simulation_output_timeaware.txt`
- `--summary-file PATH` to override the summary location
**Artifacts**:
- `models/best_model_with_players_timeaware.pkl` (best time-aware calibrated)
- `models/best_model_with_players_timeaware_stacking.pkl` (stacked meta-learner)

#### `complete_tournament_simulation.py` - Comprehensive Simulation
**Purpose**: End-to-end tournament simulation with detailed output
**Features**:
- Complete bracket simulation
- Confidence intervals
- Scenario analysis
- Visual representation of results

### Utility Scripts

#### `config.py` - Configuration Settings
**Purpose**: Centralized configuration for paths and constants
**Contents**:
- File paths (XML directory, database location, CSV output)
- Tournament settings
- Team code mappings
- Model parameters

#### `validate_data.py` - Data Quality Checks
**Purpose**: Validate XML data integrity and consistency
**Checks**:
- XML structure validation
- Score consistency
- Required fields presence
- Statistical anomalies

#### `download_matches.py` - Data Acquisition
**Purpose**: Download match XML files from PVL sources
**Note**: May require authentication or specific access

#### `fetch_all_matches.py` - Batch Download
**Purpose**: Automated batch downloading of multiple matches

#### `fix_test_pvlr25_data.py` - Data Corrections
**Purpose**: Apply specific fixes to test tournament data
**Use Case**: Correcting known data issues (e.g., score discrepancies)

#### `tournament_visualization.py` - Visual Analytics
**Purpose**: Generate visualizations of tournament data
**Outputs**:
- Team performance charts
- Win probability distributions
- Feature importance plots
- Bracket visualizations

#### `predict_tournament.py` - One-off Predictions
**Purpose**: Make predictions for specific matches or scenarios
**Use Case**: Quick predictions without full simulation

---

## Root Directory Files

### `run_simulation.py` - Main Entry Point
**Purpose**: Primary script to run tournament simulations
**Workflow**:
1. Loads best available model
2. Identifies tournament state
3. Simulates remaining matches
4. Outputs predictions and standings
**Usage**: `python run_simulation.py`
**Options**:
- `--model PATH` to select a specific saved model artifact (e.g., time-aware stacked)

### `QUICK_START.md` - Getting Started Guide
**Purpose**: Quick start instructions for new users
**Contents**:
- Setup instructions
- Basic usage examples
- Common workflows

### `README.md` - Project Documentation
**Purpose**: Main project documentation
**Contents**:
- Project overview
- Installation instructions
- Usage guide
- Feature descriptions

---

## `/docs/` - Documentation

### `PROJECT_OVERVIEW.md`
Comprehensive project documentation covering architecture, methodology, and design decisions.

### `FINAL_MODEL_SUMMARY.md`
Summary of final model performance, including:
- Model selection rationale
- Performance metrics
- Feature importance
- Known limitations

### `DATA_QUALITY_REPORT.md`
Data quality assessment including:
- Data completeness
- Known issues
- Data cleaning procedures
- Validation results

### `tournament_format.md`
Detailed explanation of PVL tournament structure:
- Pool play format
- Playoff structure
- Tiebreaker rules
- Advancement criteria

---

## `/models/` - Trained Models
Contains saved machine learning models:
- **XGBoost models**: `.model` or `.pkl` files
- **Feature scalers**: Preprocessing transformations
- **Model metadata**: Performance metrics and parameters

### `/models/catboost_info/`
CatBoost-specific training logs and metadata (if using CatBoost)

---

## `/outputs/` - Simulation Results

### `simulation_output.txt`
Latest simulation results including:
- Match predictions
- Confidence scores
- Final standings
- Playoff bracket

### `simulation_output_random.txt`
Summary from baseline random-split multi-model run (`--save-summary`)

### `simulation_output_timeaware.txt`
Summary from time-aware + calibrated + ELO multi-model run (`--save-summary`)

### `requirements.txt`
Python package dependencies needed to run the project

---

## Common Workflows

### 1. Adding New Match Data
```bash
# 1. Add XML files to data/xml_files/
# 2. Run batch processor
python scripts/batch_processor.py

# 3. (Optional) Retrain models
python scripts/train_xgboost_with_players.py

# 4. Run simulation
python run_simulation.py
```

### 2. Correcting Match Data
```bash
# 1. Edit XML file directly
# 2. Update database
python scripts/batch_processor.py

# OR update database directly with SQL
sqlite3 data/databases/volleyball_data.db "UPDATE matches SET ..."

# 3. Re-run simulation
python run_simulation.py
```

### 3. Training New Models
```bash
# Generate features
python scripts/batch_processor.py

# Train model
python scripts/train_xgboost_with_players.py

# Validate
python scripts/validate_data.py
```

### 4. Running Tournament Predictions
```bash
# Quick simulation
python run_simulation.py

# Multi-model ensemble
python scripts/multi_model_tournament_simulation.py

# Multi-model (time-aware + calibrated + ELO)
python scripts/multi_model_tournament_simulation_timeaware.py --save-summary

# Complete with visualization
python scripts/complete_tournament_simulation.py
```

---

## Key Data Formats

### Team Codes
- **AKA** - AKARI Chargers
- **CCS** - Creamline Cool Smashers
- **PGA** - Petro Gazz Angels
- **CHD** - Chery Tiggo Crossovers
- **CMF** - Choco Mucho Flying Titans
- **ZUS** - ZUS Coffee Thunderbelles
- **HSH** - PLDT High Speed Hitters
- **FFF** - F2 Logistics Cargo Movers (formerly Foton)
- **CTC** - Capital1 Solar Spikers (formerly Cignal)
- **GTH** - Galeries Tower Highrisers
- **CAP** - Farm Fresh Foxies
- **NXL** - NXLED Chameleons

### Tournament Codes
- **PVL{YEAR}A** - All-Filipino Conference
- **PVL{YEAR}B** - Open Conference  
- **PVL{YEAR}C** - Invitational Conference
- **PVL{YEAR}D** - Reinforced Conference
- **TEST_PVLR25** - 2025 Reinforced Conference Test Data

---

## Feature Categories

### Team Performance Features
- `attack_efficiency` - Attack success rate
- `block_points_per_set` - Blocking effectiveness
- `service_aces_per_set` - Serving strength
- `reception_quality` - Passing consistency
- `dig_effectiveness` - Defense quality

### Historical Features
- `win_rate_last_5` - Recent form
- `head_to_head_record` - Historical matchup results
- `set_differential` - Average point margin

### Contextual Features
- `days_since_last_match` - Rest days
- `home_away` - Venue advantage (if applicable)
- `tournament_phase` - Pool play vs. playoffs

---

## Model Information

### Pipelines and Artifacts
- Baseline (random-split): `models/best_model_with_players_random.pkl`
- Baseline stacked: `models/best_model_with_players_random_stacking.pkl`
- Time-aware calibrated: `models/best_model_with_players_timeaware.pkl`
- Time-aware stacked: `models/best_model_with_players_timeaware_stacking.pkl`
- Calibrated XGBoost: `models/calibrated_xgboost_with_players.pkl`

Use `run_simulation.py --model <path>` to pick a specific artifact at runtime.
- **Features**: 18 team-based features + player features
- **Training Data**: 501 matches from PVL 2023-2025

### Model Files
- Located in `/models/` directory
- Loaded automatically by simulation scripts
- Updated when retraining

---

## Database Schema

### `tournaments`
- `id`, `code`, `name`, `year`, `conference_type`

### `teams`
- `id`, `code`, `name`, `tournament_id`

### `matches`
- `id`, `tournament_id`, `team_a_id`, `team_b_id`
- `team_a_sets_won`, `team_b_sets_won`, `winner_id`
- `match_date`, `venue`, `phase`

### `players`
- `id`, `first_name`, `last_name`, `jersey_number`, `team_id`

### `player_statistics`
- Links players to match performance
- Includes all volleyball statistics (attacks, blocks, serves, etc.)

---

## Troubleshooting

### Issue: Simulation shows old data
**Solution**: Run `python scripts/batch_processor.py` to reload data from XML files

### Issue: Database out of sync with XML
**Solution**: Delete `data/databases/volleyball_data.db` and re-run batch processor

### Issue: Model predictions seem inaccurate
**Solution**: Retrain model with latest data: `python scripts/train_xgboost_with_players.py`

### Issue: Missing matches in simulation
**Solution**: Check that XML files are in `data/xml_files/` and properly formatted

---

## Recent Changes Log

### Latest Update: AKA vs CCS Match Correction
- **File Modified**: `PVL2025D-W06-AKA-CCS-XML.xml`
- **Change**: Corrected set count from 4-1 to 3-2
- **Sets Changed**: 
  - AKA Set 4: 25 → 23
  - CCS Set 4: 23 → 25
- **Database Updated**: Match record corrected in `volleyball_data.db`
- **Impact**: More accurate historical data for predictions

---

## Development Notes

- All scripts use UTF-8 encoding for international character support
- XML parsing handles ISO-8859-1 encoded files
- Database uses SQLite for portability
- Models are pickle-serialized for persistence
- Feature engineering is modular for easy extension

---

## Contact & Maintenance

For issues with specific files or functionality, refer to the individual script headers which contain detailed docstrings and usage examples.

Last Updated: October 30, 2025
