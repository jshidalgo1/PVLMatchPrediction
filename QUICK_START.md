# Quick Start Guide

Get up and running with the PVL Match Prediction system in minutes.

## Prerequisites

- Python 3.13+
- Node.js 18+ and npm (for dashboard)
- 2GB free disk space

## Installation

### 1. Clone and Setup Python Environment

```bash
git clone https://github.com/yourusername/PVLMatchPrediction.git
cd PVLMatchPrediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Dashboard (Optional)

```bash
cd dashboard
npm install
cd ..
```

## Complete Workflow

### Step 1: Fetch Match Data

Download PVL XML files:

```bash
# Download latest 50 matches
python scripts/fetch_all_matches.py --download --limit 50

# Or download specific tournament
python scripts/fetch_all_matches.py --download --tournament PVL2025D --week-range 46-48
```

Files saved to `data/xml_files/`.

### Step 2: Process Data

Parse XML and build database:

```bash
python scripts/batch_processor.py
```

This creates:
- `data/databases/volleyball_data.db` - SQLite database
- `data/csv_files/X_features.csv` - ML features
- `data/csv_files/y_target.csv` - Target labels

**What it does**:
- Parses XML match files
- Maps players to jersey numbers
- Calculates accurate sets_played from roster data
- Extracts team and player statistics
- Engineers features for ML

### Step 3: Train Model

Train calibrated XGBoost model:

```bash
python scripts/train_xgboost_with_players.py
```

Output: `models/calibrated_xgboost_with_players.pkl`

**Expected Results**:
- Accuracy: ~72-74%
- AUC: ~0.81
- Training time: 1-2 minutes

### Step 4: Simulate Tournament

Run tournament simulation:

```bash
python scripts/simulate_tournament.py --save_outputs --champion_analysis
```

Generates:
- Playoff bracket (QF, SF, Championship, 3rd Place)
- Confidence scores for each match
- Output saved to `outputs/tournament_simulation_YYYYMMDD_HHMMSS.json`

### Step 5: Export to Dashboard

Prepare data for visualization:

```bash
python scripts/export_dashboard_data.py
```

Creates: `dashboard/public/data.json`

### Step 6: View Dashboard

```bash
cd dashboard
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

You'll see:
- Tournament standings (current & projected)
- Player statistics (sortable by skill)
- Match predictions by phase
- Playoff bracket with all matchups

## Quick Commands Reference

```bash
# List available XML files without downloading
python scripts/fetch_all_matches.py --list

# Download with rate limiting
python scripts/fetch_all_matches.py --download --delay 0.5

# Process all XML files
python scripts/batch_processor.py

# Train model
python scripts/train_xgboost_with_players.py

# Simulate (without saving)
python scripts/simulate_tournament.py

# Simulate and save outputs
python scripts/simulate_tournament.py --save_outputs

# Export to dashboard
python scripts/export_dashboard_data.py

# Run dashboard (development)
cd dashboard && npm run dev

# Build dashboard (production)
cd dashboard && npm run build && npm start
```

## File Locations

After running the complete workflow:

```
PVLMatchPrediction/
├── data/
│   ├── xml_files/              # Downloaded XML files
│   ├── databases/
│   │   └── volleyball_data.db  # SQLite database
│   └── csv_files/
│       ├── X_features.csv      # ML features
│       └── y_target.csv        # Target labels
├── models/
│   └── calibrated_xgboost_with_players.pkl  # Trained model
├── outputs/
│   └── tournament_simulation_*.json  # Simulation results
└── dashboard/
    └── public/
        └── data.json           # Dashboard data
```

## Configuration

All paths configured in `scripts/config.py`. Defaults should work for most cases.

To customize:

```python
# scripts/config.py
DATA_DIR = Path(__file__).parent.parent / "data"
XML_FILES_DIR = DATA_DIR / "xml_files"
DB_FILE_STR = str(DATA_DIR / "databases" / "volleyball_data.db")
# ... etc
```

## Troubleshooting

### "No matches found"
- Download XML files first: `python scripts/fetch_all_matches.py --download`
- Check `data/xml_files/` is not empty

### "Model training failed - not enough data"
- Need at least 100 matches for reasonable accuracy
- Download more: `python scripts/fetch_all_matches.py --download --limit 100`

### "Player stats not showing in dashboard"
- Verify database has player_match_stats: `sqlite3 data/databases/volleyball_data.db "SELECT COUNT(*) FROM player_match_stats WHERE player_id IS NOT NULL;"`
- Re-process if needed: `python scripts/batch_processor.py`
- Re-export: `python scripts/export_dashboard_data.py`

### "Dashboard shows old data"
- Re-export after simulation: `python scripts/export_dashboard_data.py`
- Hard refresh browser: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows)

### "Playoff bracket empty"
- Run simulation with --save_outputs: `python scripts/simulate_tournament.py --save_outputs`
- Check outputs/ directory has simulation file
- Re-export: `python scripts/export_dashboard_data.py`

## Next Steps

1. **Add More Data**: Download additional matches for better predictions
   ```bash
   python scripts/fetch_all_matches.py --download --year 2024 --limit 200
   python scripts/batch_processor.py
   python scripts/train_xgboost_with_players.py
   ```

2. **Experiment with Models**: Try different parameters in `train_xgboost_with_players.py`

3. **Customize Dashboard**: Edit `dashboard/src/app/page.tsx` for custom features

4. **Read Full Documentation**:
   - [Project Structure](README.md)
   - [Dashboard Guide](docs/DASHBOARD_GUIDE.md)
   - [Data Pipeline](docs/DATA_PIPELINE.md)
   - [Player Statistics](docs/PLAYER_STATISTICS.md)

## Common Workflows

### Update Predictions After New Matches

```bash
# 1. Download new matches
python scripts/fetch_all_matches.py --download --year 2025 --limit 20

# 2. Update database
python scripts/batch_processor.py

# 3. Retrain model (optional, if significant new data)
python scripts/train_xgboost_with_players.py

# 4. Run fresh simulation
python scripts/simulate_tournament.py --save_outputs

# 5. Update dashboard
python scripts/export_dashboard_data.py
```

### Compare Models

```bash
# Compare two model artifacts
python scripts/compare_metrics.py \
  models/calibrated_xgboost_with_players.pkl \
  models/best_model_with_players_timeaware.pkl

# Results saved to outputs/metrics_comparison_*.md
```

### Database Queries

```bash
# Open database
sqlite3 data/databases/volleyball_data.db

# Example queries
SELECT * FROM tournaments;
SELECT * FROM teams;
SELECT COUNT(*) FROM matches;
SELECT full_name, SUM(sets_played) FROM player_match_stats pms JOIN players p ON pms.player_id = p.id GROUP BY p.id ORDER BY SUM(sets_played) DESC LIMIT 10;
```

---

For detailed documentation, see `docs/` directory.

**Last Updated**: November 23, 2025
