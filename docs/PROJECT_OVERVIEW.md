# 🏐 PVL Match Prediction System - Project Overview

## ✅ Project Status
The PVL Match Prediction System is a comprehensive end-to-end solution for predicting volleyball match outcomes. It includes data ingestion, feature engineering, machine learning modeling, tournament simulation, and a modern interactive dashboard.

### 🚀 Key Capabilities
- **Automated Data Ingestion**: Fetches match data directly from the PVL website.
- **Advanced Analytics**: Calculates 80+ features including player efficiency, team form, and ELO ratings.
- **Machine Learning**: Uses XGBoost with time-aware validation and probability calibration.
- **Tournament Simulation**: Simulates entire tournaments (Preliminary -> Playoffs) to predict champions.
- **Interactive Dashboard**: A Next.js web application to visualize stats, predictions, and brackets.

---

## 📁 System Architecture

### 1. Data Pipeline (`scripts/`)
The core logic resides in the `scripts/` directory, handling everything from raw data to model predictions.

| Script | Description |
|--------|-------------|
| **`fetch_all_matches.py`** | **Data Ingestion**. Scrapes match IDs and downloads XML data from the PVL dashboard. |
| **`parse_volleyball_data.py`** | **Parsing**. Converts raw XML into structured JSON and stores it in the SQLite database. |
| **`feature_engineering_with_players.py`** | **Feature Engineering**. Calculates stats, efficiency metrics, and ELO ratings. Generates training datasets. |
| **`train_xgboost_with_players.py`** | **Modeling**. Trains the XGBoost model using time-aware splits and calibrates probabilities. |
| **`simulate_tournament.py`** | **Simulation**. Simulates the remaining matches of a tournament using the trained model to predict final standings. |
| **`export_dashboard_data.py`** | **Integration**. Exports processed data (matches, stats, predictions) to JSON for the dashboard. |
| **`database_manager.py`** | **Storage**. Manages the SQLite database schema and operations. |

### 2. Dashboard (`dashboard/`)
A modern web interface built with **Next.js**, **Tailwind CSS**, and **Shadcn UI**.

- **Home (`/`)**: Tournament overview, recent results, and upcoming match predictions.
- **Teams (`/teams`)**: Detailed team statistics, rosters, and performance metrics.
- **Players (`/players`)**: Top player rankings and individual statistics.
- **History (`/history`)**: Historical match results and model performance tracking.

### 3. Data Storage
- **`data/volleyball_data.db`**: SQLite database storing all relational data (teams, players, matches, stats).
- **`data/volleyball_matches.json`**: Raw parsed match data.
- **`dashboard/public/data.json`**: Exported data consumed by the frontend.

---

## 🔄 End-to-End Workflow

### Step 1: Ingest Data
Download the latest match results from the PVL website.
```bash
python scripts/fetch_all_matches.py
```

### Step 2: Process & Train
Parse the data, engineer features, and retrain the model.
```bash
python scripts/batch_processor.py
```
*Note: `batch_processor.py` orchestrates parsing, feature engineering, and training.*

### Step 3: Simulate Tournament
Run Monte Carlo simulations to predict the tournament outcome based on current standings.
```bash
python scripts/simulate_tournament.py
```

### Step 4: Update Dashboard
Export the latest data and predictions to the dashboard.
```bash
python scripts/export_dashboard_data.py
```

### Step 5: Launch Dashboard
Start the local web server to view the results.
```bash
cd dashboard
npm run dev
```
Visit `http://localhost:3000` to explore the insights.

---

## 📊 Data & Features

### Extracted Features (80+)
- **Skills**: Attack, Block, Serve, Reception, Dig, Set (Efficiency & Error rates).
- **Context**: Home/Away (nominal), Set number, Match duration.
- **Advanced**:
    - **ELO Ratings**: Chronological team strength tracking.
    - **Form**: Performance over the last N matches.
    - **H2H**: Head-to-head historical records.

### Database Schema
- `tournaments`: Conference details.
- `teams`: Team metadata.
- `players`: Roster information.
- `matches`: Match results and metadata.
- `team_match_stats`: Aggregated stats per match.
- `set_scores`: Set-by-set scores.

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **ML Framework**: XGBoost, Scikit-learn
- **Database**: SQLite
- **Frontend**: Next.js 14, React, Tailwind CSS, Recharts
- **Utilities**: Pandas, NumPy, BeautifulSoup4

## 📞 Next Steps
- [ ] **Advanced Metrics**: Implement rotation-based analysis.
- [ ] **Real-time**: Integrate live score updates.
- [ ] **Deployment**: Deploy the dashboard to Vercel/Netlify.
