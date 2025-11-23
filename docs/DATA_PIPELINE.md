# Data Pipeline Documentation

Complete documentation of the PVL Match Prediction data processing pipeline, from XML file fetching to dashboard export.

## Pipeline Overview

```
┌──────────────────┐
│  fetch_all_      │
│  matches.py      │──┐
└──────────────────┘  │
                      ▼
┌──────────────────────────────────────┐
│  XML Files (data/xml_files/)         │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────┐  ┌──────────────────┐
│  parse_          │  │  database_       │
│  volleyball_     │─▶│  manager.py      │
│  data.py         │  └──────────────────┘
└──────────────────┘           │
         │                     ▼
         │         ┌───────────────────────┐
         │         │  SQLite Database      │
         │         │  volleyball_data.db   │
         │         └───────────────────────┘
         │                     │
         ▼                     ▼
┌──────────────────────────────────────┐
│  feature_engineering_                │
│  with_players.py                     │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  CSV Files (X_features, y_target)    │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────┐
│  train_xgboost_  │
│  with_players.py │
└──────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Model (.pkl)                        │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────┐
│  simulate_       │
│  tournament.py   │
└──────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  Simulation Output (JSON)            │
└──────────────────────────────────────┘
                      │
                      ▼
┌──────────────────┐
│  export_         │
│  dashboard_      │
│  data.py         │
└──────────────────┘
                      │
                      ▼
┌──────────────────────────────────────┐
│  dashboard/public/data.json          │
└──────────────────────────────────────┘
```

## 1. Data Acquisition

### fetch_all_matches.py

**Purpose**: Download PVL XML match files from official dashboard

**Features**:
- List available files without downloading
- Filter by year, tournament, week range
- Batch download with rate limiting
- Skip existing files

**Usage**:
```bash
# List all available files
python scripts/fetch_all_matches.py --list

# Download specific tournament
python scripts/fetch_all_matches.py --download --tournament PVL2025D

# Download date range
python scripts/fetch_all_matches.py --download --week-range 46-50

# Download with rate limiting
python scripts/fetch_all_matches.py --download --delay 0.5 --limit 100
```

**Implementation Details**:
- Base URL: `https://dashboard.pvl.ph/assets/match_results/xml/`
- File format: `TOURNAMENTCODE-WXX-TTTvTTT-XML.xml`
- Fetches index page to discover available files
- Parses links matching pattern
- Downloads to `data/xml_files/`

## 2. Data Parsing

### parse_volleyball_data.py

**Purpose**: Parse XML match files into structured Python data

**Key Functions**:

#### parse_xml_file()
Extracts from XML:
- Tournament info (name, code, season)
- Team data (name, coach, roster)
- Match metadata (date, location, phase)
- Set scores
- Player statistics by set

#### _extract_player_match_stats()
**Critical Enhancement**: Sets Played Calculation

Previously calculated `sets_played` by checking if player had any stats in a set:
```python
# OLD - Inaccurate
if len(player.attrib) > 1:  # Has any stat
    stats.sets_played += 1
```

**New Approach - Accurate**:
Parses `<Roster>` tag for each set:
```python
# NEW - Accurate
for set_element in match_element.findall('.//Set'):
    roster_element = set_element.find('Roster')
    if roster_element is not None:
        # Extract all jersey numbers present
        starters = [roster_element.get(f'p{i}') for i in range(1, 7)]
        substitutes = [roster_element.get(f'r{i}') for i in range(1, 7)]
        liberos = [roster_element.get(f'l{i}') for i in range(1, 3)]
        
        # Count appearances
        all_jerseys = set(filter(None, starters + substitutes + liberos))
        for jersey in all_jerseys:
            if jersey == player_jersey:
                sets_played += 1
```

**Why This Matters**:
- Player can be on court without recording a stat
- Liberos especially affected (no attack/block points)
- Enables accurate per-set average calculations
- Critical for defensive specialist metrics

#### Player-Jersey Mapping

**Challenge**: XML has two sources of player info:
1. Top-level `<Player>` tags with full names (no jersey numbers)
2. Team-level `<Player NoShirt="X">` tags with jersey numbers (no full names)

**Solution**: Positional mapping within team roster

```python
def _build_roster_map(team_element):
    # Get top-level names in order
    top_level_players = root.findall('.//Player[@code]')
    names = [p.get('text') for p in top_level_players]
    
    # Get team jerseys in order
    team_players = team_element.findall('.//Player[@NoShirt]')
    jerseys = [p.get('NoShirt') for p in team_players]
    
    # Map by position
    roster_map = {}
    for idx, jersey in enumerate(jerseys):
        if idx < len(names):
            roster_map[jersey] = names[idx]
    
    return roster_map
```

**Output**: `volleyball_matches.json` containing structured match data

## 3. Database Storage

### database_manager.py

**Purpose**: Store parsed data in SQLite for efficient querying

**Schema** (6 tables):

```sql
-- Tournament metadata
CREATE TABLE tournaments (
    id INTEGER PRIMARY KEY,
    code TEXT UNIQUE,
    name TEXT,
    season TEXT,
    year INTEGER
);

-- Team information
CREATE TABLE teams (
    id INTEGER PRIMARY KEY,
    code TEXT UNIQUE,
    name TEXT,
    coach TEXT,
    assistant_coach TEXT
);

-- Player roster
CREATE TABLE players (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    full_name TEXT UNIQUE
);

-- Match details
CREATE TABLE matches (
    id INTEGER PRIMARY KEY,
    tournament_id INTEGER,
    match_no TEXT,
    date TEXT,
    phase_no INTEGER,
    phase_description TEXT,
    team_a_id INTEGER,
    team_b_id INTEGER,
    team_a_sets_won INTEGER,
    team_b_sets_won INTEGER,
    winner_id INTEGER,
    status TEXT,
    FOREIGN KEY (tournament_id) REFERENCES tournaments(id),
    FOREIGN KEY (team_a_id) REFERENCES teams(id),
    FOREIGN KEY (team_b_id) REFERENCES teams(id),
    FOREIGN KEY (winner_id) REFERENCES teams(id)
);

-- Player match statistics
CREATE TABLE player_match_stats (
    id INTEGER PRIMARY KEY,
    match_id INTEGER,
    team_id INTEGER,
    player_id INTEGER,  -- Linked via name matching
    jersey_number INTEGER,
    is_starter BOOLEAN,
    is_libero BOOLEAN,
    sets_played INTEGER,  -- Calculated from Roster tags
    attack_points INTEGER,
    block_points INTEGER,
    serve_points INTEGER,
    dig_excellent INTEGER,
    reception_excellent INTEGER,
    set_excellent INTEGER,
    -- ... (additional stats)
    FOREIGN KEY (match_id) REFERENCES matches(id),
    FOREIGN KEY (team_id) REFERENCES teams(id),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Set scores
CREATE TABLE set_scores (
    id INTEGER PRIMARY KEY,
    match_id INTEGER,
    set_number INTEGER,
    team_a_score INTEGER,
    team_b_score INTEGER,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
```

**Player ID Linking**:

After inserting players by name, map back to jersey numbers:

```python
# Build name-to-ID map
cursor.execute("SELECT id, full_name FROM players WHERE full_name IN (...)")
name_to_id = {name: pid for pid, name in cursor.fetchall()}

# Update player_match_stats
for stat in player_stats:
    if stat['full_name'] in name_to_id:
        stat['player_id'] = name_to_id[stat['full_name']]
```

This enables joining player names with their statistics.

## 4. Feature Engineering

### feature_engineering_with_players.py

**Purpose**: Generate ML features from raw database data

**Feature Categories** (34 total):

1. **ELO Ratings** (3 features)
   - `team_a_elo`, `team_b_elo`
   - `elo_diff`, `elo_prob_team_a`
   
2. **Team Aggregates** (12 features)
   - Historical: wins, win_rate, matches_played
   - Points: avg_points, avg_attack, avg_block, avg_serve
   - Per team (A & B)

3. **Player Statistics** (18 features)
   - Starters: starter_avg_attack, starter_avg_block, starter_avg_serve
   - Top scorers: top_scorer_attack
   - Liberos: libero_avg_digs, libero_avg_reception
   - Roster: roster_depth, avg_sets_per_player
   - High performers: count_10plus_scorers
   - Per team (A & B)

4. **Match Context** (1 feature)
   - Head-to-head stats (deprecated in current model)

**ELO Calculation**:

```python
def _compute_current_elo(conn):
    K = 20  # K-factor
    DEFAULT_ELO = 1500
    
    elo = {}
    for match in chronological_matches:
        team_a_elo = elo.get(team_a, DEFAULT_ELO)
        team_b_elo = elo.get(team_b, DEFAULT_ELO)
        
        expected_a = 1 / (1 + 10 ** (-(team_a_elo - team_b_elo) / 400))
        score_a = 1 if winner == team_a else 0
        
        elo[team_a] = team_a_elo + K * (score_a - expected_a)
        elo[team_b] = team_b_elo + K * ((1 - score_a) - (1 - expected_a))
    
    return elo
```

**Output**: `data/csv_files/X_features.csv`, `y_target.csv`

## 5. Model Training

### train_xgboost_with_players.py

**Purpose**: Train calibrated prediction model

**Process**:
1. Load features from CSV
2. Time-aware chronological split (80/20)
3. Train XGBoost classifier
4. 4-fold time-blocked cross-validation
5. Platt scaling calibration on holdout window
6. Evaluate metrics (Accuracy, AUC, LogLoss, Brier, ECE, MCE)

**Output**: `models/calibrated_xgboost_with_players.pkl`

## 6. Tournament Simulation

### simulate_tournament.py

**Purpose**: Simulate complete tournament with FIVB rules

**Phases**:
1. Complete preliminary rounds (if incomplete)
2. Determine second-round pools (Groups C & D)
3. Complete second-round cross-pool matches
4. Rank top 8 by FIVB rules
5. Simulate quarterfinals (#1 vs #8, etc.)
6. Simulate semifinals
7. Simulate championship
8. **NEW**: Simulate third place match

**Third Place Match** (added Nov 2025):

```python
# Extract semifinal losers
sf_losers = []
for sf in sf_results:
    loser = sf['team_b'] if sf['winner'] == sf['team_a'] else sf['team_a']
    sf_losers.append(loser)

# Predict third place
third_winner, third_conf = predict_match(model, features, conn, elo_map, 
                                          sf_losers[0], sf_losers[1])

# Save to output
result['third_place'] = {
    'team_a': sf_losers[0],
    'team_b': sf_losers[1],
    'winner': third_winner,
    'confidence': third_conf
}
```

**Output**: `outputs/tournament_simulation_YYYYMMDD_HHMMSS.json`

## 7. Dashboard Export

### export_dashboard_data.py

**Purpose**: Consolidate all data for dashboard consumption

**Process**:

1. **Load Simulation Results**
   ```python
   # Find latest simulation file
   sim_files = sorted(Path('outputs').glob('tournament_simulation_*.json'), reverse=True)
   latest_sim = sim_files[0]
   
   # Load playoff bracket from file
   with open(latest_sim) as f:
       sim_data = json.load(f)
       playoffs = extract_playoffs(sim_data)
   ```

2. **Query Database**
   - Current standings
   - All matches with results
   - Player statistics (filtered by tournament)
   - Team information

3. **Aggregate Player Stats**
   ```python
   query = """
       SELECT 
           pms.player_id,
           SUM(pms.sets_played) as total_sets,
           SUM(pms.attack_points) as attack_total,
           SUM(pms.block_points) as block_total,
           SUM(pms.serve_points) as serve_total,
           SUM(pms.dig_excellent) as dig_total,
           SUM(pms.reception_excellent) as rec_total,
           SUM(pms.set_excellent) as set_total
       FROM player_match_stats pms
       JOIN matches m ON pms.match_id = m.id
       WHERE m.tournament_id = ? AND pms.player_id IS NOT NULL
       GROUP BY pms.player_id
   """
   ```

4. **Format for Dashboard**
   - Calculate total_points = attack + block + serve
   - Include all tournaments (multi-tournament support)
   - Add ISO timestamp

5. **Write JSON**
   ```python
   data = {
       'tournaments': tournaments_data,
       'players': players_list,
       'teams': teams_list,
       'last_updated': datetime.now().isoformat()
   }
   
   with open('dashboard/public/data.json', 'w') as f:
       json.dump(data, f, indent=2)
   ```

**Output**: `dashboard/public/data.json` (~500KB-2MB)

## Complete Workflow

Run entire pipeline:

```bash
# 1. Fetch latest matches
python scripts/fetch_all_matches.py --download --year 2025 --limit 50

# 2. Process all XML files
python scripts/batch_processor.py

# 3. Train model (if new data added)
python scripts/train_xgboost_with_players.py

# 4. Simulate tournament
python scripts/simulate_tournament.py --save_outputs --champion_analysis

# 5. Export to dashboard
python scripts/export_dashboard_data.py

# 6. View in browser
cd dashboard
npm run dev
```

## Data Quality Checks

### Validation Points

1. **XML Parsing**
   - Verify all expected tags present
   - Check for missing player names
   - Validate set scores add up correctly

2. **Database Integrity**
   ```sql
   -- Check for orphaned stats
   SELECT COUNT(*) FROM player_match_stats WHERE player_id IS NULL;
   
   -- Verify sets_played > 0 for players
   SELECT COUNT(*) FROM player_match_stats WHERE sets_played = 0;
   
   -- Check match consistency
   SELECT * FROM matches WHERE team_a_sets_won + team_b_sets_won != (
       SELECT COUNT(*) FROM set_scores WHERE match_id = matches.id
   );
   ```

3. **Feature Engineering**
   - Confirm no NaN values in features
   - Verify ELO ratings within reasonable range (1200-1800)
   - Check feature correlations

4. **Export Validation**
   - Confirm data.json is valid JSON
   - Check file size is reasonable
   - Verify tournament count matches database

## Troubleshooting

### Common Issues

**Issue**: Player stats not appearing in dashboard
- **Solution**: Check `player_id` not NULL in `player_match_stats`
- Run: `python scripts/batch_processor.py` to reprocess with updated parser

**Issue**: Sets played = 0 for some players
- **Solution**: XML missing Roster tags, or player not in any set roster
- Check raw XML file for completeness

**Issue**: Simulation missing third place
- **Solution**: Re-run simulation with latest code that includes third place calculation
- Run: `python scripts/simulate_tournament.py --save_outputs`

**Issue**: Export fails with "No module named 'scripts.simulate_tournament'"
- **Solution**: Ensure running from project root, not scripts/ directory
- Check Python path includes project root

---

**Last Updated**: November 23, 2025
